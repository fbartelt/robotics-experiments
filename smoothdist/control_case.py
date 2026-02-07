import numpy as np
import plotly.graph_objects as go
import plotly.colors as pc
import cyipopt

# from distances import signed_dist2convex, phi, smooth_min
from smoothfunctions import (
    holderMeanWithGradient,
    signedDist2Convex,
    smoothMinListWithGradient,
    smoothMin2ElementsGradient,
    smoothMin2ElementsWithGradient,
    smoothMinList,
    phi,
    ESDF_CGAL,
)
from typing import List, Tuple, Optional, Callable
from polygon import (
    Polytope,
    add_polygon,
    create_level_sets,
    NonConvexPolygon,
    generate_random_polyhedron,
    generate_random_polyhedron_set,
)
from polyhedron import add_polyhedron
from path_planning import (
    deform_path_ipopt,
    add_path,
    add_path3d,
    animate_deformation_matplotlib,
    show_animation,
)
from uaibot.utils import Utils


def add_point(fig, p, color="black", name="point"):
    fig.add_trace(
        go.Scatter(
            x=[p[0]],
            y=[p[1]],
            mode="markers",
            marker=dict(size=8, color=color),
            name=name,
        )
    )


def add_vector_at_point(fig, p, v, color="black", name="vector"):
    fig.add_trace(
        go.Scatter(
            x=[p[0], p[0] + v[0]],
            y=[p[1], p[1] + v[1]],
            mode="lines",
            line=dict(color=color, width=2),
            marker=dict(size=4, color=color),
            name=name,
            showlegend=False,
        )
    )
    # Add triangle head to indicate direction
    head_size = 0.1
    fig.add_trace(
        go.Scatter(
            x=[p[0] + v[0]],
            y=[p[1] + v[1]],
            mode="markers",
            marker=dict(size=8, color=color, symbol="triangle-up"),
        )
    )


def dist2set(q, obstacles, r=0.1, h=1e-2, method="ours"):
    dists = np.zeros(len(obstacles))
    grads = np.zeros((q.shape[0], len(obstacles)))  # n x m
    p_i = q.reshape(-1, 1)
    # print(f"[DEBUG-PY] Received r={r}, h={h} in dist2set")

    for j, obs in enumerate(obstacles):
        if method.lower() == "esdf":
            dist_ij, grad, closest_pt = ESDF_CGAL(p_i, obs.A, obs.b.reshape(-1, 1))
        else:
            try:
                dist_ij, grad = signedDist2Convex(
                    p_i,
                    obs.A,
                    obs.b.reshape(-1, 1),
                    r=r,
                    eps=h,
                )
            except:
                print(f"Failed signedDist at point {p_i.flatten()} for obstacle {j}")
                raise ValueError("Signed distance computation failed.")
        dists[j] = dist_ij
        grads[:, j] = grad.reshape(-1)

    if method == "esdf":
        id_min = np.argmin(dists)
        min_dist = min(dists)
        min_grad = grads[:, id_min]
    else:
        min_dist, smooth_min_grad = smoothMinListWithGradient(dists, r=r)
        min_grad = grads @ smooth_min_grad.reshape(-1, 1)  # n x 1

    return min_dist, min_grad


def cbf_2nd_test(q, q_dot, qd, circ_rad, circ_center, eps, k, eta, gamma=1e-3):
    r"""
    min ||dr/dq u + 2k dr/dq qdot + k^2r + qdot^T d^2r/dq^2 qdot||^2 + eps ||u||^2
    s.t.
        \ddot{B} \ge -2\eta\dot{B} - \eta^2 B
    -->
        min ||dr/dq u + 2k dr/dq qdot + k^2r + qdot^T d^2r/dq^2 qdot||^2 + eps ||u||^2
    """
    qd = qd.reshape(-1, 1)
    q_dot = q_dot.reshape(-1, 1)
    p_i = q.reshape(-1, 1)
    r_task = p_i - qd  # 1 x N vector
    drdq = np.eye(q.shape[0])
    # d2rdq2 is a null tensor

    # B(q) will be the distance to the obstacles
    dist_vec = p_i - circ_center.reshape(-1, 1)
    dist2center = np.linalg.norm(dist_vec)
    B = dist2center - circ_rad
    dBdq = dist_vec / (dist2center + 1e-6)
    dot_B = (dBdq.T @ q_dot).item()
    p_next = p_i + gamma * q_dot
    p_prev = p_i - gamma * q_dot
    dist_vec_prev = p_prev - circ_center.reshape(-1, 1)
    dist_vec_next = p_next - circ_center.reshape(-1, 1)
    dist2center_prev = np.linalg.norm(dist_vec_prev)
    dist2center_next = np.linalg.norm(dist_vec_next)
    dBdq_prev = dist_vec_prev / (dist2center_prev + 1e-6)
    dBdq_next = dist_vec_next / (dist2center_next + 1e-6)
    quad_term = (dBdq_next - dBdq_prev).T / (2 * gamma) @ q_dot

    A = dBdq.T
    b = -2 * eta * dot_B - eta**2 * B - quad_term

    # Uaibot uses 1/2 u^T H u + f^T u s.t. Au >= b
    # Functional
    H = 2.0 * (eps * np.eye(q.shape[0])) + (drdq @ drdq.T)
    f_transpose = (
        2 * k * drdq.T @ q_dot
        + k**2 * r_task  # + q_dot.T @ d2rdq2 @ q_dot # This is zero
    ).T @ drdq.T
    f_transpose = f_transpose.reshape(1, -1)

    u = Utils.solve_qp(H, f_transpose, A, b)
    u = np.array(u).reshape(-1, 1)
    obj_test = u.T @ H @ u + f_transpose @ u
    constraint_test = A @ u - b
    flag = False

    if constraint_test.flatten()[0] <= 0:
        flag = True
        # print(
        #     f"Constraint violation: Au = {(A @ u).flatten()[0]}, b = {b.flatten()[0]}"
        # )
        # print(
        #     f"At point {p_i.flatten()}. Current dB/dq: {dBdq.flatten()}. dot_B: {dot_B.flatten()[0]}, B: {B}, quad_term: {quad_term.flatten()[0]}. u: {u.flatten()}"
        # )
    # print(f"Objective: {obj_test.flatten()[0]}")
    # print(f"Constraint (should be >=0): {constraint_test.flatten()[0]}")
    return u, dBdq, flag


def cbf_2nd(q, q_dot, qd, obstacles, eps, k, eta, r, h, method="ours", gamma=1e-3):
    r"""
    min ||dr/dq u + 2k dr/dq qdot + k^2r + qdot^T d^2r/dq^2 qdot||^2 + eps ||u||^2
    s.t.
        \ddot{B} \ge -2\eta\dot{B} - \eta^2 B
    -->
        min ||dr/dq u + 2k dr/dq qdot + k^2r + qdot^T d^2r/dq^2 qdot||^2 + eps ||u||^2
    """
    qd = qd.reshape(-1, 1)
    q_dot = q_dot.reshape(-1, 1)
    p_i = q.reshape(-1, 1)
    r_task = p_i - qd  # 1 x N vector
    drdq = np.eye(q.shape[0])
    # d2rdq2 is a null tensor

    # B(q) will be the distance to the obstacles
    B, dBdq = dist2set(p_i, obstacles, r=r, h=h, method=method)
    # dBdq u >= -2 eta dot_B - eta^2 B - qdot^T d2Bdq2 qdot
    # d2Bdq2 = np.zeros((q.shape[0], q.shape[0]))
    # for i in range(q.shape[0]):
    #     e_i = np.zeros((q.shape[0], 1))
    #     e_i[i, 0] = 1.0
    #     _, dBdq_prev = dist2set(p_i - gamma * e_i, obstacles, r=r, h=h, method=method)
    #     _, dBdq_next = dist2set(p_i + gamma * e_i, obstacles, r=r, h=h, method=method)
    #     d2Bdq2[:, i] = (dBdq_next - dBdq_prev).reshape(-1) / (2 * gamma)
    #
    # quad_term = q_dot.T @ d2Bdq2 @ q_dot
    dot_B = dBdq.T @ q_dot
    _, dBdq_prev = dist2set(p_i - gamma * q_dot, obstacles, r=r, h=h, method=method)
    _, dBdq_next = dist2set(p_i + gamma * q_dot, obstacles, r=r, h=h, method=method)
    quad_term = (dBdq_next - dBdq_prev).T / (2 * gamma) @ q_dot

    A = dBdq.T
    b = -2 * eta * dot_B - eta**2 * B - quad_term

    ## Uaibot uses 1/2 u^T H u + f^T u s.t. Au >= b
    # Functional
    H = 2.0 * (eps * np.eye(q.shape[0])) + (drdq @ drdq.T)
    f_transpose = (
        2 * k * drdq.T @ q_dot
        + k**2 * r_task  # + q_dot.T @ d2rdq2 @ q_dot # This is zero
    ).T @ drdq.T
    f_transpose = f_transpose.reshape(1, -1)

    u = Utils.solve_qp(H, f_transpose, A, b)
    u = np.array(u).reshape(-1, 1)
    obj_test = u.T @ H @ u + f_transpose @ u
    constraint_test = A @ u - b
    flag = False
    if constraint_test.flatten()[0] <= 0:
        flag = True
        # print(
        #     f"Constraint violation: Au = {(A @ u).flatten()[0]}, b = {b.flatten()[0]}"
        # )
        # print(
        #     f"At point {p_i.flatten()}. Current dB/dq: {dBdq.flatten()}. dot_B: {dot_B.flatten()[0]}, B: {B}, quad_term: {quad_term.flatten()[0]}. u: {u.flatten()}"
        # )
    # print(f"Objective: {obj_test.flatten()[0]}")
    # print(f"Constraint (should be >=0): {constraint_test.flatten()[0]}")
    return u, dBdq, flag


A1 = np.array(
    [
        [1.0, 0.0],
        [-1.0, 0.0],
        [0.0, 1.0],
        [0.0, -1.0],
    ]
)

# Room-like environment
bounding_box = (-12.0, -10.0, 7.0, 10.0)
q0 = np.array([-5.0, -6.0]).reshape(-1, 1)  # 1111
qd = np.array([1.6, 5.0]).reshape(-1, 1)  # 1111
qd = np.array([-1., 8.0]).reshape(-1, 1)  # LImit CYcle
# qd = np.array([-10.6, 8.0]).reshape(-1, 1)  # Works
# qd = np.array([-3.5, 8.0]).reshape(-1, 1)  # LImit CYcle
# qd = np.array([-4.1, 8.0]).reshape(-1, 1)  # Works
# qd = np.array([6.0, 2.0]).reshape(-1, 1)
# qd = np.array([-3.0, 4.0]).reshape(-1, 1)

# left_wall = np.array([-5, 7.0, 5.0, 5.0])
# o1 = Polytope(A1, left_wall)
# top_wall = np.array([3.0, 7.0, 7.0, -5.0])
# o2 = Polytope(A1, top_wall)
# bottom_wall = np.array([-5.0, 12.0, -5.0, 8.0])
# o3 = Polytope(A1, bottom_wall)
# bottom_wall2 = np.array([7.0, 4.0, -5.0, 8.0])
# o4 = Polytope(A1, bottom_wall2)
right_wall = np.array([7.0, -1.0, 7.0, 5.0])
left_wall = right_wall + A1 @ np.array([-10.0, 0.0])
o1 = Polytope(A1, left_wall)
o2 = Polytope(A1, right_wall)
obstacles = [o1, o2]
# o5 = Polytope(A1, right_wall)
# obstacles = [
#     o1,
#     o2,
#     o3,
#     o4,
#     o5,
# ]
# table = np.array([4.0, 4.0, 3.0, 3.0])
# o6 = Polytope(A1, table)
# # obstacles.append(o6)
# desk_base = np.array([-2.0, 4.0, 3.0, -1.5])
# for i in range(3):
#     for j in range(4):
#         displacement = np.array([i * 2.5, -j * 2.0])
#         desk_i = desk_base + A1 @ displacement
#         o_desk = Polytope(A1, desk_i)
#         obstacles.append(o_desk)

# obstacles = [o1, o2, o6]
# Second order kinematic control for a point mass
reached = False
q = q0.copy()
q_dot = np.zeros((2, 1))
dt = 1e-3
# 10 seconds max based on dt
max_iters = int(10.0 / dt) * 2.0
# max_iters = 1700
# max_iters = 1
iter = 0
r = 1e-1
h = 1e-2
eta = 0.5 * 1# * 25.0
eps = 1e-13
k = 1.0
gamma = dt
method = "ours"
method = "esdf"

hist = [q.copy().T]
hist_grad = [q.copy().T * 0]
flags = [False]
flag = False

while not reached:
    # Compute u using a second-order CBF
    if np.linalg.norm(q - qd) < 1e-2:
        reached = True
        print("Reached the goal!")
    if flag:
        pass
        # break

    # try:
    # u, grad, flag = cbf_2nd_test(
    #     q,
    #     q_dot,
    #     qd,
    #     circ_rad=circ_rad,
    #     circ_center=circ_center,
    #     eps=eps,
    #     k=k,
    #     eta=eta,
    #     gamma=gamma,
    # )
    u, grad, flag = cbf_2nd(
        q,
        q_dot,
        qd,
        obstacles,
        eps=eps,
        k=k,
        eta=eta,
        r=r,
        h=h,
        method=method,
        gamma=gamma,
    )
    if iter % 1000 == 0:
        print(u)
    # except:
    #     print("QP failed, stopping simulation.")
    #     x, y = -7.98005, -17.97
    #     x_bar, y_bar = -1/x, -1/y
    #     r = 0.1
    #     v1, g1 = smoothMin2ElementsWithGradient(x, y, r=r)
    #     v1, g1
    #     v, g = holderMeanWithGradient(-1 / x, -1 / y, r=0.1)
    #     -1 / v, g
    #     x_bar, y_bar
    #
    #     break
    q += q_dot * dt
    q_dot = q_dot + u * dt
    # q_dot = -k * (q - qd)
    # q_dot += u * dt
    hist.append(q.copy().T)
    hist_grad.append(grad.copy().T)
    flags.append(flag)
    iter += 1
    if iter >= max_iters:
        print("Max iterations reached, stopping simulation.")
        break

print(f"Number of iterations: {iter}, number of constraint violations: {sum(flags)}")
# %%
fig, cmap_data = create_level_sets(
    obstacles,
    r=r,
    h=h,
    kind="both",
    method=method,
    bbox=bounding_box,
    n_points=20,
    n_contours=40,
    add_reference=False,
    test=None,
    rescale=True,
    return_cmap_data=True,
)
print("Level sets created.")
#
add_point(fig, q0.flatten(), color="green", name="Start")
add_point(fig, qd.flatten(), color="red", name="Goal")
# Convert hist of points to a 2 x N path
full_path = np.array(hist).reshape(-1, 2).T


def f(fig):
    fig.add_trace(
        go.Scatter(
            x=full_path[0, :],
            y=full_path[1, :],
            mode="lines",
            line=dict(color="black", width=2),
            name="Trajectory",
        )
    )


f(fig)

# max_grads = 100
# plot_idxs = np.linspace(0, len(hist) - 1, max_grads).astype(int)
# for i, (p, grad_) in enumerate(zip(hist, hist_grad)):
#     if i not in plot_idxs:
#         continue
#     norm_ = np.linalg.norm(grad_)
#     norm_ = max(norm_, 1e-6)
#     grad = grad_ / norm_ * min(norm_, 0.5)
#     add_vector_at_point(fig, p.flatten(), grad.flatten(), color="blue", name="Grad")

fig.show()
