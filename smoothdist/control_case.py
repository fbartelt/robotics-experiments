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

def dist2set(q, obstacles, r=0.1, h=1e-2, method="ours"):
    dists = np.zeros(len(obstacles))
    grads = np.zeros((q.shape[0], len(obstacles)))  # n x m
    p_i = q.reshape(-1, 1)
    # print(f"[DEBUG-PY] Received r={r}, h={h} in dist2set")

    for j, obs in enumerate(obstacles):
        if method == "esdf":
            dist_ij, grad, closest_pt = ESDF_CGAL(p_i, obs.A, obs.b.reshape(-1, 1))
        else:
            try: 
                dist_ij, grad = signedDist2Convex(
                    p_i, obs.A, obs.b.reshape(-1, 1), r=r, eps=h, test="both"
                )
            except:
                print(f"Failed signedDist at point {p_i.flatten()} for obstacle {j}")
        dists[j] = dist_ij
        grads[:, j] = grad.reshape(-1)

    if method == "esdf":
        id_min = np.argmin(dists)
        min_dist = min(dists)
        min_grad = -grads[:, id_min]
    else:
        min_dist, smooth_min_grad = smoothMinListWithGradient(dists, r=r)
        min_grad = -(grads @ smooth_min_grad.reshape(-1, 1))  # n x 1

    return min_dist, min_grad


def cbf_2nd(q, q_dot, qd, obstacles, eps, k, eta, r, h, method="ours", gamma=1e-3):
    r"""
    min ||dr/dq u + 2k dr/dq qdot + k^2r + qdot^T d^2r/dq^2 qdot||^2 + eps ||u||^2
    s.t.
        \ddot{B} \ge -2\eta\dot{B} - \eta^2 B
    -->
        min ||dr/dq u + 2k dr/dq qdot + k^2r + qdot^T d^2r/dq^2 qdot||^2 + eps ||u||^2
    """
    r_task = 0.5 * np.linalg.norm(q - qd) ** 2
    p_i = q.reshape(-1, 1)
    # B(q) will be the distance to the obstacles
    B, dBdq = dist2set(p_i, obstacles, r=r, h=h, method=method)
    dot_B = dBdq.T @ q_dot
    drdq = q - qd
    d2rdq2 = np.eye(q.shape[0])
    # Uaibot uses 1/2 u^T H u + f^T u
    # Functional
    H = 2.0 * (eps * np.eye(q.shape[0]) + drdq @ drdq.T)
    f_transpose = (
        2 * (2 * k * drdq.T @ q_dot + k**2 * r_task + q_dot.T @ d2rdq2 @ q_dot).T @ drdq.T
    )
    # Constraint
    # dBdq u >= -2 eta dot_B - eta^2 B - qdot^T d2Bdq2 qdot
    d2Bdq2 = np.zeros((q.shape[0], q.shape[0]))
    for i in range(q.shape[0]):
        e_i = np.zeros((q.shape[0], 1))
        e_i[i, 0] = 1.0
        _, dBdq_prev = dist2set(p_i - gamma * e_i, obstacles, r=r, h=h, method=method)
        _, dBdq_next = dist2set(p_i + gamma * e_i, obstacles, r=r, h=h, method=method)
        d2Bdq2[:, i] = (dBdq_next - dBdq_prev).reshape(-1) / (2 * gamma)

    A = dBdq.T
    b = -2 * eta * dot_B - eta**2 * B - q_dot.T @ d2Bdq2 @ q_dot
    # A_eq = np.zeros((1, q.shape[0]))
    # b_eq = np.zeros((1, 1))
    # print(f"A.shape={A.shape}, b.shape={b.shape}, A_eq.shape={A_eq.shape}, b_eq.shape={b_eq.shape}")
    # print(f"H.shape={H.shape}, f_transpose.shape={f_transpose.shape}")
    # np.vstack([A, A_eq])
    # print(A.shape, b.shape, q.shape)
    u = Utils.solve_qp(H, f_transpose, A, b)
    u = np.array(u).reshape(-1, 1)
    obj_test = u.T @ H @ u + f_transpose @ u
    constraint_test = A @ u - b
    # print(f"Objective: {obj_test.flatten()[0]}")
    # print(f"Constraint (should be >=0): {constraint_test.flatten()[0]}")
    return u


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
q0 = np.array([-8., -4.]).reshape(-1, 1)  # 1111
qd = np.array([1.6, 8.0]).reshape(-1, 1)  # 1111
left_wall = np.array([-5, 7.0, 5.0, 5.0])
o1 = Polytope(A1, left_wall)
top_wall = np.array([3.0, 7.0, 7.0, -5.0])
o2 = Polytope(A1, top_wall)
bottom_wall = np.array([-5.0, 12.0, -5.0, 8.0])
o3 = Polytope(A1, bottom_wall)
bottom_wall2 = np.array([7.0, 4.0, -5.0, 8.0])
o4 = Polytope(A1, bottom_wall2)
right_wall = np.array([7.0, -5.0, 7.0, 5.0])
o5 = Polytope(A1, right_wall)
obstacles = [
    o1,
    o2,
    o3,
    o4,
    o5,
]
table = np.array([4.0, 4.0, 3.0, 3.0])
o6 = Polytope(A1, table)
# obstacles.append(o6)
desk_base = np.array([-2.0, 4.0, 3.0, -1.5])
for i in range(3):
    for j in range(4):
        displacement = np.array([i * 2.5, -j * 2.0])
        desk_i = desk_base + A1 @ displacement
        o_desk = Polytope(A1, desk_i)
        obstacles.append(o_desk)

# obstacles = [o6]
# Second order kinematic control for a point mass
reached = False
q = q0.copy()
q_dot = np.zeros((2, 1))
dt = 1e-3
hist = [q.copy().T]
# 10 seconds max based on dt
max_iters = int(15.0 / dt) / 2
# max_iters = 1000
# max_iters = 1
iter = 0
r = 0.1
h = 1e-2
eta = 0.5
eps = 1e-3
k = 2.0
method = "ours"
# method = "esdf"

while not reached:
    # Compute u using a second-order CBF
    if np.linalg.norm(q - qd) < 1e-2:
        reached = True
        print("Reached the goal!")

    # try:
    u = cbf_2nd(
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
        gamma=dt,
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
    q_dot += u * dt
    # q_dot = -k * (q - qd)
    # q_dot += u * dt
    hist.append(q.copy().T)
    iter += 1
    if iter >= max_iters:
        print("Max iterations reached, stopping simulation.")
        break

print(q)
print("a")
print(len(hist))
np.array(hist).reshape(-1, 2).shape
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
fig.add_trace(
    go.Scatter(
        x=full_path[0, :],
        y=full_path[1, :],
        mode="lines",
        line=dict(color="black", width=2),
        name="Trajectory",
        )
    )
fig.show()
