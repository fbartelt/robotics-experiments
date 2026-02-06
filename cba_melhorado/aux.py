import pickle
import os
import time
import uaibot as ub
import numpy as np
import sys
from adaptive_cpp import AdaptiveController, ControlLoop
from uaibot_cpp_bind import expSO3, SmapSO3, SmapSE3, expSE3
from uaibot.utils import Utils
from scipy.linalg import block_diag
from scipy.optimize import root
from vfutils import vector_field_plot
from pathlib import Path
import webbrowser


def nvim_anim(sim, path="./", file_name="ub_example"):
    sim.set_parameters(width=800, height=600)
    sim.save(path, file_name)
    final_path = os.path.join(path, file_name + ".html")
    print(f"Saved animation to {final_path}")
    path = Path(final_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    # Convert to file:// URL and open
    webbrowser.open_new_tab(path.as_uri())


# %%
# Kinematic Controller
def hd(s, r=1, b=1, d=0.2):
    """Curve parametrization used in paper. This is based on the hyperbolic
    paraboloid.

    Parameters
    ----------
    s : float
        Parameter of the curve. It must be in the interval [0, 1].
    r : float, optional
        Radius of the curve in XY plane. The default is 1.
    b : float, optional
        Height of the curve. The default is 1.
    d : float, optional
        Curvature of the curve. The default is 0.2.

    Returns
    -------
    hds : np.array
        Homogeneous transformation matrix of the curve evaluated at parameter s.
        This is a 'list' of elements of the SE(3) group.
    """
    theta = 2 * np.pi * s
    hds = np.identity(4)  # initialize the homogeneous transformation matrix
    # position = [
    #     r * np.cos(theta),
    #     r * np.sin(theta),
    #     b + d * r**2 * (np.cos(theta) ** 2 - np.sin(theta) ** 2),
    # ]
    position = [
        r * (np.sin(theta) + 2 * np.sin(2 * theta)),
        r * (np.cos(theta) - 2 * np.cos(2 * theta)),
        b + r * (-np.sin(3 * theta)),
    ]

    hds[:3, 3] = np.array(position)
    angle = np.pi / 6 * np.sin(2 * np.pi * s)
    # angle = theta
    orientation = np.array(
        [
            [1, 0, 0],
            [0, np.cos(angle), np.sin(angle)],
            [0, -np.sin(angle), np.cos(angle)],
        ]
    )
    hds[:3, :3] = orientation
    return hds

def precomputed_hd(curve_fun, n_points, *args, **kwargs):
    """Function that precomputes the curve for each parameter s.

    Parameters
    ----------
    curve_fun : function
        Function that computes the curve. It must be a function that takes as
        first argument the parameter s, and returns a homogeneous transformation
        matrix.
    n_points : int
        Number of points in the curve.
    *args : list
        Arguments of the curve function.
    **kwargs : dict
        Keyword arguments of the curve function.

    Returns
    -------
    precomputed : np.array
        Array with the precomputed curve. The shape is (n_points, 4, 4).
    """
    s = np.linspace(0, 1, num=n_points)
    precomputed = []
    for si in s:
        precomputed.append(curve_fun(si, *args, **kwargs))
    # precomputed = np.array(precomputed)
    return precomputed


def pose2htm(p, R):
    """Homogeneous transformation matrix from position and rotation."""
    p = np.array(p)
    htm = np.eye(4)
    htm[0:3, 0:3] = R
    htm[0:3, 3] = p.ravel()
    return htm


def progress_bar(i, imax):
    """Prints a progress bar in the terminal.

    Parameters
    ----------
    i : int
        Current iteration.
    imax : int
        Maximum number of iterations.
    """
    sys.stdout.write("\r")
    sys.stdout.write(
        "[%-20s] %d%%" % ("=" * round(20 * i / (imax - 1)), round(100 * i / (imax - 1)))
    )
    sys.stdout.flush()


def skew(q):
    """Maps a vector to a skew-symmetric matrix"""
    q = np.array(q).ravel()
    return np.array([[0, -q[2], q[1]], [q[2], 0, -q[0]], [-q[1], q[0], 0]])


def L(v_):
    if v_.shape != (3, 1):
        print(f"Vector shape is {v_.shape}, but should be (3, 1)")
    v = v_.copy().ravel()
    return np.array(
        [
            [v[0], v[1], v[2], 0, 0, 0],
            [0, v[0], 0, v[1], v[2], 0],
            [0, 0, v[0], 0, v[1], v[2]],
        ]
    )


def adaptive_dyn(p, R, xi, xi_d, psi_dot, o_hat, r_hat, I_p, r_p, r_i, m, N, Kd):
    # inputs: position x, desired pos. x_d, orientation R, des. orient. R_d, velocity dq, param estimates a_hat, r_hat, time t;
    # optional : drop time t_drop
    # outputs: state & parameter derivatives
    ## REMOVED ADAPTATION FOR TESTING (PARAMETERS WILL BE THE TRUE ONES)
    xi = np.array(xi).reshape(-1, 1)
    xi_d = np.array(xi_d).reshape(-1, 1)
    dx = np.array(xi[:3]).reshape(-1, 1)
    w = np.array(xi[3:]).reshape(-1, 1)
    s = xi - xi_d

    # Reference signals
    al_r = np.array(psi_dot[3:]).reshape(-1, 1)
    w_r = np.array(xi_d[3:]).reshape(-1, 1)
    a_r = np.array(psi_dot[:3]).reshape(-1, 1)
    v_r = np.array(xi_d[:3]).reshape(-1, 1)

    # Compute regressors
    Y_l = np.hstack([a_r, -skew(al_r) @ R - skew(w) @ skew(w_r) @ R, np.zeros((3, 6))])
    Y_r = np.hstack(
        [
            np.zeros((3, 1)),
            skew(a_r) @ R + skew(w) @ skew(v_r) @ R - skew(w_r) @ skew(dx) @ R,
            R @ L(R.T @ al_r) + skew(w) @ R @ L(R.T @ w_r),
        ]
    )
    Y_o = np.vstack([Y_l, Y_r])

    # Compute true dynamics matrices
    H = np.vstack(
        [
            np.hstack([m * np.eye(3), m * skew(R @ r_p)]),
            np.hstack([-m * skew(R @ r_p), R @ I_p @ R.T]),
        ]
    )
    C = np.vstack(
        [
            np.hstack([np.zeros((3, 3)), m * skew(w) @ skew(R @ r_p)]),
            np.hstack(
                [
                    -m * skew(w) @ skew(R @ r_p),
                    skew(w) @ R @ I_p @ R.T - m * skew(skew(R @ r_p) @ dx),
                ]
            ),
        ]
    )
    off_diag = m * skew(w) @ R @ skew(r_p) @ R.T - m * R @ skew(r_p) @ R.T @ skew(w)
    # Apply the adaptive control law
    input_ = np.zeros((6, 1))
    F = np.zeros((N, 6, 1))
    wrenches = []
    for i in range(N):
        # Grasp matrix (M in paper)
        G = np.vstack(
            [
                np.hstack([np.eye(3), np.zeros((3, 3))]),
                np.hstack([skew(R @ r_i[i]), np.eye(3)]),
            ]
        )
        G_h = np.vstack(
            [
                np.hstack([np.eye(3), np.zeros((3, 3))]),
                np.hstack([skew(R @ r_hat[i]), np.eye(3)]),
            ]
        )
        G_h_inv = np.vstack(
            [
                np.hstack([np.eye(3), np.zeros((3, 3))]),
                np.hstack([-skew(R @ r_hat[i]), np.eye(3)]),
            ]
        )
        F[i] = Y_o @ o_hat[i].reshape(-1, 1) - Kd @ s  # implement control law
        tau = G_h_inv @ F[i]  # compensate for est. torque
        input_ += tau  # compute real applied wrench; add to running
        wrenches.append(tau)

    ddq = np.array(ub.Utils.dp_inv(H) @ (input_ - C @ xi)).reshape(-1, 1)

    return ddq, wrenches


# Check if all variables were defined for adaptyve

# Initial conditions
n_points = 2000
r, b, d = 0.08, 0.4, 0
curve = precomputed_hd(hd, n_points, r, b, d)
curve_points = np.array([c[:3, 3] for c in curve])

# Box
l, w, h = 0.5, 0.3, 0.2  # y, x, z
m = 10.0 / 10
print(m)


rng = np.random.default_rng(42)
N = 1
# Initial positions of the agents
r_i = np.array(
    [
        [0.15, 0.25 * 0, -0.1 * 0],
        [-0.15, -0.25 * 0, 0.1 * 0],
    ]
)
r_p = r_i[0]
r_hat = r_i
print(f"Agents distributed as: {r_i}")

# Inertia tensor (Box)
I_cm = (1.0 / 12.0) * np.eye(3)
I_cm[0, 0] *= m * (w**2 + h**2)
I_cm[1, 1] *= m * (l**2 + h**2)
I_cm[2, 2] *= m * (l**2 + w**2)
I_p = np.array(I_cm - m * Utils.S(r_p) @ Utils.S(r_p))


mean_a, std_a = 0.0, 1.0
o_hat = [rng.normal(mean_a, std_a, (10,)) for _ in range(N)]
o_true = np.array(
    [
        m,
        0,
        0,
        0,  # m * r_p
        I_p[0, 0],
        I_p[0, 1],
        I_p[0, 2],
        I_p[1, 1],
        I_p[1, 2],
        I_p[2, 2],
    ]
)
o_true[1:4] = m * r_p
o_i = o_true / N

o_hat = [o_i for _ in range(N)]

p0 = np.array([-0.1, 0, 0.2]).reshape(-1, 1)
# p0 = curve[0, :3, 3].reshape(-1, 1)  # Start at beginning of curve
p = p0.copy()
# R0 = curve[0, :3, :3]  # Start with orientation of curve
R0 = np.eye(3)  # Start with no rotation
R = R0.copy()
htm0 = pose2htm(p0, R0)
object = ub.Box(htm=htm0, width=w, depth=l, height=h, color="orange", opacity=0.5)
htm = htm0.copy()

xi = np.zeros((6, 1))
v, omega = xi[0:3], xi[3:6]

kn1, kn2 = 0.2, 1.0
kt1, kt2, kt3 = kn1, 1, kn2

### Agents
htm1 = np.array(
    htm0 @ pose2htm([p0[0].item() + 2.4 * w, 0, 0], np.eye(3)) @ Utils.rotz(np.pi)
)
agent1 = ub.Robot.create_kinova_gen3(htm=htm1)
htm1_eef = htm0 @ pose2htm(
    r_i[0],
    np.array(R0 @ Utils.roty(-np.pi / 2)[:3, :3]) @ Utils.rotz(-np.pi / 2)[:3, :3],
)
q1_0 = agent1.ikm(
    htm_tg=htm1_eef,
    htm=htm1,
    check_auto=False,
    check_joint=True,
    q0=agent1.q0,
    no_iter_max=2000,
    no_tries=100,
)
agent1.set_ani_frame(q=q1_0)
htm0_to_eef0_aux = agent1.fkm(axis="eef")
htm_eef_ag1_to_obstacle = np.array(ub.Utils.inv_htm(htm0_to_eef0_aux) @ htm0)

htm2 = htm0 @ pose2htm([p0[0].item() - 0.5, 0, 0], np.eye(3))
agent2 = ub.Robot.create_kinova_gen3(htm=htm2, color="#1c1c1c")
htm2_eef = htm0 @ pose2htm(
    r_i[1], np.array(R0 @ Utils.roty(np.pi / 2)[:3, :3]) @ Utils.rotz(np.pi / 2)[:3, :3]
)
q2_0 = agent2.ikm(
    htm_tg=htm2_eef,
    htm=htm2,
    check_auto=False,
    check_joint=False,
    q0=agent2.q0,
    no_iter_max=2000,
    no_tries=100,
)
agent2.set_ani_frame(q=q2_0)
htm0_to_eef0_aux = agent2.fkm(axis="eef")
htm_eef_ag2_to_obstacle = np.array(ub.Utils.inv_htm(htm0_to_eef0_aux) @ htm0)


agents = [agent1, agent2]

htm_obj_to_r_p = pose2htm(r_p, np.eye(3))

# USE ONLY ONE AGENT FOR TESTING
agents = [agent1]
r_i = np.array(
    [
        [0.15, 0.25 * 0, -0.1 * 0],
        # [-0.15, -0.25 * 0, 0.1 * 0],
    ]
)

ball_rp = ub.Ball(htm=htm0 @ htm_obj_to_r_p, radius=0.02, color="purple")
frame1 = ub.Frame(htm=htm1_eef)
frame2 = ub.Frame(htm=htm2_eef)
sim = ub.Simulation.create_sim_grid(agents + [object, ball_rp])
sim.add(frame1)
sim.add(frame2)
curve_pointcloud = ub.PointCloud(points=curve_points.T, size=0.03, color="cyan")
sim.add(curve_pointcloud)

# %%
qdots = [np.zeros_like(agent.q) for agent in agents]
qdots_int = [np.zeros_like(agent.q) for agent in agents]
qddots = [np.zeros_like(agent.q) for agent in agents]
chidot = np.zeros((6, 1))

small_limit = 0.62832  # 6rpm
big_limit = 0.83776  # 8rpm
qdot_limits = 10 * np.array(
    [
        [big_limit],
        [big_limit],
        [big_limit],
        [small_limit],
        [small_limit],
        [small_limit],
        [small_limit],
    ]
)

# %%
# Simulation [jump]
dt = 1e-3
T = 10.0
delta = 1e-3
ds = 1e-3
deadband = 0.01
Kd = 1e-3 * np.eye(6) * 1000 * 5
Kmd, Kmi = 5 * 2 * 10, 5 * 2 * 5
k1 = 35e-1 / 500
# k1 = 35e-3
K_adap = k1 * block_diag(20e3 / N * np.eye(3), 25e3 / N * np.eye(3))


n_steps = int(T / dt)
vf_dict = {
    "curve": curve,
    "kt1": kt1,
    "kt2": kt2,
    "kt3": kt3,
    "kn1": kn1,
    "kn2": kn2,
    "delta": delta,
    "ds": ds,
}

# Log variables
log_htm = np.zeros((n_steps, 4, 4))
log_xi = np.zeros((n_steps, 6))
log_xi_d = np.zeros((n_steps, 6))
log_o_hat = np.zeros((n_steps, N, 10))
log_r_hat = np.zeros((n_steps, N, 3))
log_psi = np.zeros((n_steps, 6))
log_dist = np.zeros((n_steps, 1))
log_imin = np.zeros((n_steps, 1))
log_zeta = np.zeros((n_steps, 1))

htm = htm0.copy() @ htm_obj_to_r_p

for i in range(n_steps):
    progress_bar(i, n_steps)
    t = i * dt
    p = np.array(htm[0:3, 3]).reshape(-1, 1)
    R = np.array(htm[0:3, 0:3])
    v, omega = xi[0:3], xi[3:6]
    xi_d, dist, imin = ub.Robot.vector_field_SE3(htm, **vf_dict)
    psi_next, dist_next, idx_next = ub.Robot.vector_field_SE3(
        expSE3(SmapSE3(xi_d) * dt) @ htm, **vf_dict
    )
    psi_prev, *_ = ub.Robot.vector_field_SE3(
        expSE3(SmapSE3(-xi_d) * dt) @ htm, **vf_dict
    )
    psi_dot = np.array((psi_next - psi_prev)) / (2 * dt)
    # print(f"#DEBUG# shapes: {p.shape}, {R.shape}, {xi.shape}, {xi_d.shape}, {psi_dot.shape}")
    zeta = xi - xi_d.reshape(-1, 1)
    if i % 50 == 0:
        print(f"||zeta|| = {np.linalg.norm(zeta)},  dist = {dist}, imin = {imin}")
    #### TODO: WORKAROUND TODO: REMOVE ####
    # o_hat = [np.array(o_i) for _ in range(N)]
    # r_hat = [np.array(r_i) for _ in range(N)]
    # First step of Heun's method
    chiddot, wrenches = adaptive_dyn(
        p, R, xi, xi_d, psi_dot, o_hat, r_hat, I_p, r_p, r_i, m, N, Kd=Kd
    )
    
    # We need to guarantee that each agent applies wrenches[i] at its end-effector
    xi_eef = np.zeros((6, 1))
    for idx, agent in enumerate(agents):
        M_agent, C_agent, g_agent = agent.dynamic_model(q=agent.q, qdot=qdots[idx])
        q_curr = np.array(agent.q.copy()).reshape(-1, 1)
        # Compute dJ/dt using the jacobian of the geometric jacobian
        dJ, J, _ = agent.jac_jac_geo(q=agent.q, axis="eef")
        # dJ is a list of the jacobians of each column of J
        dotJ = np.sum([dJ[k] * qdots[idx][k, 0] for k in range(len(q_curr))], axis=0)
        wrench = np.array(wrenches[idx]).reshape(-1, 1)
        # ddq is the acceleration of the measurement point r_p expressed in the inertial frame
        # ddq_mapped is the acceleration of the end-effector expressed in the inertial frame
        # r_p to r_i displacement (expressed in r_p frame)
        r_p_to_r_i_expr_r_p = np.array(r_i[idx]).ravel() - np.array(r_p).ravel()
        r_p_to_r_i_expr_inertial = R @ r_p_to_r_i_expr_r_p.reshape(-1, 1)
        ddq_mapped = np.zeros((6, 1))
        ddq_mapped[:3] = (
            chiddot[:3]
            + skew(omega) @ skew(omega) @ r_p_to_r_i_expr_inertial
            + skew(chiddot[3:6]) @ r_p_to_r_i_expr_inertial
        )
        ddq_mapped[3:6, 0] = chiddot[3:6, 0]

        # a_des = np.linalg.pinv(J) @ (
        #     ddq_mapped - dotJ @ qdots[idx]
        # )
        # joint_torque = M_agent @ a_des + (C_agent + g_agent - J.T @ wrench)
        # ith_qddot = np.array(
        #     np.linalg.inv(M_agent)
        #     @ (joint_torque - C_agent - g_agent)
        # ).reshape(-1, 1)

        # Resulting acceleration (perfect inverse dynamics)
        veef_d = xi_d.copy().reshape(-1, 1)
        veef_d[:3] = np.array(
            veef_d[:3] + ub.Utils.S(xi_d[3:]) @ r_p_to_r_i_expr_inertial
        )
        dotq_error = qdots[idx].reshape(-1, 1) - np.linalg.pinv(J) @ veef_d.reshape(
            -1, 1
        )

        # vel_err_term = Kmd * dotq_error
        # torque_desired = J.T @ wrench
        # integral_term = Kmi * (qdots_int[idx].reshape(-1, 1))
        #

        # Assume we will cancel dynamics so, Mddq = Tau
        # Since we wanna apply wrench at eef, we have:
        #  Mddq = -J^T * W_i
        ith_qddot = np.linalg.inv(M_agent) @ (-J.T @ wrench)
        next_q = q_curr + qdots[idx] * dt

        agent.add_ani_frame(time=i * dt, q=next_q)

        qdot_pre_clip = np.array(ith_qddot * dt).reshape(-1, 1)
        qdots[idx] += np.clip(qdot_pre_clip, -qdot_limits, qdot_limits)
        qdots_int[idx] += dotq_error * dt
        if np.any(np.abs(ith_qddot) > 100):
            print(
                f"[DEBUG] Agent {idx} a_des: {a_des.ravel()}, xi_d: {xi_d.ravel()}, veef_d: {veef_d.ravel()}, r_p_to_r_i: {r_p_to_r_i_expr_inertial.ravel()}, qdot_pre_clip: {qdot_pre_clip.ravel()}"
            )
            print(
                f"[DEBUG] Agent {idx} ddq_mapped: {ddq_mapped.ravel()}, chiddot: {chiddot.ravel()}, torque_desired: {torque_desired.ravel()}"
            )
            print(
                f"[DEBUG] Agent {idx} applied wrench: {(-J.T @ wrench).ravel()}, velocity error: {vel_err_term.ravel()}"
            )
            print(f"[DEBUG] Agent {idx} ith_qddot: {ith_qddot.ravel()}")
            print(
                f"[DEBUG] Agent {idx} qdots: {qdots[idx].ravel()}, next_q: {next_q.ravel()}"
            )
        qddots[idx] = ith_qddot
        if idx == 0:
            xi_eef = np.array(J @ qdots[idx]).reshape(-1, 1)

    # Break if any element in list qddots above 1e3
    if any(np.abs(np.array(qddots).ravel()) > 1e4):
        print("\nSimulation aborted: qddot too high!")
        print(qddots)
        break

    # Update object states based on resulting agents configuration
    Jac1, htm_ag1_eef = agents[0].jac_geo(axis="eef")
    # Twist of measurement point is the twist of agent0's eef
    xi = xi_eef.copy()

    # Next htm of the mearsurement point is exactly htm of agent0's eef with possible rotation
    htm = np.array(
        htm_ag1_eef @ pose2htm(np.zeros((3,)), htm_eef_ag1_to_obstacle[0:3, 0:3])
    )
    htm_cm = np.array(htm @ np.array(ub.Utils.inv_htm(np.matrix(htm_obj_to_r_p))))

    object.add_ani_frame(time=i * dt, htm=htm_cm)
    ball_rp.add_ani_frame(time=i * dt, htm=htm)
    frame1.add_ani_frame(time=i * dt, htm=htm)

    # Log data
    log_htm[i, :, :] = htm
    log_xi[i, :] = xi.flatten()
    log_xi_d[i, :] = xi_d.flatten()
    for j in range(N):
        log_o_hat[i, j, :] = o_hat[j].flatten()
        log_r_hat[i, j, :] = r_hat[j].flatten()

    log_psi[i, :] = psi_dot.flatten()
    log_dist[i, 0] = dist
    log_imin[i, 0] = imin
    log_zeta[i, 0] = np.linalg.norm(zeta)

print("\nSimulation finished!")

nvim_anim(sim, path="./", file_name="manip_test")
print("Initial configuration saved!")

# [back]
# %%
# Plotting
import plotly.graph_objects as go


def nvim_zeta():
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=np.arange(len(log_zeta)) * dt,
            y=log_zeta.flatten(),
            mode="lines",
            name="||zeta||",
        )
    )
    # Linear velocity error
    fig.add_trace(
        go.Scatter(
            x=np.arange(len(log_dist)) * dt,
            y=np.linalg.norm(log_xi[:, 0:3] - log_xi_d[:, 0:3], axis=1),
            mode="lines",
            name="Linear velocity error",
        )
    )
    # Angular velocity error
    fig.add_trace(
        go.Scatter(
            x=np.arange(len(log_dist)) * dt,
            y=np.linalg.norm(log_xi[:, 3:6] - log_xi_d[:, 3:6], axis=1),
            mode="lines",
            name="Angular velocity error",
        )
    )
    return fig


def nvim_dist():
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=np.arange(len(log_dist)) * dt,
            y=log_dist.flatten(),
            mode="lines",
            name="Distance to curve",
        )
    )
    return fig


fig = nvim_zeta()
fig.show()

curve = np.array(curve)
vffig = vector_field_plot(
    coordinates=log_htm[:, 0:3, 3],
    field_values=log_xi_d[:, 0:3],
    orientations=log_htm[:, 0:3, 0:3],
    curve=curve[:, 0:3, 3],
    show_curve=True,
    num_arrows=0,
    num_balls=5,
    init_ball=0,
    final_ball=None,
    frame_scale=0.1,
    curve_width=5,
    path_width=3,
)
vffig.show()

fig = nvim_dist()
fig.show()
