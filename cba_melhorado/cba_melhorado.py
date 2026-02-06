# %%
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
    # axis = np.array([1, 1, 1])
    # axis = axis / np.linalg.norm(axis)
    # skew_mat = SmapSO3(axis)
    # orientation = expSO3(theta * skew_mat)
    # orientation = np.eye(3)
    hds[:3, :3] = orientation
    return hds


def hd_derivative(s, r=1, b=1, d=0.2):
    theta = 2 * np.pi * s
    dhds = np.zeros((4, 4))
    # dposition_ds = [
    #     -r * 2 * np.pi * np.sin(theta),
    #     r * 2 * np.pi * np.cos(theta),
    #     d
    #     * r**2
    #     * 2
    #     * (-2 * np.cos(theta) * np.sin(theta) - 2 * np.sin(theta) * np.cos(theta))
    #     * 2
    #     * np.pi,
    # ]

    dposition_ds = [
        r * 2 * np.pi * (np.cos(theta) + 2 * 2 * np.cos(2 * theta)),
        r * 2 * np.pi * (-np.sin(theta) + 2 * 2 * np.sin(2 * theta)),
        r * 2 * np.pi * (-3 * np.cos(3 * theta)),
    ]

    dhds[:3, 3] = np.array(dposition_ds)
    angle = np.pi / 6 * np.sin(2 * np.pi * s)
    # angle = theta
    orientation = np.array(
        [
            [1, 0, 0],
            [0, np.cos(angle), np.sin(angle)],
            [0, -np.sin(angle), np.cos(angle)],
        ]
    )
    chain = np.pi / 6 * 2 * np.pi * np.cos(2 * np.pi * s)
    # chain = 2 * np.pi
    dorientation_ds = chain * SmapSO3(np.array([1, 0, 0])) @ orientation
    # axis = np.array([1, 1, 1])
    # axis = axis / np.linalg.norm(axis)
    # dorientation_ds = 2 * np.pi * SmapSO3(axis * theta)
    dhds[:3, :3] = dorientation_ds

    # dhds[:3, :3] = 2 * np.pi * np.array(
    #     [
    #         [0, 0, 0],
    #         [0, -np.sin(theta), np.cos(theta)],
    #         [0, -np.cos(theta), -np.sin(theta)],
    #     ]
    # )
    return dhds


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


# Initial conditions
n_points = 3000
n_points = 2000
r, b, d = 0.35, 1, 0.2
r, b = 0.15, 0.4
curve = precomputed_hd(hd, n_points, r, b, d)
curve_points = np.array([c[:3, 3] for c in curve])
curve_derivative = precomputed_hd(hd_derivative, n_points, r, b, d)

# Sytem
rho = 8050.0  # Density of steel in kg/m^3
r = 15e-2  # Radius of the cylinder in meters
h = 10e-2  # Height of the cylinder in meters
r = 0.25
h = 1.0
m = rho * np.pi * r**2 * h  # Mass of the cylinder in Kg

# Box
l, w, h = 0.5, 0.3, 0.2  # y, x, z
m = 10.0 / 10
print(m)


rng = np.random.default_rng(42)
# r_p = np.array([0.0, 0.0, h / 2.0])  # Measurement point
# N = 6  # Number of agents
N = 2
# Initial positions of the agents
# r_i = np.array([[0, 0, h / 2], [0, 0, -h / 2], [r, 0, 0], [-r, 0, 0], [0, r, 0], [0, -r, 0]])
# Radially distributed around the object
# angle_dist = np.linspace(0, 2 * np.pi, N, endpoint=False)
# r_i = np.array([[r * np.cos(angle), r * np.sin(angle), h / 2.0] for angle in angle_dist])
r_i = np.array(
    [
        [0.15, 0.25 * 0, -0.1 * 0],
        [-0.15, -0.25 * 0, 0.1 * 0],
    ]
)
r_p = r_i[0]
print(f"Agents distributed as: {r_i}")

# # Inertia tensor (Cilinder)
# I_cm = (1.0 / 12.0) * np.eye(3)
# I_cm[0, 0] *= m * (3 * r**2 + h**2)
# I_cm[1, 1] *= m * (3 * r**2 + h**2)
# I_cm[2, 2] *= 6 * m * r**2
# I_p = np.array(I_cm - m * Utils.S(r_p) @ Utils.S(r_p))

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

mean_r, std_r = mean_a, 2 * std_a
r_hat = [rng.normal(mean_r, std_r, (3,)) for _ in range(N)]

# TODO: WORKAROUND REMOVE THIS
o_hat = [np.array(o_i) for _ in range(N)]
r_hat = [np.array(r_i[j]) for j in range(N)]

k1 = 35e-1 / 500
# k1 = 35e-3
K_adap = k1 * block_diag(20e3 / N * np.eye(3), 25e3 / N * np.eye(3))


abs_o_i = np.abs(o_i) + 1e-2 * np.ones(o_i.shape)
kgo, kgr = 3e1, 3e3
factor_ = 1.0
Gamma_o_inv = factor_ * kgo * np.linalg.inv(np.diag(abs_o_i.flatten()))  # 3e1
Gamma_r_inv = factor_ * kgr * np.eye(3)  # 3e3

print(type(Gamma_o_inv), Gamma_o_inv.shape)
print(type(Gamma_r_inv), Gamma_r_inv.shape)
print(type(K_adap), K_adap.shape)
print(type(o_i), o_i.shape)
print(type(r_i), len(r_i), len(r_i[0]))
print(type(I_p), I_p.shape)
print(type(m), type(r_hat))
print(type(r_hat), len(r_hat), r_hat[0].shape)
adaptiveSys = AdaptiveController(
    Gamma_o_inv,
    Gamma_r_inv,
    K_adap,
    o_i,
    r_i,
    I_p,
    m,
    N,
    r_p,
)

p0 = np.array([-0.1, 0, 0.2]).reshape(-1, 1)
# p0 = curve[0, :3, 3].reshape(-1, 1)  # Start at beginning of curve
p = p0.copy()
# R0 = curve[0, :3, :3]  # Start with orientation of curve
R0 = np.eye(3)  # Start with no rotation
R = R0.copy()
htm0 = pose2htm(p0, R0)
object = ub.Box(htm=htm0, width=w, depth=l, height=h)
htm = htm0.copy()

xi = np.zeros((6, 1))
v, omega = xi[0:3], xi[3:6]

# kn1, kn2 = 1.0, 10.0
kn1, kn2 = 0.2, 1.0
# kt1, kt2, kt3 = 0.2, 1, 1.0
kt1, kt2, kt3 = kn1, 1, kn2

### Agents
htm1 = htm0 @ pose2htm([p0[0].item() - 2 * w, 0, 0], np.eye(3))
agent1 = ub.Robot.create_kinova_gen3(htm=htm1)
htm1_eef = htm0 @ pose2htm(r_i[1], np.array(R0 @ Utils.roty(np.pi / 2)[:3, :3]))
q1_0 = agent1.ikm(
    htm_tg=htm1_eef,
    htm=htm1,
    check_auto=False,
    check_joint=False,
    q0=agent1.q0,
    no_iter_max=2000,
    no_tries=100,
)
agent1.set_ani_frame(q=q1_0)
htm0_to_eef0 = agent1.fkm(axis="eef")
htm_eef_ag1_to_obstacle = np.array(ub.Utils.inv_htm(htm0_to_eef0) @ htm0)

htm2 = htm0 @ pose2htm([p0[0].item() + 2 * w, 0, 0], np.eye(3))
agent2 = ub.Robot.create_kinova_gen3(htm=htm2)
htm2_eef = htm0 @ pose2htm(r_i[0], np.array(R0 @ Utils.roty(-np.pi / 2)[:3, :3]))
q2_0 = agent2.ikm(
    htm_tg=htm2_eef,
    htm=htm2,
    check_auto=False,
    check_joint=True,
    q0=agent2.q0,
    no_iter_max=2000,
    no_tries=100,
)
agent2.set_ani_frame(q=q2_0)
htm0_to_eef0 = agent2.fkm(axis="eef")
htm_eef_ag2_to_obstacle = np.array(ub.Utils.inv_htm(htm0_to_eef0) @ htm0)

agents = [agent1, agent2]

frame1 = ub.Frame(htm=htm1_eef)
frame2 = ub.Frame(htm=htm2_eef)
sim = ub.Simulation.create_sim_grid(agents + [object])
sim.add(frame1)
sim.add(frame2)
curve_pointcloud = ub.PointCloud(points=curve_points.T, size=0.03, color="cyan")
sim.add(curve_pointcloud)

# nvim_anim(sim, path="./", file_name="adaptive_anim")
# print("Initial configuration saved!")

# %%
qdots = [np.zeros_like(agent.q) for agent in agents]
qddots = [np.zeros_like(agent.q) for agent in agents]

small_limit = 0.62832  # 6rpm
big_limit = 0.83776  # 8rpm
qdot_limits = 1e-2 * np.array(
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
# Simulation
dt = 1e-3
T = 2.0
delta = 1e-3
ds = 1e-3
deadband = 0.01
n_steps = int(T / dt)
vf_dict = {
    "curve": curve,
    "kt1": kt1,
    "kt2": kt2,
    "kt3": kt3,
    "kn1": kn1,
    "kn2": kn2,
    "curve_derivative": curve_derivative,
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

for i in range(n_steps):
    progress_bar(i, n_steps)
    t = i * dt
    p = htm[0:3, 3].reshape(-1, 1)
    R = htm[0:3, 0:3]
    v, omega = xi[0:3], xi[3:6]
    xi_d, dist, imin = ub.Robot.vector_field_SE3(htm, **vf_dict)
    psi_next, dist_next, idx_next = ub.Robot.vector_field_SE3(
        expSE3(SmapSE3(xi_d) * dt) @ htm, **vf_dict
    )
    psi_prev, *_ = ub.Robot.vector_field_SE3(
        expSE3(SmapSE3(-xi_d) * dt) @ htm, **vf_dict
    )
    psi_dot = (psi_next - psi_prev) / (2 * dt)

    # print(f"#DEBUG# shapes: {p.shape}, {R.shape}, {xi.shape}, {xi_d.shape}, {psi_dot.shape}")
    zeta = xi - xi_d
    if i % 50 == 0:
        print(f"||zeta|| = {np.linalg.norm(zeta)},  dist = {dist}, imin = {imin}")
    #### TODO: WORKAROUND TODO: REMOVE ####
    # o_hat = [np.array(o_i) for _ in range(N)]
    # r_hat = [np.array(r_i) for _ in range(N)]
    # First step of Heun's method
    ddq, dohat, drhat, wrenches = adaptiveSys.adaptiveDynamics(
        p, R, xi, xi_d, psi_dot, o_hat, r_hat
    )
    # print(f"#DEBUG# shapes DYN: {ddq.shape}, {dohat[0].shape}, {drhat[0].shape}")
    o_int = np.array([np.zeros((10,)) for _ in range(N)])
    r_int = np.array([np.zeros((3,)) for _ in range(N)])
    # print(o_int.shape, r_int.shape)
    # print(len(o_i), o_i[0].shape, len(r_i), r_i[0].shape)

    for j in range(N):
        if np.linalg.norm(zeta) > deadband:
            o_int[j, :] = o_hat[j].ravel() + dohat[j].ravel() * dt
            r_int[j, :] = r_hat[j].ravel() + drhat[j].ravel() * dt
        else:
            dohat[j] = np.zeros((10,))
            drhat[j] = np.zeros((3,))
            o_int[j] = o_hat[j]
            r_int[j] = r_hat[j]

    # Second step of Heun's method
    htm_ref_int = expSE3(SmapSE3(dt * xi_d)) @ htm
    htm_int = expSE3(SmapSE3(dt * xi)) @ htm
    R_d_int = htm_ref_int[0:3, 0:3]
    p_d_int = htm_ref_int[0:3, 3].reshape(-1, 1)
    R_int = htm_int[0:3, 0:3]
    p_int = htm_int[0:3, 3].reshape(-1, 1)
    xi_int = xi.ravel() + ddq.ravel() * dt
    xi_int = xi_int.reshape(-1, 1)
    # print(f"#DEBUG INPS: {xi.shape}, {ddq.shape}, {xi_int.shape}")
    xi_d_int, *_ = ub.Robot.vector_field_SE3(htm_int, **vf_dict)
    psi_int_next, *_ = ub.Robot.vector_field_SE3(
        expSE3(SmapSE3(xi_d_int) * dt) @ htm_int, **vf_dict
    )
    psi_int_prev, *_ = ub.Robot.vector_field_SE3(
        expSE3(SmapSE3(-xi_d_int) * dt) @ htm_int, **vf_dict
    )
    psi_int_dot = (psi_int_next - psi_int_prev).reshape(-1, 1) / (2 * dt)
    # print(f"#DEBUG# shit: {xi_d_int.shape}, {psi_int_next.shape}, {psi_int_dot.shape}")

    #### TODO: WORKAROUND TODO: REMOVE ####
    # o_int = [np.array(o_i) for _ in range(N)]
    # r_int = [np.array(r_i) for _ in range(N)]

    ddq_int, dohat_int, drhat_int, wrenches_int = adaptiveSys.adaptiveDynamics(
        p_int, R_int, xi_int, xi_d_int, psi_int_dot, o_int, r_int
    )
    # print(f"[DEBUG] DDQINT: {ddq_int.shape}, {dohat_int[0].shape}, {drhat_int[0].shape}")
    for j in range(N):
        if np.linalg.norm(zeta) > deadband:
            o_hat[j] += 0.5 * (dohat[j].ravel() + dohat_int[j].ravel()) * dt
            r_hat[j] += 0.5 * (drhat[j].ravel() + drhat_int[j].ravel()) * dt
        else:
            dohat[j] = np.zeros((10,))
            drhat[j] = np.zeros((3,))
            o_hat[j] = o_hat[j]
            r_hat[j] = r_hat[j]

    # Map desired applied wrenches to each agent -> get real applied wrenches
    for idx, agent in enumerate(agents):
        M_agent, C_agent, g_agent = agent.dynamic_model(q=agent.q, qdot=qdots[idx])
        q_curr = np.array(agent.q.copy()).reshape(-1, 1)
        J, _ = agent.jac_geo()
        wrench = 0.5 * np.array(wrenches[idx] + wrenches_int[idx]).reshape(-1, 1)
        wrench = np.array(wrenches[idx]).reshape(-1, 1)
        # print(f"[DEBUG] Agent {idx} wrench: {wrench.ravel()}")
        # print(f"[DEBUG] Agent {idx} term: {(J.T @ wrench - C_agent - g_agent)}")
        # print(f"[DEBUG] Agent {idx} M_agent: {M_agent}")
        ith_qddot = np.array(
            np.linalg.pinv(M_agent) @ (-J.T @ wrench - C_agent - g_agent)
        ).reshape(-1, 1)
        next_q = q_curr + qdots[idx] * dt
        agent.add_ani_frame(time=i * dt, q=next_q)
        qdot_pre_clip = np.array(ith_qddot * dt).reshape(-1, 1)
        # print(f"[DEBUG] Agent {idx} qdot_pre_clip: {qdot_pre_clip.ravel()}")
        qdots[idx] += np.clip(qdot_pre_clip, -qdot_limits, qdot_limits)
        # print(f"[DEBUG] Agent {idx} qddot: {ith_qddot.ravel()}")
        # print(f"[DEBUG] Agent {idx} qdot: {qdots[idx].ravel()}")
        qddots[idx] = ith_qddot

    # Update object states based on resulting agents configuration
    Jac1, htm_ag1_eef = agents[0].jac_geo(axis="eef")
    htm = htm_ag1_eef @ htm_eef_ag1_to_obstacle
    xi_eef = np.array(Jac1 @ qdots[0]).reshape(-1, 1)
    # Compute object xi
    Jv, Jw = np.array(Jac1[0:3, :]), np.array(Jac1[3:6, :])
    v_object = (Jv - ub.Utils.S(htm[:3, :3].T @ np.array(r_i[0]).reshape(-1, 1)) @ Jw) @ qdots[0]
    xi_eef_mapped = np.vstack((v_object, xi_eef[3:6]))
    xi = np.array(xi_eef_mapped).reshape(-1, 1)

    object.add_ani_frame(time=i * dt, htm=htm)

    # print(f"[DEBUG] HTM:{htm.shape}, xi:{xi.shape}, xi_int:{xi_int.shape}")
    # htm = expSE3(SmapSE3(0.5 * (xi + xi_int) * dt)) @ htm
    # print(f"[DEBUG] DDQ: {ddq.shape}, ddqint: {ddq_int.shape}")
    # xi += 0.5 * (ddq + ddq_int).reshape(-1, 1) * dt

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

nvim_anim(sim, path="./", file_name="adaptive_anim")
print("Initial configuration saved!")

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
