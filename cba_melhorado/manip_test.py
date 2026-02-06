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


def _dyn_model(robot_, q, qdot):
    n = len(robot_.links)
    q = np.matrix(q).reshape((n, 1))
    qdot = np.matrix(qdot).reshape((n, 1))
    # Error handling
    if not Utils.is_a_vector(qdot, n):
        raise Exception(
            "The parameter 'qdot' should be a " + str(n) + " dimensional vector."
        )

    if not Utils.is_a_vector(q, n):
        raise Exception(
            "The parameter 'q' should be a " + str(n) + " dimensional vector."
        )
    # end error handling

    jj_geo, jac_geo, htm_com = robot_.jac_jac_geo(q=q, axis="com")

    i_mat_rot = []

    for i in range(n):
        i_mat = robot_.links[i].inertia_matrix + robot_.links[i].mass * Utils.S(
            robot_.links[i].center_of_mass
        ) * Utils.S(robot_.links[i].center_of_mass)
        i_mat_rot.append(htm_com[i][0:3, 0:3] * i_mat * htm_com[i][0:3, 0:3].T)

    list_pjac_i_pk = []

    for i in range(n):
        list = []
        for k in range(n):
            pjac_i_pk = np.matrix(np.zeros((6, n)))
            for j in range(i + 1):
                pjac_i_pk[:, j] = jj_geo[i][j][:, k]

            list.append(pjac_i_pk)

        list_pjac_i_pk.append(list)

    list_pm_pk = []

    for k in range(n):
        pm_pk = np.zeros((n, n))
        for i in range(k, n):
            pm_pk += (
                robot_.links[i].mass
                * list_pjac_i_pk[i][k][0:3, :].T
                * jac_geo[i][0:3, :]
            )
            pm_pk += (
                (
                    list_pjac_i_pk[i][k][3:6, :].T
                    + jac_geo[i][3:6, :].T * Utils.S(jac_geo[i][3:6, k])
                )
                * i_mat_rot[i]
                * jac_geo[i][3:6, :]
            )

        list_pm_pk.append(pm_pk + (pm_pk).T)

    # Compute the M matrix

    dyn_m = np.zeros((n, n))

    for i in range(n):
        dyn_m += robot_.links[i].mass * jac_geo[i][0:3, :].T * jac_geo[i][0:3, :]
        dyn_m += jac_geo[i][3:6, :].T * i_mat_rot[i] * jac_geo[i][3:6, :]

    # Compute the C matrix

    dyn_c1 = np.matrix(np.zeros((n, 1)))
    dyn_c2 = np.matrix(np.zeros((n, 1)))
    qdot_v = np.matrix(qdot).reshape((n, 1))

    for j in range(n):
        dyn_c1 = dyn_c1 + (qdot_v[j, 0] * list_pm_pk[j]) * qdot_v

    for i in range(n):
        dyn_c2[i] = 0.5 * qdot_v.T * list_pm_pk[i] * qdot_v

    dyn_c = (dyn_c1 - dyn_c2).reshape((n, 1))

    # Compute the G matrix

    dyn_g = np.zeros((n, 1))
    GRAVITY_ACC = 9.81
    for i in range(n):
        dyn_g += GRAVITY_ACC * robot_.links[i].mass * jac_geo[i][2, :].reshape((n, 1))

    return dyn_m, dyn_c, dyn_g


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
    # position = [
    #     r * (np.sin(theta) + 2 * np.sin(2 * theta)),
    #     r * (np.cos(theta) - 2 * np.cos(2 * theta)),
    #     b + r * (-np.sin(3 * theta)),
    # ]
    position = [r * np.cos(theta), r * np.sin(theta), b]

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

    # dposition_ds = [
    #     r * 2 * np.pi * (np.cos(theta) + 2 * 2 * np.cos(2 * theta)),
    #     r * 2 * np.pi * (-np.sin(theta) + 2 * 2 * np.sin(2 * theta)),
    #     r * 2 * np.pi * (-3 * np.cos(3 * theta)),
    # ]
    dposition_ds = [
        -r * 2 * np.pi * np.sin(theta),
        r * 2 * np.pi * np.cos(theta),
        0,
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


def object_model(xi, R, m, r_p, r_i, I_p, applied_torques):
    xi = np.array(xi).reshape(-1, 1)
    dx = np.array(xi[:3]).reshape(-1, 1)
    w = np.array(xi[3:]).reshape(-1, 1)
    input_ = np.zeros((6, 1))

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

    for i, tau in enumerate(applied_torques):
        # Grasp matrix (M in paper)
        G = np.vstack(
            [
                np.hstack([np.eye(3), np.zeros((3, 3))]),
                np.hstack([skew(R @ r_i[i]), np.eye(3)]),
            ]
        )
        # input_ += G @ tau  # compute real applied wrench; add to running
        input_ += tau

    # ddq = np.array(ub.Utils.dp_inv(H) @ (input_ - C @ xi)).reshape(-1, 1)
    ddq = np.array(np.linalg.inv(H) @ (input_ - C @ xi)).reshape(-1, 1)

    return ddq


def adaptive_dyn(p, R, xi, xi_d, psi_dot, o_hat, r_hat, I_p, r_p, r_i, m, N, Kd):
    # inputs: position x, desired pos. x_d, orientation R, des. orient. R_d, velocity dq, param estimates a_hat, r_hat, time t;
    # optional : drop time t_drop
    # outputs: state & parameter derivatives
    N = len(r_i)
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
    # H_dot = np.vstack(
    #     [
    #         np.hstack([np.zeros((3, 3)), off_diag]),
    #         np.hstack(
    #             [
    #                 -off_diag,
    #                 skew(w) @ R @ I_p @ R.T - R @ I_p @ R.T @ skew(w),
    #             ]
    #         ),
    #     ]
    # )
    #
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
        # print(f"[DEBUG] shapes Y_o: {Y_o.shape}, o_hat[{i}]: {o_hat[i].shape}, s: {s.shape}, Kd: {Kd.shape}")
        F[i] = Y_o @ o_hat[i].reshape(-1, 1) - Kd @ s  # implement control law
        tau = G_h_inv @ F[i]  # compensate for est. torque
        # MISSING G HERE CHECK THIS URGENT TODO
        input_ += G @ tau  # compute real applied wrench; add to running
        wrenches.append(tau)

    # input_hist.append(input_)
    # ddq = root(
    #     lambda x: (H @ x.reshape(-1, 1) + C @ xi - input_).ravel(),
    #     np.zeros((6,)),
    #     method="lm",
    # ).x.reshape(-1, 1)
    ddq = np.array(ub.Utils.dp_inv(H) @ (input_ - C @ xi)).reshape(-1, 1)
    # ddq = np.clip(ddq, -0.1, 0.1)
    # print(f'ddq: {ddq.ravel()}')

    # Adaption laws
    # Y_g, dr, a_t, r_t = (
    #     np.zeros((N, 6, 3)),
    #     np.zeros((N, 3, 1)),
    #     np.zeros((N, 10, 1)),
    #     np.zeros((N, 3, 1)),
    # )
    # da, g_o, g_r = np.zeros((N, 10, 1)), np.zeros((N, 10, 10)), np.zeros((N, 3, 3))
    # for i in range(N):
    #     Y_g[i] = np.vstack(
    #         [np.zeros((3, 3)), skew(F[i][:3]) @ R]
    #     )  # Compute arm regressor
    #     # Compute inverse Hessian weighting (Bregman divergence p.11)
    #     # g_o[i] = np.linalg.inv(P_o) @ np.linalg.inv(
    #     #     pnorm_hessian(P_o @ a_hat[i], p, tol)) @ np.linalg.inv(P_o) # TODO: correct this
    #     # g_r[i] = np.linalg.inv(P_r) @ np.linalg.inv(
    #     #     pnorm_hessian(P_r @ r_hat[i], p, tol)) @ np.linalg.inv(P_r) # TODO: correct this
    #     g_o[i] = np.linalg.inv(Gamma_o)
    #     g_r[i] = np.linalg.inv(P_r)
    #     # Compute parameters derivatives and errors
    #     da[i] = -g_o[i] @ Y_o.T @ s
    #     dr[i] = -g_r[i] @ Y_g[i].T @ s
    #     a_t[i] = o_hat[i] - a_i
    #     r_t[i] = r_hat[i] - r_i[i]
    #     # print(f'Y_g[{i}]: {Y_g[i].ravel()}')
    #     # print(f'g_o[{i}]: {g_o[i].ravel()}')
    #     # print(f'g_r[{i}]: {g_r[i].ravel()}')
    #     # print(f'da[{i}]: {da[i].ravel()}')
    #     # print(f'dr[{i}]: {dr[i].ravel()}')
    # print(f'a_t[{i}]: {a_t[i].ravel()}')
    # print(f'r_t[{i}]: {r_t[i].ravel()}')

    return ddq, wrenches


# Check if all variables were defined for adaptyve

# Initial conditions
n_points = 3000
n_points = 2000
r, b, d = 0.35, 1, 0.2
r, b = 0.08, 0.7
r, b = 0.2, 0.4
curve = precomputed_hd(hd, n_points, r, b, d)
curve_points = np.array([c[:3, 3] for c in curve])
curve_derivative = precomputed_hd(hd_derivative, n_points, r, b, d)

# Box
l, w, h = 0.5, 0.3, 0.2  # y, x, z
m = 1e1
print(m)


rng = np.random.default_rng(42)
N = 2
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

pz0 = 0.4
p0 = np.array([-0.1, 0, pz0]).reshape(-1, 1)
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

# kn1, kn2 = 1.0, 10.0
kn1, kn2 = 0.4, 1.0
# kt1, kt2, kt3 = 0.2, 1, 1.0
kt1, kt2, kt3 = kn1, 1, kn2

### Agents
htm1 = np.array(
    ub.Utils.trn([0, 0, -pz0])
    @ htm0
    @ pose2htm([p0[0].item() + 2.4 * w, 0, 0], np.eye(3))
    @ Utils.rotz(np.pi)
)
agent1 = ub.Robot.create_jaco(htm=htm1)
htm1_eef = htm0 @ pose2htm(
    r_i[0],
    np.array(R0 @ Utils.roty(-np.pi / 2)[:3, :3]) @ Utils.rotz(-np.pi / 2)[:3, :3],
)
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
htm0_to_eef0_aux = agent1.fkm(axis="eef")
htm_eef_ag1_to_obstacle = np.array(ub.Utils.inv_htm(htm0_to_eef0_aux) @ htm0)

htm2 = np.array(
    ub.Utils.trn([0, 0, -pz0]) @ htm0 @ pose2htm([p0[0].item() - 0.5, 0, 0], np.eye(3))
)
agent2 = ub.Robot.create_jaco(htm=htm2, color="#1c1c1c")
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

# agents = [agent1]
# r_i = np.array(
#     [
#         [0.15, 0.25 * 0, -0.1 * 0],
#         # [-0.15, -0.25 * 0, 0.1 * 0],
#     ]
# )

ball_rp = ub.Ball(htm=htm0 @ htm_obj_to_r_p, radius=0.02, color="purple")
frame1 = ub.Frame(htm=htm1_eef)
frame2 = ub.Frame(htm=htm2_eef)
sim = ub.Simulation.create_sim_grid(agents + [object, ball_rp])
sim.add(frame1)
sim.add(frame2)
curve_pointcloud = ub.PointCloud(points=curve_points.T, size=0.03, color="cyan")
sim.add(curve_pointcloud)

# nvim_anim(sim, path="./", file_name="adaptive_anim")
# print("Initial configuration saved!")

# %%
qdots = [np.zeros_like(agent.q) for agent in agents]
qdots_int = [np.zeros_like(agent.q) for agent in agents]
qddots = [np.zeros_like(agent.q) for agent in agents]
ag_torques = [np.zeros_like(agent.q) for agent in agents]
force_err_ints = [np.zeros((6, 1)) for _ in agents]
chidot = np.zeros((6, 1))

small_limit = 0.62832  # 6rpm
big_limit = 0.83776  # 8rpm
qdot_limits = 1e10 * np.array(
    [
        [big_limit],
        [big_limit],
        [big_limit],
        [small_limit],
        [small_limit],
        [small_limit],
        # [small_limit],
    ]
)

# %%
# Simulation [jump]
dt = 1e-3
T = 10.0 / 3
delta = 1e-3
ds = 1e-3
deadband = 0.01
kd = 5
Kd = np.eye(6) * kd
Kd = block_diag(10 * kd * np.eye(3), kd * np.eye(3))
Kmd, Kmi = 5 * 2 * 10, 5 * 2 * 5
Kpf, Kif = 0.1 * np.diag([2] * 6), 0.1 * np.diag([1] * 6)
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
    # "curve_derivative": curve_derivative,
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
    # ddq, dohat, drhat, wrenches = adaptiveSys.adaptiveDynamics(
    #     p, R, xi, xi_d, psi_dot, o_hat, r_hat
    # )
    # # print(f"#DEBUG# shapes DYN: {ddq.shape}, {dohat[0].shape}, {drhat[0].shape}")
    # o_int = np.array([np.zeros((10,)) for _ in range(N)])
    # r_int = np.array([np.zeros((3,)) for _ in range(N)])
    # # print(o_int.shape, r_int.shape)
    # # print(len(o_i), o_i[0].shape, len(r_i), r_i[0].shape)
    #
    # for j in range(N):
    #     if np.linalg.norm(zeta) > deadband:
    #         o_int[j, :] = o_hat[j].ravel() + dohat[j].ravel() * dt
    #         r_int[j, :] = r_hat[j].ravel() + drhat[j].ravel() * dt
    #     else:
    #         dohat[j] = np.zeros((10,))
    #         drhat[j] = np.zeros((3,))
    #         o_int[j] = o_hat[j]
    #         r_int[j] = r_hat[j]
    #
    # Second step of Heun's method
    # htm_ref_int = expSE3(SmapSE3(dt * xi_d)) @ htm
    # htm_int = expSE3(SmapSE3(dt * xi)) @ htm
    # R_d_int = htm_ref_int[0:3, 0:3]
    # p_d_int = htm_ref_int[0:3, 3].reshape(-1, 1)
    # R_int = htm_int[0:3, 0:3]
    # p_int = htm_int[0:3, 3].reshape(-1, 1)
    # xi_int = xi.ravel() + ddq.ravel() * dt
    # xi_int = xi_int.reshape(-1, 1)
    # # print(f"#DEBUG INPS: {xi.shape}, {ddq.shape}, {xi_int.shape}")
    # xi_d_int, *_ = ub.Robot.vector_field_SE3(htm_int, **vf_dict)
    # psi_int_next, *_ = ub.Robot.vector_field_SE3(
    #     expSE3(SmapSE3(xi_d_int) * dt) @ htm_int, **vf_dict
    # )
    # psi_int_prev, *_ = ub.Robot.vector_field_SE3(
    #     expSE3(SmapSE3(-xi_d_int) * dt) @ htm_int, **vf_dict
    # )
    # psi_int_dot = (psi_int_next - psi_int_prev).reshape(-1, 1) / (2 * dt)
    # print(f"#DEBUG# shit: {xi_d_int.shape}, {psi_int_next.shape}, {psi_int_dot.shape}")

    #### TODO: WORKAROUND TODO: REMOVE ####
    # o_int = [np.array(o_i) for _ in range(N)]
    # r_int = [np.array(r_i) for _ in range(N)]

    # ddq_int, dohat_int, drhat_int, wrenches_int = adaptiveSys.adaptiveDynamics(
    #     p_int, R_int, xi_int, xi_d_int, psi_int_dot, o_int, r_int
    # )
    # print(f"[DEBUG] DDQINT: {ddq_int.shape}, {dohat_int[0].shape}, {drhat_int[0].shape}")
    # for j in range(N):
    #     if np.linalg.norm(zeta) > deadband:
    #         o_hat[j] += 0.5 * (dohat[j].ravel() + dohat_int[j].ravel()) * dt
    #         r_hat[j] += 0.5 * (drhat[j].ravel() + drhat_int[j].ravel()) * dt
    #     else:
    #         dohat[j] = np.zeros((10,))
    #         drhat[j] = np.zeros((3,))
    #         o_hat[j] = o_hat[j]
    #         r_hat[j] = r_hat[j]
    #
    # Map desired applied wrenches to each agent -> get real applied wrenches
    xi_eef = np.zeros((6, 1))

    for idx, agent in enumerate(agents):
        # M_agent, C_agent, g_agent = agent.dynamic_model(q=agent.q, qdot=qdots[idx])
        M_agent, C_agent, g_agent = _dyn_model(agent, q=agent.q, qdot=qdots[idx])
        M_agent, C_agent, g_agent = (
            np.array(M_agent),
            np.array(C_agent).reshape(-1, 1),
            np.array(g_agent).reshape(-1, 1),
        )
        q_curr = np.array(agent.q.copy()).reshape(-1, 1)
        # r_p to r_i displacement (expressed in r_p frame)
        r_p_to_r_i_expr_r_p = np.array(r_i[idx]).ravel() - np.array(r_p).ravel()
        r_p_to_r_i_expr_inertial = R @ r_p_to_r_i_expr_r_p.reshape(-1, 1)
        # Compute dJ/dt using the jacobian of the geometric jacobian
        dJ, J, _ = agent.jac_jac_geo(q=agent.q, axis="eef")
        J = np.array(J)
        # dJ is a list of the jacobians of each column of J
        dotJ = np.array(
            np.sum([dJ[k] * qdots[idx][k, 0] for k in range(len(q_curr))], axis=0)
        )
        wrench = np.array(wrenches[idx]).reshape(-1, 1)
        # Wrench[idx] is the wrench applied at the agent's grasping point, but without the correct
        # orientation frame (it assumes the orientation is the same as the object)
        # R_obj_to_eef = np.array(ub.Utils.inv_htm(np.matrix(htm_eef_ag1_to_obstacle)))[:3, :3]
        # wrench = block_diag(R_obj_to_eef, R_obj_to_eef) @ wrench

        # Compute qddot_d from wrench
        # qddot_d = np.linalg.inv(M_agent) @ (-J.T @ wrench - C_agent - g_agent)
        # print(f"[DEBUG] qddot_desired from wrench: {qddot_d.ravel()}")
        # print(f"[DEBUG] MinvJTW: {(np.linalg.inv(M_agent) @ J.T @ wrench).ravel()}")

        # print(f"[DEBUG] Agent {idx} wrench: {wrench.ravel()}")
        # print(f"[DEBUG] Agent {idx} term: {(J.T @ wrench - C_agent - g_agent)}")
        # print(f"[DEBUG] Agent {idx} M_agent: {M_agent}")
        # ddq is the acceleration of the measurement point r_p expressed in the inertial frame
        # ddq_mapped is the acceleration of the end-effector expressed in the inertial frame
        ddq_mapped = np.zeros((6, 1))
        ddq_mapped[:3] = (
            chiddot[:3]
            + skew(omega) @ skew(omega) @ r_p_to_r_i_expr_inertial
            + skew(chiddot[3:6]) @ r_p_to_r_i_expr_inertial
        )
        ddq_mapped[3:6, 0] = chiddot[3:6, 0]

        a_des = np.linalg.pinv(J) @ (
            ddq_mapped - dotJ @ qdots[idx]
        )  # + (np.eye(len(q_curr)) - np.linalg.pinv(J) @ J) @ np.zeros_like(q_curr)
        # check_torque = M_agent @ a_des
        # joint_torque = M_agent @ a_des + (C_agent + g_agent - J.T @ wrench)
        # ith_qddot = np.array(
        #     np.linalg.inv(M_agent)
        #     @ (joint_torque - C_agent - g_agent)
        #     # np.linalg.pinv(M_agent) @ (-J.T @ wrench - C_agent - g_agent*0)
        # ).reshape(-1, 1)
        # # Resulting acceleration (perfect inverse dynamics)
        # veef_d = xi_d.copy().reshape(-1, 1)
        # veef_d[:3] = np.array(
        #     veef_d[:3] + ub.Utils.S(xi_d[3:]) @ r_p_to_r_i_expr_inertial
        # )
        #
        # dotq_error = qdots[idx].reshape(-1, 1) - np.linalg.pinv(J) @ veef_d.reshape(
        #     -1, 1
        # )

        M_inv = np.array(ub.Utils.dp_inv(M_agent, eps=1e-1))
        # M_inv = np.linalg.inv(M_agent)

        # NEW FORCE CONTROLLER (DEEPSEEK based)
        JTinv = np.array(ub.Utils.dp_inv(J.T, eps=1e-5))
        # W_estimate = JTinv @ (M_agent @ qddots[idx] + C_agent + g_agent - ag_torques[idx])
        # force_err = -wrench - W_estimate
        # W_cmd = -wrench + Kpf @ force_err + Kif @ force_err_ints[idx]
        # tau = J.T @ W_cmd - C_agent - g_agent
        # print(f"[DEBUG]  force error: {force_err.ravel()}")
        # print(f"[DEBUG]  force_err_int: {force_err_ints[idx].ravel()}")
        # # print(f"[DEBUG]  W_estimate: {W_estimate.ravel()}")
        # # print(f"[DEBUG]  W_cmd: {W_cmd.ravel()}")
        # print(f"[DEBUG]  Joint torque (tau): {tau.ravel()}")
        # print(f"[DEBUG]  Rough joint accel(task space): {(J @ (M_inv @ (tau - C_agent - g_agent)) + dotJ @ qdots[idx]).ravel()}")

        # vel_err_term = Kmd * dotq_error
        # torque_desired = J.T @ wrench
        # integral_term = Kmi * (qdots_int[idx].reshape(-1, 1))

        # ith_qddot = a_des - torque_desired - vel_err_term
        # ith_qddot = -vel_err_term - integral_term
        # M_inv = np.array(ub.Utils.dp_inv(M_agent))
        # ith_qddot = root(
        #         lambda x: (M_agent @ x.reshape(-1, 1) - torque_desired.reshape(-1, 1)).ravel(),
        #         np.zeros((7,)),
        #         method="lm",
        #     ).x.reshape(-1, 1)

        # ith_qddot = M_inv @ (torque_desired) #- C_agent - g_agent
        # ith_qddot = a_des - vel_err_term
        # ith_qddot = - 1 * M_inv @ ((ag_torques[idx] - (J.T @ wrench)))#- C_agent - g_agent
        #### TEST WRENCH LOGIC ####
        # wrench = np.array([0, 0, 0, 1, 0, 0]).reshape(-1, 1)
        tau = -J.T @ wrench
        ith_qddot = M_inv @ (tau)
        # ith_qddot = M_inv @ (tau + C_agent + g_agent)
        # ith_qddot = a_des
        next_q = q_curr + qdots[idx] * dt

        agent.add_ani_frame(time=i * dt, q=next_q)

        qdot_pre_clip = np.array(ith_qddot * dt).reshape(-1, 1)
        # print(f"[DEBUG] Agent {idx} qdot_pre_clip: {qdot_pre_clip.ravel()}")
        # print(f"[DEBUG] Torque that would work: {((M_agent @ a_des) + C_agent + g_agent).ravel()}.")
        print(f"[DEBUG] qddot that would work: {(a_des).ravel()}.")
        print(f"[DEBUG] ith_qddot: {ith_qddot.ravel()}")
        # print(f"[DEBUG] Obj accel. chiddot: {chiddot.ravel()}")
        # 4) estimated wrench implied by tau (should be approx W_cmd if consistent):
        #    use residual formulation: J^T f = M ddq + C + g - tau  => f = pinv(J^T)@(M ddq + C + g - tau)
        residual = (
            M_agent @ qddots[idx] + C_agent + g_agent - ag_torques[idx]
        )  # (n,1) use previous applied torque ag_torques
        W_est_from_residual = np.linalg.pinv(J.T) @ residual  # (6,1)
        # print("W_est (residual):", W_est_from_residual.ravel())
        # print(f"[DEBUG] Agent {idx} wrench error: {(M_agent @ ith_qddot + C_agent + g_agent - torque_desired).ravel()}")
        print(f"[DEBUG] Zeta norm: {np.linalg.norm(zeta)}")
        print(f"[DEBUG] CURR ERR: {np.linalg.norm(ag_torques[idx] - (J.T @ wrench))}")
        print(f"[DEBUG] dist: {dist}, imin: {imin}")
        # print(f"[DEBUG] Agent {idx} dotq_error: {dotq_error.ravel()}, qdots_int: {qdots_int[idx].ravel()}")
        # print(f"[DEBUG] Agent {idx} qddot: {ith_qddot.ravel()}")
        # print(f"[DEBUG] Agent {idx} qdot: {qdots[idx].ravel()}")
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
        qdots[idx] += np.clip(qdot_pre_clip, -qdot_limits, qdot_limits)
        # qdots_int[idx] += dotq_error * dt
        # force_err_ints[idx] += force_err * dt
        ag_torques[idx] = tau
        qddots[idx] = ith_qddot
        if idx == 0:
            xi_eef = np.array(J @ qdots[idx]).reshape(-1, 1)

    # Break if any element in list qddots above 1e3
    if any(np.abs(np.array(qddots).ravel()) > 100):
        print("\nSimulation aborted: qddot too high!")
        print(np.linalg.det(J))
        print(qddots)
        break

    # bla

    # v_object = (
    #     Jv - ub.Utils.S(htm[:3, :3].T @ np.array(r_i[0]).reshape(-1, 1)) @ Jw
    # ) @ qdots[0]
    # xi_eef_mapped = np.vstack((v_object, xi_eef[3:6]))
    # xi = np.array(xi_eef_mapped).reshape(-1, 1)

    # # Ignore manipulator dynamics and directly set object acceleration for testing
    # htm = expSE3(SmapSE3(xi) * dt) @ htm
    # # htm = expSE3(SmapSE3(xi_d) * dt) @ htm
    # htm_cm = np.array(htm @ np.array(ub.Utils.inv_htm(np.matrix(htm_obj_to_r_p))))
    # try:
    #     agents[0].add_ani_frame(
    #         time=i * dt,
    #         q=agents[0].ikm(
    #             htm_tg=htm @ ub.Utils.inv_htm(np.matrix(htm_eef_ag1_to_obstacle @ htm_obj_to_r_p)),
    #             check_auto=False,
    #             check_joint=False,
    #             q0=agents[0].q,
    #             # no_iter_max=2000,
    #             # no_tries=100,
    #         ),
    #     )
    # except:
    #     pass
    # chidot += chiddot * dt
    # xi = chidot.copy()
    # # xi = xi_d.copy()

    # MODIFIED:
    # Update object states based on resulting agents configuration
    Jac1, htm_ag1_eef = agents[0].jac_geo(axis="eef")
    # htm = htm_ag1_eef @ htm_eef_ag1_to_obstacle
    # xi_eef = np.array(Jac1 @ qdots[0]).reshape(-1, 1)
    # Compute object xi
    # Jv, Jw = np.array(Jac1[0:3, :]), np.array(Jac1[3:6, :])
    # xi = xi_eef.copy()
    # twist at measurement point is the twist of agent0's eef twist with appropriate mapping

    # Next htm of the mearsurement point is exactly htm of agent0's eef with possible rotation
    # htm = np.array(
    #     htm_ag1_eef @ pose2htm(np.zeros((3,)), htm_eef_ag1_to_obstacle[0:3, 0:3])
    # )

    # TEST INTEGRATE OBJECT DYNAMICS DIRECTLY
    ddchi = object_model(xi, R, m, r_p, r_i, I_p, wrenches)
    print(f"[DEBUG] ddchi: {ddchi.ravel()}\nchiddot: {chiddot.ravel()}")
    htm = expSE3(SmapSE3(xi) * dt) @ htm
    xi += ddchi * dt

    htm_cm = np.array(htm @ np.array(ub.Utils.inv_htm(np.matrix(htm_obj_to_r_p))))

    object.add_ani_frame(time=i * dt, htm=htm_cm)
    ball_rp.add_ani_frame(time=i * dt, htm=htm)
    frame1.add_ani_frame(time=i * dt, htm=htm)

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
    # bla

ag_torques
wrenches
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
