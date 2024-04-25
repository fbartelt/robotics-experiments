# %%
# Translation of https://github.com/pculbertson/hamilton_ac to Python
import numpy as np
from scipy.optimize import root, least_squares
from scipy.linalg import expm
from scipy.spatial.transform import Rotation
import sys
sys.path.append('/home/fbartelt/Documents/Projetos/robotics-experiments/')
sys.path.append('/home/fbartelt/Documents/Projetos/robotics-experiments/slotine')
from utils import (
    quat_to_rot,
    skew,
    Pa,
    p_norm,
    p_grad,
    pnorm_divergence,
    pnorm_hessian,
    L,
    vee,
    progress_bar,
)
from uaibot_addons.vfcomplete import VectorField

# Physical parameters
rho = 8050  # Density of steel in kg/m^3
r = 0.2  # Radius of the cylinder in meters
h = 1  # Height of the cylinder in meters
m = rho * np.pi * r**2 * h  # Mass of the cylinder in kg
num_freq = 5  # Number of frequencies to use in the trajectory
rng = np.random.default_rng(42)  # Random number generator

# For fixed sample
r_p = np.array([[0], [0], [h / 2]])  # Measurement point
# N = 2  # Number of agents # TODO
N=6
r_i = np.array(
    [[0, 0, h / 2], [0, 0, -h / 2], [r, 0, 0], [-r, 0, 0], [0, r, 0], [0, -r, 0]]
).reshape(
    -1, 3, 1
)  # Initial positions of the agents # TODO
# r_i = np.array(
#     [[r, 0, 0], [-r, 0, 0]]
# ).reshape(
#     -1, 3, 1
# )  # Initial positions of the agents

# Inertia Tensor
I_cm = (1 / 12) * np.diag([m * (3 * r**2 + h**2), m * (3 * r**2 + h**2), 6 * m * r**2])
I_p = I_cm - m * skew(r_p) @ skew(r_p)  # Steiner theorem

# Parametric curve definition
maxtheta = 1500

def parametric_eq_factory(w1, w2, c1, c2, c3, h0, maxtheta, T, dt, timedependent=True):
    theta = np.linspace(0, 2 * np.pi, num=maxtheta)
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    precomputed = ()
    cw1t = np.cos(0)
    sw1t = np.sin(0)
    cw2t = np.cos(0)
    rotz = np.matrix([[cw1t, -sw1t, 0], [sw1t, cw1t, 0], [0, 0, 1]])

    curve = np.empty((3, len(theta)))
    for i, _ in enumerate(theta):
        curve[:, i] = rotz @ np.array([
            c1 * cos_theta[i],
            c2 * sin_theta[i],
            h0 + c3 * cw2t * cos_theta[i] ** 2
        ])
    orientations = np.empty((len(theta), 3, 3))
    for i, ang in enumerate(theta):
        orientations[i, :, :] = Rotation.from_euler('z', ang).as_matrix()
    
    precomputed = (curve.T, orientations)
    
    def parametric_eq(time):
        return precomputed


    return parametric_eq

eq = parametric_eq_factory(w1=0, w2=0, c1=1, c2=1, c3=0, h0=0.6, maxtheta=maxtheta, T=2, dt=1e-2, timedependent=False)
vf = VectorField(eq, False, kf=1, vr=2e-1, wr=4e-1, beta=0.5, dt=1e-2)
# Reference signals (combination of sinusoids)
# v_d is the desired velocity, a_d is the desired acceleration
# w_d is the desired angular velocity, al_d is the desired angular acceleration
def twist_d(p, R, t=0, store_points=False):
    twist = vf.psi(p, R, time=t, store_points=store_points)
    return twist

def twist_derivative_d(p, R, v, w, t=0):
    twist_derivative = vf.acceleration(p, R, v, w, time=t)
    return twist_derivative

def dq_des(p, R, t=0, store_points=False):
    return twist_d(p, R, t=t, store_points=store_points)

def ddq_des(p, R, v, w, t=0):
    return twist_derivative_d(p, R, v, w, t=t)


def adaptive_dyn(
    x, x_d, R, R_d, dq, a_hat, r_hat, t, lambda_, Kd, t_drop=np.inf, verbose=False
):
    # inputs: position x, desired pos. x_d, orientation R, des. orient. R_d, velocity dq, param estimates a_hat, r_hat, time t;
    # optional : drop time t_drop
    # outputs: state & parameter derivatives
    global P_o, P_r, p, a_i, tol

    dx = dq[:3, :].copy()
    # print('dx', dx.ravel())
    w = dq[3:, :].copy()
    # print('w', w.ravel())
    # Compute the error signals
    Re = R_d.T @ R
    # print('Re', Re.ravel())
    psi = twist_d(x, R, t, store_points=False)
    psi_dot = twist_derivative_d(x, R, dx, w, t)
    we = w - psi[3:, :]
    # print('we', we.ravel())
    sigma = we + lambda_ * R_d @ vee(Pa(Re))
    # print('sigma', sigma.ravel())
    dx_t = dx - psi_dot[:3, :]
    # print('dx_t', dx_t.ravel())
    x_t = x - x_d
    # print('x_t', x_t.ravel())
    s = np.vstack([dx, w]) - psi
    # print('s', s.ravel())

    # Reference signals
    al_r = psi_dot[3:, :]
    w_r = psi[3:, :]
    a_r = psi_dot[:3, :]
    v_r = psi[3:, :]
    ddq_r = psi_dot
    dq_r = psi
    # print(f'al_r: {al_r.ravel()}')
    # print(f'w_r: {w_r.ravel()}')
    # print(f'a_r: {a_r.ravel()}')
    # print(f'v_r: {v_r.ravel()}')
    # print(f'ddq_r: {ddq_r.ravel()}')
    # print(f'dq_r: {dq_r.ravel()}')

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
    # print(f'Y_l: {Y_l.ravel()}')
    # print(f'Y_r: {Y_r.ravel()}')

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
    H_dot = np.vstack(
        [
            np.hstack([np.zeros((3, 3)), off_diag]),
            np.hstack(
                [
                    -off_diag,
                    skew(w) @ R @ I_p @ R.T - R @ I_p @ R.T @ skew(w),
                ]
            ),
        ]
    )
    # print(f"H: {H.ravel()}")
    # print(f"C: {C.ravel()}")
    # print(f"off_diag: {off_diag.ravel()}")
    # print(f"H_dot: {H_dot.ravel()}")

    # Apply the adaptive control law
    input_ = np.zeros((6, 1))
    F = np.zeros((N, 6, 1))
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
        F[i] = Y_o @ a_hat[i] - Kd @ s  # implement control law
        # print(f'F[{i}]: {F[i].ravel()}')
        tau = G_h_inv @ F[i]  # compensate for est. torque
        input_ += tau  # compute real applied wrench; add to running
        #TODO compare s with julia result. Doesnt match

    # print(f's: {s.ravel()}')
    # print(f'input: {input_.ravel()}')
    # Find accelearation
    # ddq = np.linalg.pinv(H) @ (input_ - C @ dq - H_dot @ dq)
    # ddq = least_squares(
    #     lambda x: (H @ x.reshape(-1, 1) + C @ dq - input_).ravel(), np.zeros((6,))
    # ).x.reshape(-1, 1)
    ddq = root(
        lambda x: (H @ x.reshape(-1, 1) + C @ dq - input_).ravel(), np.zeros((6,)), method='lm'
    ).x.reshape(-1, 1)
    # print(f'ddq: {ddq.ravel()}')

    # Adaption laws
    Y_g, dr, a_t, r_t = (
        np.zeros((N, 6, 3)),
        np.zeros((N, 3, 1)),
        np.zeros((N, 10, 1)),
        np.zeros((N, 3, 1)),
    )
    da, g_o, g_r = np.zeros((N, 10, 1)), np.zeros((N, 10, 10)), np.zeros((N, 3, 3))
    for i in range(N):
        Y_g[i] = np.vstack(
            [np.zeros((3, 3)), skew(F[i][:3]) @ R]
        )  # Compute arm regressor
        # Compute inverse Hessian weighting (Bregman divergence p.11)
        # g_o[i] = np.linalg.inv(P_o) @ np.linalg.inv(
        #     pnorm_hessian(P_o @ a_hat[i], p, tol)) @ np.linalg.inv(P_o) # TODO: correct this
        # g_r[i] = np.linalg.inv(P_r) @ np.linalg.inv(
        #     pnorm_hessian(P_r @ r_hat[i], p, tol)) @ np.linalg.inv(P_r) # TODO: correct this
        g_o[i] = np.linalg.inv(P_o)
        g_r[i] = np.linalg.inv(P_r)
        # Compute parameters derivatives and errors
        da[i] = -g_o[i] @ Y_o.T @ s
        dr[i] = -g_r[i] @ Y_g[i].T @ s
        a_t[i] = a_hat[i] - a_i
        r_t[i] = r_hat[i] - r_i[i]
        # print(f'Y_g[{i}]: {Y_g[i].ravel()}')
        # print(f'g_o[{i}]: {g_o[i].ravel()}')
        # print(f'g_r[{i}]: {g_r[i].ravel()}')
        # print(f'da[{i}]: {da[i].ravel()}')
        # print(f'dr[{i}]: {dr[i].ravel()}')
        # print(f'a_t[{i}]: {a_t[i].ravel()}')
        # print(f'r_t[{i}]: {r_t[i].ravel()}')

    ds = ddq - ddq_r
    ds2 = np.vstack(
        [
            ddq[:3] - psi_dot[:3, :] + lambda_ * (dx - psi[:3, :]),
            ddq[3:]
            - psi_dot[3:, :]
            + lambda_
            * (skew(psi[3:, :]) @ R_d @ vee(Pa(Re)) + R_d @ vee(Pa(skew(R_d.T @ we) @ Re))),
        ]
    )
    dV = s.T @ (-H @ ddq_r - C @ dq_r)
    dV2 = s.T @ H @ ds + 0.5 * s.T @ H_dot @ s
    r_err = 0
    o_err = 0

    for i in range(N):
        dV += (
            s.T @ (F[i] + Y_g[i] @ r_t[i])
            + a_t[i].T @ np.linalg.inv(g_o[i]) @ da[i]
            + r_t[i].T @ np.linalg.inv(g_r[i]) @ dr[i]
        )
        dV2 += (
            a_t[i].T @ np.linalg.inv(g_o[i]) @ da[i]
            + r_t[i].T @ np.linalg.inv(g_r[i]) @ dr[i]
        )
        o_err += np.abs(
            s.T @ Y_o @ a_t[i]
            + a_t[i].T @ P_o @ pnorm_hessian(P_o @ a_hat[i], p, tol) @ P_o @ da[i]
        )
        r_err += np.abs(
            s.T @ Y_g[i] @ r_t[i]
            + r_t[i].T @ P_r @ pnorm_hessian(P_r @ r_hat[i], p, tol) @ P_r @ dr[i]
        )

    if verbose:
        print(f't: {t}')
        print(dx)
        print(s)
        print(
            f"H_dot - 2C {np.linalg.norm((H_dot - 2 * C + (H_dot - 2*C).T)/np.linalg.norm(H_dot - 2*C))}"
        )
        print(f"dV {dV}")
        print(f"dV - dV2 {dV - dV2}")
        # print(f"ds-ds2: {np.linalg.norm(ds - ds2)}")
        # print(f"o_err, r_err: {o_err, r_err}")
        print('--'*20)

    return ddq, da, dr, dV, ds
# %%
"""#############################################################################
################################################################################
"""
a_hat = rng.normal(0, 1, (N, 10, 1))  # Initial parameter estimates

# True ai
a_I = np.array(
    [[I_p[0, 0]], [I_p[0, 1]], [I_p[0, 2]], [I_p[1, 1]], [I_p[1, 2]], [I_p[2, 2]]]
)
a = np.vstack([m, m * r_p, a_I])
a_i = (1 / N) * a

# Initial r_hat
r_hat = 2 * rng.normal(0, 1, (N, 3, 1))

# Copy julia res
# a_hat = np.array([[-0.35678967689971935, 0.2817599178248406, -0.9721434620400129, 0.8530630738504944, -1.389675891885181, -0.8560025515676851, 2.0189557509692553, 0.792858454107557, -0.7104758448377438, -0.11457073493316754], [2.4976493714992114, 1.4453986778545451, -2.7298806112833565, -0.5800271314241826, -2.215368095471661, -0.5127696665528575, 0.21099758948114417, 0.7233321019768398, -0.27110882482197846, 0.839810330572599], [1.0989109624206148, 0.25217159648087967, 0.4311320588372825, 0.635695858960056, 3.2302574678632023, 0.05819285880565524, -2.3079262142274306, -1.7195124005404891, -1.456136135841728, -0.4077914712450328], [0.8573045473107758, 0.5233863334856645, 0.5213564588053645, 1.92614535776942, -0.18114110487213408, -0.06680310393192065, -0.41249178785318313, 0.1577986800938906, 0.8838644526006741, 1.0872201601847489], [0.23308023222020632, 1.518482725486419, 0.05326741749137093, -0.2227522156842677, -0.4047375540569501, 0.4194233519392662, 0.8897868788816855, 0.5498571029831476, 0.3082570604079867, -1.4073808343637413], [-2.0076199115476085, 0.5483692642428976, 1.547313569784439, -0.48115506752570525, -1.310988976764861, 0.18665503640398487, -0.5220737792315289, 1.128790409287485, 0.03702257180202528, 0.9112942343226685]]).reshape(N, 10, 1)
# r_hat = np.array([[1.2103672222723798, -1.2578264254007059, 0.17401029166998716], [0.7577464960840908, -1.4181297137883901, -0.15937922040589292], [0.13650277449971854, -0.1939803962956579, -2.0549818960617134], [-0.232384984738905, 0.7980370387992967, -1.4896893122043025], [1.8276407724988828, -0.8393987348215005, 0.27349689514658587], [1.1720469850042583, 1.4971668404622318, -2.749834242138131]]).reshape(N, 3, 1)


# Kd Matrix
Kd = 1e-1*np.diag(np.hstack([(5e4 / N) * np.ones(3), (5e3 / N) * np.ones(3)]))

# Sample random initial orientation
quat = rng.random((4, 1))
quat = np.array([0.32898353636318073, 0.5573012924772142, 0.05592892602696582, 0.7603006362214045])
quat = quat / np.linalg.norm(quat)
quat_d = rng.random((4, 1))
quat_d = np.array([0.47257332117500883, 0.1079226612474078, 0.7420069295427152, 0.46309056545506455])
quat_d = quat_d / np.linalg.norm(quat_d)
R = quat_to_rot(quat)
R_d = quat_to_rot(quat_d)

# Initializations
dq = np.zeros((6, 1))  # Initial velocity
w, x, x_d, dx, s = (
    np.zeros((3, 1)),
    np.zeros((3, 1)),
    np.zeros((3, 1)),
    np.zeros((3, 1)),
    np.zeros((6, 1)),
)

# Simulation parameters
dt = 1e-2  # Timestep -- numerical integration via Heun's; error is ~ O(dt^2)
T = 20
lambda_ = 1.5
deadband = 0.01  # Deadband in which to stop adaptation
t_drop = T + 10  # Time to turn off agents
p = 2  # l-p norm to be used for regularization
verbose = False

P_o = 3e1 * np.linalg.inv(np.diag(np.abs(a_i.ravel()) + 1e-2))
P_r = 3e1 * np.eye(3)
tol = 1e-5

# Storage for simulation data
s_t, rot_err, V, Vdot, s_hat, Vhat, Vdot = [], [], [], [], [], [], []
lyap_params, lyap_dynamics, err_mean, param_est, arm_est = [], [], [], [], []
dV_params, dV_dynamics, hist_x, hist_dx, hist_vf = [], [], [], [], []
hist_xd, hist_R, hist_w = [], [], []
dropped = False

# Simulation Loop
for t in np.arange(0, T + dt, dt):
    index = int(t / dt)
    index_max = int((T + dt) / dt)
    progress_bar(index, index_max)
    if (t >= t_drop) and (not dropped):
        N = int(N / 2)
        dropped = True

    dx = dq[:3].copy()
    w = dq[3:].copy()
    psi = twist_d(x, R, t, store_points=True)
    psi_dot = twist_derivative_d(x, R, dx, w, t)
    Re = R_d.T @ R
    we = w - psi[3:, :]
    dx_t = dx - psi[:3, :]
    x_t = x - x_d
    s = np.vstack([dx, w]) - psi
    vf_aux = psi

    if np.linalg.norm(s) > 1e3:
        print(f"Diverged in t={t}s --- |s|={np.linalg.norm(s)}")
        break

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
    H_dot = np.vstack(
        [
            np.hstack([np.zeros((3, 3)), off_diag]),
            np.hstack(
                [
                    -off_diag,
                    skew(w) @ R @ I_p @ R.T - R @ I_p @ R.T @ skew(w),
                ]
            ),
        ]
    )

    # Compute lyap function and param errors
    a_t, r_t, g_o, g_r = (
        np.zeros((N, 10, 1)),
        np.zeros((N, 3, 1)),
        np.zeros((N, 10, 10)),
        np.zeros((N, 3, 3)),
    )
    V_curr, V_params = 0.5 * s.T @ H @ s, 0.0

    for i in range(N):
        g_o[i] = np.linalg.inv(P_o) @ np.linalg.inv(
            pnorm_hessian(P_o @ a_hat[i], p, tol) @ np.linalg.inv(P_o)
        )
        g_r[i] = np.linalg.inv(P_r) @ np.linalg.inv(
            pnorm_hessian(P_r @ r_hat[i], p, tol) @ np.linalg.inv(P_r)
        )
        a_t[i] = a_hat[i] - a_i
        r_t[i] = r_hat[i] - r_i[i]
        aux = pnorm_divergence(P_o @ a_i, P_o @ a_hat[i], p, tol) + pnorm_divergence(
            P_r @ r_i[i], P_r @ r_hat[i], p, tol
        )
        V_params += aux
        V_curr += aux

    V = V_curr
    if t > 0:
        pass
        # lyap_params = ...
        # lyap_obj = ...

    at_mean = (1 / N) * np.sum(a_t, axis=0)
    rt_mean = (1 / N) * np.sum(r_t, axis=0)

    # Logging
    err_mean.append(np.linalg.norm(at_mean) + np.linalg.norm(rt_mean))
    s_t.append(np.linalg.norm(s))
    rot_err.append(np.linalg.norm(R_d.T @ R - np.eye(3), 'fro'))
    lyap_params.append(V_params)
    lyap_dynamics.append(0.5 * s.T @ H @ s)

    # Get dynamics at current time
    ddq, da, dr, dV, ds = adaptive_dyn(
        x, x_d, R, R_d, dq, a_hat, r_hat, t, lambda_, Kd, t_drop, verbose
    )

    if t == 0:
        Vhat.append(V[-1])
        s_hat.append(s)

    a_int, r_int = np.zeros((N, 10, 1)), np.zeros((N, 3, 1))

    # Intermediate step of Heun's
    for i in range(N):
        if np.linalg.norm(s) > deadband:
            a_int[i] = a_hat[i] + dt * da[i]
            r_int[i] = r_hat[i] + dt * dr[i]
        else:
            da[i] = np.zeros((10, 1))
            dr[i] = np.zeros((3, 1))
            a_int[i] = a_hat[i]
            r_int[i] = r_hat[i]

    R_d_int = expm(dt * skew(psi[3:, :])) @ R_d
    R_int = expm(dt * skew(w)) @ R
    x_d_int = x_d + dt * psi[:3, :]
    x_int = x + dt * dx
    dq_int = dq + dt * ddq
    dx_int = dq_int[:3].copy()
    w_int = dq_int[3:].copy()

    # Second step of Heun's
    ddq_p, da_p, dr_p, dV_p, ds_p = adaptive_dyn(
        x_int,
        x_d_int,
        R_int,
        R_d_int,
        dq_int,
        a_int,
        r_int,
        t + dt,
        lambda_,
        Kd,
        t_drop,
        verbose,
    )
    Vhat.append(Vhat[-1] + (dt / 2) * (dV + dV_p))

    for i in range(N):
        if np.linalg.norm(s) > deadband:
            a_hat[i] += (dt / 2) * (da[i] + da_p[i])
            r_hat[i] += (dt / 2) * (dr[i] + dr_p[i])
        else:
            da_p[i] = np.zeros((10, 1))
            dr_p[i] = np.zeros((3, 1))

    dV_dynamics.append(0.5 * s.T @ H @ (ds + ds_p) + 0.5 * s.T @ H_dot @ s)
    vdot_params = 0.0
    for i in range(N):
        vdot_params += 0.5 * a_t[i].T @ np.linalg.inv(g_o[i]) @ (
            da[i] + da_p[i]
        ) + 0.5 * r_t[i].T @ np.linalg.inv(g_r[i]) @ (dr[i] + dr_p[i])

    dV_params.append(vdot_params)
    s_hat.append(s_hat[-1] + (dt / 2) * (ds + ds_p))

    next_psi = vf.psi(x_int, R_int, time=t + dt, store_points=False)
    R_d = expm((dt / 2) * skew(psi[3:, :] + next_psi[3:, :])) @ R_d
    R = expm((dt / 2) * (skew(w) + skew(w_int))) @ R
    x_d = x_d + (dt / 2) * (psi[:3, :] + next_psi[:3, :])
    x = x + (dt / 2) * (dx + dx_int)
    dq = dq + (dt / 2) * (ddq + ddq_p)
    dx = dq[:3].copy()
    w = dq[3:].copy()

    hist_x.append(x)
    hist_dx.append(dx)
    hist_vf.append(vf_aux)
    hist_xd.append(x_d)
    hist_R.append(R)
    hist_w.append(w)
    param_est.append(a_hat.reshape(-1, 1))
    arm_est.append(r_hat.reshape(-1, 1))

#%%
import plotly.express as px
px.scatter(np.array(s_t)).show()
pp, rr = zip(*vf.nearest_points)
px.line(np.linalg.norm(np.array(hist_x).reshape(-1, 3) - np.array(pp[::1]).reshape(-1, 3), axis=1)).show()
fro_norms = []
for rot, rot_d in zip(hist_R, rr):
    fro_norms.append(0.5 * np.linalg.norm(np.eye(3) - rot_d.T @ rot)**2)
fig2 = px.line(x=np.arange(0, T+dt, dt), y=np.array(fro_norms))
# px.line(np.array(hist_vf).reshape(-1, 6)[:, :3]).show()
# px.line(np.array(hist_vf).reshape(-1, 6)[:, 3:]).show()
# %%
"""" COPY PASTE DEBUG"""
t=0
dq = np.zeros((6, 1))
a_hat = np.array([[-0.35678967689971935, 0.2817599178248406, -0.9721434620400129, 0.8530630738504944, -1.389675891885181, -0.8560025515676851, 2.0189557509692553, 0.792858454107557, -0.7104758448377438, -0.11457073493316754], [2.4976493714992114, 1.4453986778545451, -2.7298806112833565, -0.5800271314241826, -2.215368095471661, -0.5127696665528575, 0.21099758948114417, 0.7233321019768398, -0.27110882482197846, 0.839810330572599], [1.0989109624206148, 0.25217159648087967, 0.4311320588372825, 0.635695858960056, 3.2302574678632023, 0.05819285880565524, -2.3079262142274306, -1.7195124005404891, -1.456136135841728, -0.4077914712450328], [0.8573045473107758, 0.5233863334856645, 0.5213564588053645, 1.92614535776942, -0.18114110487213408, -0.06680310393192065, -0.41249178785318313, 0.1577986800938906, 0.8838644526006741, 1.0872201601847489], [0.23308023222020632, 1.518482725486419, 0.05326741749137093, -0.2227522156842677, -0.4047375540569501, 0.4194233519392662, 0.8897868788816855, 0.5498571029831476, 0.3082570604079867, -1.4073808343637413], [-2.0076199115476085, 0.5483692642428976, 1.547313569784439, -0.48115506752570525, -1.310988976764861, 0.18665503640398487, -0.5220737792315289, 1.128790409287485, 0.03702257180202528, 0.9112942343226685]]).reshape(N, 10, 1)
r_hat = np.array([[1.2103672222723798, -1.2578264254007059, 0.17401029166998716], [0.7577464960840908, -1.4181297137883901, -0.15937922040589292], [0.13650277449971854, -0.1939803962956579, -2.0549818960617134], [-0.232384984738905, 0.7980370387992967, -1.4896893122043025], [1.8276407724988828, -0.8393987348215005, 0.27349689514658587], [1.1720469850042583, 1.4971668404622318, -2.749834242138131]]).reshape(N, 3, 1)
R = quat_to_rot(quat)
R_d = quat_to_rot(quat_d)
dx = dq[:3, :].copy()
# print('dx', dx.ravel())
w = dq[3:, :].copy()
# print('w', w.ravel())
# Compute the error signals
Re = R_d.T @ R
# print('Re', Re.ravel())
we = w - w_d(t)
# print('we', we.ravel())
sigma = we + lambda_ * R_d @ vee(Pa(Re))
# print('sigma', sigma.ravel())
dx_t = dx - v_d(t)
# print('dx_t', dx_t.ravel())
x_t = x - x_d
# print('x_t', x_t.ravel())
s = np.vstack([dx_t + lambda_ * x_t, sigma])
# print('s', s.ravel())

# Reference signals
al_r = (
    al_d(t)
    - lambda_ * skew(w_d(t)) @ R_d @ vee(Pa(Re))
    - lambda_ * R_d @ vee(Pa(skew(R_d.T @ we) @ Re))
)
w_r = w_d(t) - lambda_ * R_d @ vee(Pa(Re))
a_r = a_d(t) - lambda_ * dx_t
v_r = v_d(t) - lambda_ * x_t
ddq_r = np.vstack([a_r, al_r])
dq_r = np.vstack([v_r, w_r])
# print(f'al_r: {al_r.ravel()}')
# print(f'w_r: {w_r.ravel()}')
# print(f'a_r: {a_r.ravel()}')
# print(f'v_r: {v_r.ravel()}')
# print(f'ddq_r: {ddq_r.ravel()}')
# print(f'dq_r: {dq_r.ravel()}')

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
H_dot = np.vstack(
    [
        np.hstack([np.zeros((3, 3)), off_diag]),
        np.hstack(
            [
                -off_diag,
                skew(w) @ R @ I_p @ R.T - R @ I_p @ R.T @ skew(w),
            ]
        ),
    ]
)
input_ = np.zeros((6, 1))
F = np.zeros((N, 6, 1))
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
    F[i] = Y_o @ a_hat[i] - Kd @ s  # implement control law
    print(F[i].ravel())
    tau = G_h_inv @ F[i]  # compensate for est. torque
    input_ += tau  # compute real applied wrench; add to running