#%%
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
        # curve[:, i] = rotz @ np.array([
        #     c1 * cos_theta[i],
        #     c2 * sin_theta[i],
        #     h0 + c3 * cw2t * cos_theta[i] ** 2
        # ])
        curve[:, i] = rotz @ np.array([
            c1*(sin_theta[i] + 2*np.sin(2*theta[i])),
            c1*(cos_theta[i] - 2*np.cos(2*theta[i])),
            h0 + c1*(-np.sin(3*theta[i]))
        ])
    orientations = np.empty((len(theta), 3, 3))
    for i, ang in enumerate(theta):
        # orientations[i, :, :] = Rotation.from_euler('z', 2*ang).as_matrix() @ rotz
        orientations[i, :, :] = Rotation.from_euler('z', ang).as_matrix() @ Rotation.from_euler('x', 2*ang).as_matrix()
        # orientations[i, :, :] = Rotation.from_euler('z', np.pi).as_matrix()
    
    precomputed = ((curve.T, orientations))
    
    def parametric_eq(time):
        return precomputed


    return parametric_eq
# def parametric_eq_factory(w1, w2, c1, c2, c3, h0, maxtheta, T, dt, timedependent=True):
#     theta = np.linspace(0, 2 * np.pi, num=maxtheta)
#     cos_theta = np.cos(theta)
#     sin_theta = np.sin(theta)
#     precomputed = ()
#     cw1t = np.cos(0)
#     sw1t = np.sin(0)
#     cw2t = np.cos(0)
#     rotz = np.matrix([[cw1t, -sw1t, 0], [sw1t, cw1t, 0], [0, 0, 1]])

#     curve = np.empty((3, len(theta)))
#     for i, _ in enumerate(theta):
#         curve[:, i] = rotz @ np.array([
#             c1 * cos_theta[i],
#             c2 * sin_theta[i],
#             h0 + c3 * cw2t * cos_theta[i] ** 2
#         ])
#     orientations = np.empty((len(theta), 3, 3))
#     for i, ang in enumerate(theta):
#         orientations[i, :, :] = Rotation.from_euler('z', ang).as_matrix()
#         # orientations[i, :, :] = np.eye(3)

#     precomputed = (curve.T, orientations)

#     def parametric_eq(time):
#         return precomputed


#     return parametric_eq

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
    global P_o, P_r, p, a_i, tol, psi_old, aprox_hist, input_hist

    dx = dq[:3, :].copy()
    # print('dx', dx.ravel())
    w = dq[3:, :].copy()
    # print('w', w.ravel())
    # Compute the error signals
    Re = R_d.T @ R
    # print('Re', Re.ravel())
    psi = twist_d(x, R, t, store_points=False)
    psi_dot_ = twist_derivative_d(x, R, dx, w, t) #TODO
    psi_next = twist_d(x + dx * dt, expm(skew(w) * dt) @ R, t, store_points=False)
    psi_dot = (psi_next - psi) / (dt)
    psi_dot[:3, :] = psi_dot_[:3, :]
    aprox_hist.append(psi_dot)
    # psi_old = psi.copy()
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
        tau = G_h_inv @ F[i]  # compensate for est. torque
        input_ += tau  # compute real applied wrench; add to running

    input_hist.append(input_)
    ddq = root(
        lambda x: (H @ x.reshape(-1, 1) + C @ dq - input_).ravel(), np.zeros((6,)), method='lm'
    ).x.reshape(-1, 1)
    # ddq = np.clip(ddq, -0.1, 0.1)
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

    return ddq, da, dr

#%%
# Physical parameters
rho = 8050  # Density of steel in kg/m^3
r = 0.25  # Radius of the cylinder in meters
h = 1  # Height of the cylinder in meters
# r = 0.25  # Radius of the cylinder in meters
# h = 1.5  # Height of the cylinder in meters
m = rho * np.pi * r**2 * h  # Mass of the cylinder in kg
num_freq = 5  # Number of frequencies to use in the trajectory
rng = np.random.default_rng(42)  # Random number generator

# For fixed sample
r_p = np.array([[0], [0], [h / 2]])  # Measurement point

N=6
r_i = np.array(
    [[0, 0, h / 2], [0, 0, -h / 2], [r, 0, 0], [-r, 0, 0], [0, r, 0], [0, -r, 0]]
).reshape(
    -1, 3, 1
)  # Initial positions of the agents 

# Inertia Tensor
I_cm = (1 / 12) * np.diag([m * (3 * r**2 + h**2), m * (3 * r**2 + h**2), 6 * m * r**2])
I_p = I_cm - m * skew(r_p) @ skew(r_p) 

########################################
a_hat = rng.normal(0, 1, (N, 10, 1))  # Initial parameter estimates

# True ai
a_I = np.array(
    [[I_p[0, 0]], [I_p[0, 1]], [I_p[0, 2]], [I_p[1, 1]], [I_p[1, 2]], [I_p[2, 2]]]
)
a = np.vstack([m, m * r_p, a_I])
a_i = (1 / N) * a

# Initial r_hat
r_hat = 2 * rng.normal(0, 1, (N, 3, 1))

########################################
maxtheta = 5000 #6284
T = 20
dt = 1e-2
# eq = parametric_eq_factory(w1=0, w2=0, c1=.3, c2=.3, c3=0, h0=0.4, maxtheta=maxtheta, T=2, dt=1e-2, timedependent=False)
# eq = parametric_eq_factory(w1=0, w2=0, c1=1, c2=1, c3=0, h0=0.4, maxtheta=maxtheta, T=2, dt=1e-2, timedependent=False)
eq = parametric_eq_factory(w1=0, w2=0, c1=0.7, c2=0.7, c3=0, h0=0.4, maxtheta=maxtheta, T=2, dt=1e-2, timedependent=False)
vf = VectorField(eq, False, kf=5, vr=1, wr=70, beta=1, dt=1e-2)

Kd = 1e-1*np.diag(np.hstack([(5e4 / N) * np.ones(3), (5e3 / N) * np.ones(3)]))
# Kd = 1*np.diag(np.hstack([(5e4 / N) * np.ones(3), (5e3 / N) * np.ones(3)]))
R = np.eye(3)
R_d = np.eye(3)
dq = np.zeros((6, 1))  # Initial velocity
w, x, x_d, dx, s = (
    np.zeros((3, 1)),
    np.zeros((3, 1)),
    np.zeros((3, 1)),
    np.zeros((3, 1)),
    np.zeros((6, 1)),
)
x = np.array([[-0.1], [0], [0.2]])
psi_old = np.zeros((6, 1))

# Simulation parameters
lambda_ = 1.5
deadband = 0.01  # Deadband in which to stop adaptation
t_drop = T + 10  # Time to turn off agents
p = 2 
P_o = 3e1 * np.linalg.inv(np.diag(np.abs(a_i.ravel()) + 1e-2)) #3e1 #TODO 3e2 better
P_r = 3e3 * np.eye(3)
tol = 1e-5
verbose=False

# Storage for simulation data
s_t, rot_err, V, Vdot, s_hat, Vhat, Vdot = [], [], [], [], [], [], []
lyap_params, lyap_dynamics, err_mean, param_est, arm_est = [], [], [], [], []
dV_params, dV_dynamics, hist_x, hist_dx, hist_vf = [], [], [], [], []
hist_xd, hist_R, hist_w, hist_s, hist_ddq = [], [], [], [], []
aprox_hist, input_hist = [], []
dropped = False

for t in np.arange(0, T + dt, dt):
    index = int(t / dt)
    index_max = int((T + dt) / dt)
    progress_bar(index, index_max)

    dx = dq[:3].copy()
    w = dq[3:].copy()
    psi = twist_d(x, R, t, store_points=True)
    if np.iscomplex(psi).any():
        print('Complex number found')
    # psi_dot = twist_derivative_d(x, R, dx, w, t)
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
    # V_curr, V_params = 0.5 * s.T @ H @ s, 0.0

    for i in range(N):
        g_o[i] = np.linalg.inv(P_o) @ np.linalg.inv(
            pnorm_hessian(P_o @ a_hat[i], p, tol) @ np.linalg.inv(P_o)
        )
        g_r[i] = np.linalg.inv(P_r) @ np.linalg.inv(
            pnorm_hessian(P_r @ r_hat[i], p, tol) @ np.linalg.inv(P_r)
        )
        a_t[i] = a_hat[i] - a_i
        r_t[i] = r_hat[i] - r_i[i]
        # aux = pnorm_divergence(P_o @ a_i, P_o @ a_hat[i], p, tol) + pnorm_divergence(
        #     P_r @ r_i[i], P_r @ r_hat[i], p, tol
        # )
        # V_params += aux
        # V_curr += aux

    # V = V_curr
    # if t > 0:
    #     pass
    #     # lyap_params = ...
        # lyap_obj = ...

    at_mean = (1 / N) * np.sum(a_t, axis=0)
    rt_mean = (1 / N) * np.sum(r_t, axis=0)

    # Logging
    err_mean.append(np.linalg.norm(at_mean) + np.linalg.norm(rt_mean))
    s_t.append(np.linalg.norm(s))
    hist_s.append(s)
    rot_err.append(np.linalg.norm(R_d.T @ R - np.eye(3), 'fro'))
    # lyap_params.append(V_params)
    # lyap_dynamics.append(0.5 * s.T @ H @ s)

    # Get dynamics at current time
    ddq, da, dr = adaptive_dyn(
        x, x_d, R, R_d, dq, a_hat, r_hat, t, lambda_, Kd, t_drop, verbose
    )

    # if t == 0:
    #     Vhat.append(V[-1])
    #     s_hat.append(s)

    a_int, r_int = np.zeros((N, 10, 1)), np.zeros((N, 3, 1))

    # Intermediate step of Heun's
    for i in range(N):
        if np.linalg.norm(s) > deadband:
            a_hat[i] = a_hat[i] + dt * da[i]
            r_hat[i] = r_hat[i] + dt * dr[i]
        else:
            da[i] = np.zeros((10, 1))
            dr[i] = np.zeros((3, 1))
            a_int[i] = a_hat[i]
            r_int[i] = r_hat[i]

    dq = dq + dt * ddq
    R_d = expm(dt * skew(psi[3:, :])) @ R_d
    R = expm(dt * skew(w)) @ R
    x_d = x_d + dt * psi[:3, :]
    dx = dq[:3].copy()
    w = dq[3:].copy()
    x = x + dt * dx
    

    # dx = dq[:3].copy()
    # w = dq[3:].copy()
    psi_old = psi.copy()

    hist_x.append(x)
    hist_dx.append(dx)
    hist_vf.append(vf_aux)
    hist_xd.append(x_d)
    hist_R.append(R)
    hist_w.append(w)
    hist_ddq.append(ddq)
    param_est.append(a_hat.reshape(-1, 1))
    arm_est.append(r_hat.reshape(-1, 1))

# %%
import plotly.express as px
import pickle

pp, rr = zip(*vf.nearest_points)
# with open('vfresults.pkl', 'wb') as f:
#     data = {
#         's_t': s_t,
#         'hist_x': hist_x,
#         'hist_R': hist_R,
#         'hist_dx': hist_dx,
#         'hist_vf': hist_vf,
#         'hist_w': hist_w,
#         'nearest_points': pp,
#         'nearest_rotations': rr,
#     }
#     pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)
time = np.arange(0, T+dt, dt)
fig = px.line(x=time, y=np.array(s_t))
fig.update_layout(xaxis_title='Time (s)', yaxis_title='Norm of the error vector', 
                  xaxis_title_font=dict(size=22), yaxis_title_font=dict(size=22), 
                  yaxis_tickfont_size=20, width=1200, height=600, 
                  margin=dict(t=0, b=0, r=0, l=5), xaxis_tickfont_size=20)
fig.show()
# px.line(np.linalg.norm(np.array(hist_x).reshape(-1, 3) - np.array(pp).reshape(-1, 3), axis=1)).show()
fro_norms = []
for rot, rot_d in zip(hist_R, rr):
    fro_norms.append(0.5 * np.linalg.norm(np.eye(3) - rot_d.T @ rot)**2)
fig2 = px.line(x=np.arange(0, T+dt, dt), y=np.array(fro_norms) + 
               0.5*(np.linalg.norm(np.array(hist_x).reshape(-1, 3) 
                                   - np.array(pp).reshape(-1, 3), axis=1)**2))
fig2.update_layout(xaxis_title='Time (s)', yaxis_title='Value of metric <i>D</i>', 
                  xaxis_title_font=dict(size=22), yaxis_title_font=dict(size=22), 
                  yaxis_tickfont_size=20, width=1200, height=600, 
                  margin=dict(t=0, b=0, r=0, l=5), xaxis_tickfont_size=20)
fig2.show()
px.line(np.array(hist_s).reshape(-1, 6)[:, :3]).show()
px.line(np.array(hist_x).reshape(-1, 3) - np.array(pp).reshape(-1, 3)).show()
# %%
with open('vfresults.pkl', 'rb') as f:
    data = pickle.load(f)

s_t = data['s_t']
hist_x = data['hist_x']
hist_R = data['hist_R']
pp = data['nearest_points']
rr = data['nearest_rotations']
# %%
