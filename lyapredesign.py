# %%
import sys
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import plotly.figure_factory as ff
import pickle

sys.path.insert(1, "/home/fbartelt/Documents/UFMG/TCC/Sim/uaibot")
import uaibot as ub
from uaibot.utils import Utils
from uaibot.simobjects.pointcloud import PointCloud
from uaibot.simulation import Simulation
from uaibot.simobjects.pointlight import PointLight
from uaibot.simobjects.ball import Ball
from uaibot.robot._vector_field import _compute_ntd as compute_ntd
from cvxopt import matrix, solvers, spmatrix
from create_jaco import create_jaco2
from plotly.subplots import make_subplots
from itertools import product
from uaibot_addons.vectorfield import VectorField, vector_field_plot
from uaibot_addons.math import dot_J
from scipy.linalg import solve_continuous_are, solve_continuous_lyapunov


def progress_bar(i, imax):
    sys.stdout.write("\r")
    sys.stdout.write(
        "[%-20s] %d%%" % ("=" * round(20 * i / (imax - 1)), round(100 * i / (imax - 1)))
    )
    sys.stdout.flush()


def psbf(z, epsilon):
    return np.linalg.norm(z) / (epsilon - np.linalg.norm(z))


def rk4(f, t, x, dt, *args, **kwargs):
    k1 = f(t, x, *args, **kwargs, save_hist=True)
    k2 = f(t + dt / 2, x + dt / 2 * k1, *args, **kwargs)
    k3 = f(t + dt / 2, x + dt / 2 * k2, *args, **kwargs)
    k4 = f(t + dt, x + dt * k3, *args, **kwargs)
    return x + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)


# %%
"""Vector Field + Lyapunov Redesign (Mass uncertainty + disturbance) -- WORKING"""
robot = create_jaco2(thesis_parameters=True)
light1 = PointLight(
    name="light1", color="white", intensity=2.5, htm=Utils.trn([-1, -1, 1.5])
)
light2 = PointLight(
    name="light2", color="white", intensity=2.5, htm=Utils.trn([-1, 1, 1.5])
)
light3 = PointLight(
    name="light3", color="white", intensity=2.5, htm=Utils.trn([1, -1, 1.5])
)
light4 = PointLight(
    name="light4", color="white", intensity=2.5, htm=Utils.trn([1, 1, 1.5])
)
sim = Simulation.create_sim_grid([robot])  # , light1, light2, light3, light4])
sim.set_parameters(width=800, height=600, ambient_light_intensity=4)

n = len(robot.links)

# Parametric equation definition
maxtheta = 500


def eq2(time=0):
    ## 0.5e-1, 0.025e-1, 0.6, 0.6, 0.3, 0.3 works with acceleration approx
    ## 0.5e-1, 0.025e-1, 0.6, 0.6, 0.3, 0.3 works with analytic acceleration, Kd=50, const_vel=1, alpha=5
    ## 0.5e-1, 0.025e1, 0.6, 0.6, 0.3, 0.3 works with analytic acceleration, Kd=50, const_vel=1, alpha=5
    ## Above doesnt work with alpha=1
    w1, w2, c1, c2, c3, h0 = 0.5e-1 * 0, 0.025e1 * 0, 0.6, 0.6, 0.3, 0.3
    rotz = np.matrix(
        [
            [np.cos(w1 * time), -np.sin(w1 * time), 0],
            [np.sin(w1 * time), np.cos(w1 * time), 0],
            [0, 0, 1],
        ]
    )
    theta = np.linspace(0, 2 * np.pi, num=maxtheta)
    curve = np.array(
        [
            rotz
            @ np.array(
                [
                    c1 * np.cos(s),
                    c2 * np.sin(s),
                    h0 + c3 * np.cos(w2 * time) * np.cos(s) ** 2,
                ]
            ).reshape(-1, 1)
            for s in theta
        ]
    ).reshape(3, -1)
    curve = curve.reshape(-1, 3).T

    return curve


# Simulation parameters
T = 5
dt = 0.001
imax = int(T / dt)
epsilon = 1e-3
disturbance = 0
small_limit = 0.62832  # 6rpm
big_limit = 0.83776  # 8rpm
qdot_limits = 10 * np.array(
    [[big_limit], [big_limit], [big_limit], [small_limit], [small_limit], [small_limit]]
)

# Add trajectory and nearest point to simulation
print("Creating point cloud")
a = None
for i in range(int(T / 0.01)):
    curve = eq2(i * dt)
    if a is None:
        a = curve
    else:
        a = np.hstack((a, curve))
traj = PointCloud(name="traj", points=a, size=12, color="cyan")
sim.add([traj])
vf = VectorField(eq2, False, alpha=5, const_vel=1.5)
nearest_point = Ball(
    name="nearest_point",
    radius=0.03,
    color="red",
    htm=Utils.trn([0, 0, 0]),
    opacity=0.7,
)
sim.add([nearest_point])
print("Done")

# PD
n = len(robot.links)
A = np.block([[np.zeros((n, n)), np.eye(n)], [np.zeros((n, n)), np.zeros((n, n))]])
B = np.block([[np.zeros((n, n))], [np.eye(n)]])
# Q = 0.01*np.diag([30]*n + [30]*n)
Q = np.diag([0.1] * n + [30] * n)
R = np.eye(n)
P = solve_continuous_are(A, B, Q, R)
K = np.linalg.inv(R) @ B.T @ P
Kp, Kd = np.split(K, 2, axis=1)

A_lyap = np.block([[np.zeros((n, n)), np.eye(n)], [-Kp, -Kd]])
# A_lyap = np.block([[-K], [np.block([np.eye(n), np.zeros((n, n))])]])
Q_lyap = -np.eye(2 * n)
P_lyap = solve_continuous_lyapunov(A_lyap, Q_lyap)

# Initializations
q_des = np.array(
    [[0.7262458], [1.61760955], [0.11582987], [-1.14679451], [2.16399157], [2.76812822]]
)
q = robot.q.copy()
qdot = np.zeros((n, 1))
qdot_des = np.zeros((n, 1))
qdot = np.zeros((n, 1))

# Plot-related
hist_time = []
hist_qdot = np.matrix(np.zeros((n, 0)))
hist_qdot_des = np.matrix(np.zeros((n, 0)))
hist_qddot = np.matrix(np.zeros((n, 0)))
hist_qddot_des = np.matrix(np.zeros((n, 0)))
hist_q = np.matrix(np.zeros((n, 0)))
hist_error_ori = np.matrix(np.zeros((n, 0)))
hist_peef = np.zeros((3, 0))
hist_vf = np.zeros((3, 0))
hist_cond_J = []
hist_cond_Jdot = []
hist_x = np.matrix(np.zeros((2 * n, 0)))
hist_torque = np.matrix(np.zeros((n, 0)))

for i in range(1, imax):
    progress_bar(i, imax)
    # Vector Field related
    jac_eef, htm_eef = robot.jac_geo()
    p_eef = htm_eef[0:3, 3]
    target = np.matrix(np.zeros((3, 1)))
    target[0:3] = vf(p_eef, i * dt)
    jac_target = np.matrix(np.zeros((3, n)))
    jac_target[0:3, :] = jac_eef[0:3, :]
    qdot_des = np.linalg.pinv(jac_target) @ target
    q_des = q + dt * qdot_des
    a_des = vf.acceleration(p_eef, jac_target @ qdot, i * dt)
    Jdot = dot_J(robot, qdot, q)[:3, :]
    qddot_des = np.linalg.pinv(jac_target) @ (a_des - Jdot @ qdot)

    if i >= 1.2 / dt:
        disturbance = 1e-1 * np.ones((n, 1))
        if i >= 4 / dt:
            disturbance = np.sin(0.2 * i * dt) * disturbance

    # Lyapunov Redesign related
    x = np.block(
        [
            [q - q_des],
            [qdot - qdot_des],
        ]
    )
    w = B.T @ P @ x
    w_norm = np.linalg.norm(w)
    K = np.block([Kp, Kd])
    phi = (
        18.19441564410192 + 10 + 1.1210095929589532 * np.linalg.norm(x)
    )  # Bound G, disturbance and ||C\dot{q}||
    phi = 1.12 * np.linalg.norm(x)
    # alpha = 0.8 # (Mmax - Mmin)/(Mmax + Mmin) <= alpha
    alpha = 0.2  # 0.1
    # rho = 1/(1 - alpha)  * (alpha*np.linalg.norm(K, 2) * np.linalg.norm(x) + 1/0.1640573139501184*phi)
    rho = 1 / (1 - alpha) * (alpha * np.linalg.norm(K, 2) * np.linalg.norm(x) + 1400)
    if w_norm >= epsilon:
        delta_v = -rho * (w / w_norm)
    else:
        delta_v = -rho * (w / epsilon)

    M_, C_, G_ = robot.dyn_model(q, qdot)
    # M = np.eye(6) * 2/((1/0.1640573139501184) + (1/1.4397118507671303))
    rng = np.random.default_rng()
    M = (1 + alpha) * M_  # 1.1

    v = qddot_des - Kp @ (q - q_des) - Kd @ (qdot - qdot_des) + delta_v
    torque = C_ + G_ + (M_ @ v)
    # eta = np.linalg.inv(M) @ ((M_ - M) @ v + (C_ - C) + (G_ - G) + disturbance)
    eta = np.linalg.inv(M) @ ((M_ - M) @ v + disturbance)
    qddot = v + eta
    q = robot.q + qdot * dt
    # qdot = qdot + qddot * dt
    qdot = np.clip(qdot + qddot * dt, -qdot_limits, qdot_limits)
    robot.add_ani_frame(time=i * dt, q=q)
    # traj.add_ani_frame(time=i*dt, initial_ind=maxtheta*i, final_ind=maxtheta*(i+1))
    nearest_point.add_ani_frame(time=i * dt, htm=Utils.trn(vf.nearest_points[-1]))

    hist_time.append(i * dt)
    hist_q = np.block([hist_q, robot.q])
    hist_peef = np.block([hist_peef, p_eef])
    # hist_vf = np.block([hist_vf, target[0:3]])
    hist_qdot = np.block([hist_qdot, qdot])
    hist_qdot_des = np.block([hist_qdot_des, qdot_des])
    hist_qddot = np.block([hist_qddot, qddot])
    hist_qddot_des = np.block([hist_qddot_des, qddot_des])
    hist_torque = np.block([hist_torque, torque])
    # hist_cond_J.append(np.linalg.cond(jac_target))
    # hist_cond_Jdot.append(np.linalg.cond(Jdot))
    hist_x = np.block([hist_x, x])

# # with open('data_computedtorque.pkl', 'wb') as f:
# #     data = {'hist_time': hist_time,
# #             'hist_qdot': hist_qdot,
# #             'hist_qdot_des': hist_qdot_des,
# #             'hist_qddot': hist_qddot,
# #             'hist_qddot_des': hist_qddot_des,
# #             'hist_q': hist_q,
# #             'hist_peef': hist_peef,
# #             'hist_x': hist_x,
# #             'hist_torque': hist_torque,
# #             'nearest_points': vf.nearest_points}
# #     pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
# # sim.save('figures', 'computed_torque')

sim.run()
# hist_vf = np.array(hist_vf)
hist_peef = np.array(hist_peef)
# fig = vector_field_plot(hist_peef, hist_vf, add_lineplot=True, sizemode="absolute", sizeref=2.5, anchor='tail')
# fig.write_image("figures/vectorfield.pdf")
# fig.show()
fig = px.line(np.linalg.norm(hist_x, axis=0).T, title="|x|")
# fig.write_image("figures/histx.pdf")
fig.show()
fig = px.line(
    np.linalg.norm(hist_qdot - hist_qdot_des, axis=0).T,
    title="|dq/dt - dq<sub>des</sub>/dt|",
)
fig.show()
# fig.write_image("figures/qdoterrNorm.pdf")
fig = px.line(
    np.abs(hist_qdot - hist_qdot_des).T, title="abs(dq/dt - dq<sub>des</sub>/dt)"
)
# fig.write_image("figures/qdoterr.pdf")
fig.show()
fig = px.line(
    np.abs(hist_peef - np.array(vf.nearest_points).reshape(-1, 3).T).T,
    title="|p<sub>eef</sub> - x*|",
)
fig.show()
# fig=px.line(hist_cond_J, title='J condition number')
# fig.show()
# fig=px.line(hist_cond_Jdot, title=r'\dot{J} condition number')
# fig.show()


# %%
"""Vector Field + Adaptive Lyapunov Redesign (Mass uncertainty + disturbance) -- UNDER DEVELOPMENT
||peef - x|| <= 0.09 em t=4.202s para os parametros do commit anterior
||peef - x|| <= 0.07 em t≃3.8s para xi = np.diag([5e-1, 1e-2]) aumentar o primeiro termo p 7e-1 piora"""
robot = create_jaco2(thesis_parameters=True)
light1 = PointLight(
    name="light1", color="white", intensity=2.5, htm=Utils.trn([-1, -1, 1.5])
)
light2 = PointLight(
    name="light2", color="white", intensity=2.5, htm=Utils.trn([-1, 1, 1.5])
)
light3 = PointLight(
    name="light3", color="white", intensity=2.5, htm=Utils.trn([1, -1, 1.5])
)
light4 = PointLight(
    name="light4", color="white", intensity=2.5, htm=Utils.trn([1, 1, 1.5])
)
sim = Simulation.create_sim_grid([robot])  # , light1, light2, light3, light4])
sim.set_parameters(width=800, height=600, ambient_light_intensity=4)

n = len(robot.links)

# Parametric equation definition
maxtheta = 500


def eq2(time=0):
    ## 0.5e-1, 0.025e-1, 0.6, 0.6, 0.3, 0.3 works with acceleration approx
    ## 0.5e-1, 0.025e-1, 0.6, 0.6, 0.3, 0.3 works with analytic acceleration, Kd=50, const_vel=1, alpha=5
    ## 0.5e-1, 0.025e1, 0.6, 0.6, 0.3, 0.3 works with analytic acceleration, Kd=50, const_vel=1, alpha=5
    ## Above doesnt work with alpha=1
    w1, w2, c1, c2, c3, h0 = 0.5e-1 * 0, 0.025e1 * 0, 0.6, 0.6, 0.3, 0.3
    rotz = np.matrix(
        [
            [np.cos(w1 * time), -np.sin(w1 * time), 0],
            [np.sin(w1 * time), np.cos(w1 * time), 0],
            [0, 0, 1],
        ]
    )
    theta = np.linspace(0, 2 * np.pi, num=maxtheta)
    curve = np.array(
        [
            rotz
            @ np.array(
                [
                    c1 * np.cos(s),
                    c2 * np.sin(s),
                    h0 + c3 * np.cos(w2 * time) * np.cos(s) ** 2,
                ]
            ).reshape(-1, 1)
            for s in theta
        ]
    ).reshape(3, -1)
    curve = curve.reshape(-1, 3).T

    return curve


# Simulation parameters
T = 5
dt = 0.001
imax = int(T / dt)
small_limit = 0.62832  # 6rpm
big_limit = 0.83776  # 8rpm
qdot_limits = 10 * np.array(
    [[big_limit], [big_limit], [big_limit], [small_limit], [small_limit], [small_limit]]
)

# Add trajectory and nearest point to simulation
print("Creating point cloud")
a = None
for i in range(1):
    curve = eq2(i * dt)
    if a is None:
        a = curve
    else:
        a = np.hstack((a, curve))
traj = PointCloud(name="traj", points=a, size=12, color="cyan")
sim.add([traj])
vf = VectorField(eq2, False, alpha=5, const_vel=1.5)
nearest_point = Ball(
    name="nearest_point",
    radius=0.03,
    color="red",
    htm=Utils.trn([0, 0, 0]),
    opacity=0.7,
)
sim.add([nearest_point])
print("Done")

# PD
n = len(robot.links)
A = np.block([[np.zeros((n, n)), np.eye(n)], [np.zeros((n, n)), np.zeros((n, n))]])
B = np.block([[np.zeros((n, n))], [np.eye(n)]])
# Q = 0.01*np.diag([30]*n + [30]*n)
Q = np.diag([0.1] * n + [30] * n)
R = np.eye(n)
P = solve_continuous_are(A, B, Q, R)
K = np.linalg.inv(R) @ B.T @ P
Kp, Kd = np.split(K, 2, axis=1)

A_lyap = np.block([[np.zeros((n, n)), np.eye(n)], [-Kp, -Kd]])
# A_lyap = np.block([[-K], [np.block([np.eye(n), np.zeros((n, n))])]])
Q_lyap = -np.eye(2 * n)
P_lyap = solve_continuous_lyapunov(A_lyap, Q_lyap)

# Initializations
q_des = np.array(
    [[0.7262458], [1.61760955], [0.11582987], [-1.14679451], [2.16399157], [2.76812822]]
)
q = robot.q.copy()
qdot = np.zeros((n, 1))
qdot_des = np.zeros((n, 1))
qdot = np.zeros((n, 1))
L = np.eye(2) * 2
xi = np.diag([5e-1, 1e-2])
epsilon = 0.3  # (1**2) * (np.min(np.linalg.eigvals(Q)) * np.min(np.linalg.eigvals(P)) / np.max(np.linalg.eigvals(P)) )
l = 10
b0 = np.array([[7], [850]])  # 6.55737195, 854.96277489
rho0 = 110
b = b0
rho = rho0

# Plot-related
hist_time = []
hist_qdot = np.matrix(np.zeros((n, 0)))
hist_qdot_des = np.matrix(np.zeros((n, 0)))
hist_qddot = np.matrix(np.zeros((n, 0)))
hist_qddot_des = np.matrix(np.zeros((n, 0)))
hist_q = np.matrix(np.zeros((n, 0)))
hist_error_ori = np.matrix(np.zeros((n, 0)))
hist_peef = np.zeros((3, 0))
hist_vf = np.zeros((3, 0))
hist_cond_J = []
hist_cond_Jdot = []
hist_x = np.matrix(np.zeros((2 * n, 0)))
hist_torque = np.matrix(np.zeros((n, 0)))
hist_v = np.matrix(np.zeros((n, 0)))
hist_eta = np.matrix(np.zeros((n, 0)))
hist_psbf = []

for i in range(1, imax):
    progress_bar(i, imax)
    # Vector Field related
    jac_eef, htm_eef = robot.jac_geo()
    p_eef = htm_eef[0:3, 3]
    target = np.matrix(np.zeros((3, 1)))
    target[0:3] = vf(p_eef, i * dt)
    jac_target = np.matrix(np.zeros((3, n)))
    jac_target[0:3, :] = jac_eef[0:3, :]
    qdot_des = np.linalg.pinv(jac_target) @ target
    q_des = q + dt * qdot_des
    a_des = vf.acceleration(
        p_eef, jac_target @ qdot_des, i * dt
    )  # change qdot to qdot_des
    Jdot = dot_J(robot, qdot, q)[:3, :]
    qddot_des = np.linalg.pinv(jac_target) @ (a_des - Jdot @ qdot)

    if i >= 1.2 / dt:
        disturbance = 1e-1 * np.ones((n, 1))
        if i >= 4 / dt:
            disturbance = np.sin(0.2 * i * dt) * disturbance
        # if i>= 2.8/dt:
        #     disturbance = 0 * disturbance
    else:
        disturbance = np.zeros((n, 1))

    # Lyapunov Redesign related
    x = np.block(
        [
            [q - q_des],
            [qdot - qdot_des],
        ]
    )
    w = B.T @ P @ x
    w_norm = np.linalg.norm(w)
    K = np.block([Kp, Kd])
    kappa = np.block([[1], [np.linalg.norm(x)]])
    gamma = (kappa.T @ b).item()
    alpha = 0.2  # 0.1

    if w_norm >= (epsilon / 2):
        delta_v = -w - gamma * w / w_norm - rho * w / (w_norm**2)
        hist_psbf.append(False)
    else:
        delta_v = -psbf(w, epsilon) * (w / w_norm)
        hist_psbf.append(True)

    M_, C_, G_ = robot.dyn_model(q, qdot)
    # M = np.eye(6) * 2/((1/0.1640573139501184) + (1/1.4397118507671303))
    rng = np.random.default_rng()
    M = (1 + alpha) * M_  # 1.1

    v = qddot_des - Kp @ (q - q_des) - Kd @ (qdot - qdot_des) + delta_v
    torque = C_ + G_ + (M_ @ v)
    # eta = np.linalg.inv(M) @ ((M_ - M) @ v + (C_ - C) + (G_ - G) + disturbance)
    eta = np.linalg.inv(M) @ ((M_ - M) @ v + disturbance)
    qddot = v + eta
    bdot = L @ (kappa * w_norm - xi @ b)
    rhodot = l - rho

    q = robot.q + qdot * dt
    # qdot = qdot + qddot * dt
    qdot = np.clip(qdot + qddot * dt, -qdot_limits, qdot_limits)
    b = b + bdot * dt
    rho = rho + rhodot * dt
    robot.add_ani_frame(time=i * dt, q=q)
    # traj.add_ani_frame(time=i*dt, initial_ind=maxtheta*i, final_ind=maxtheta*(i+1))
    nearest_point.add_ani_frame(time=i * dt, htm=Utils.trn(vf.nearest_points[-1]))

    hist_time.append(i * dt)
    hist_q = np.block([hist_q, robot.q])
    hist_peef = np.block([hist_peef, p_eef])
    # hist_vf = np.block([hist_vf, target[0:3]])
    hist_qdot = np.block([hist_qdot, qdot])
    hist_qdot_des = np.block([hist_qdot_des, qdot_des])
    hist_qddot = np.block([hist_qddot, qddot])
    hist_qddot_des = np.block([hist_qddot_des, qddot_des])
    hist_torque = np.block([hist_torque, torque])
    hist_cond_J.append(np.linalg.cond(jac_target))
    # hist_cond_Jdot.append(np.linalg.cond(Jdot))
    hist_x = np.block([hist_x, x])
    hist_v = np.block([hist_v, v])
    hist_eta = np.block([hist_eta, eta])


# Save data and simulation
# # with open('data_adaptive_lyap.pkl', 'wb') as f:
# #     data = {'hist_time': hist_time,
# #             'hist_qdot': hist_qdot,
# #             'hist_qdot_des': hist_qdot_des,
# #             'hist_qddot': hist_qddot,
# #             'hist_qddot_des': hist_qddot_des,
# #             'hist_q': hist_q,
# #             'hist_peef': hist_peef,
# #             'hist_x': hist_x,
# #             'hist_torque': hist_torque,
# #             'nearest_points': vf.nearest_points}
# #     pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
# # sim.save('figures', 'cbf_adaptive_lyap')
sim.run()
# hist_vf = np.array(hist_vf)
hist_peef = np.array(hist_peef)
# fig = vector_field_plot(hist_peef, hist_vf, add_lineplot=True, sizemode="absolute", sizeref=2.5, anchor='tail')
# fig.write_image("figures/vectorfield.pdf")
# fig.show()
fig = px.line(np.linalg.norm(hist_x, axis=0).T, title="|x|")
# fig.write_image("figures/histx.pdf")
fig.show()
fig = px.line(
    np.linalg.norm(hist_qdot - hist_qdot_des, axis=0).T,
    title="|dq/dt - dq<sub>des</sub>/dt|",
)
fig.show()
# fig.write_image("figures/qdoterrNorm.pdf")
fig = px.line(
    np.abs(hist_qdot - hist_qdot_des).T, title="abs(dq/dt - dq<sub>des</sub>/dt)"
)
# fig.write_image("figures/qdoterr.pdf")
fig.show()
fig = px.line(
    np.linalg.norm(hist_peef - np.array(vf.nearest_points).reshape(-1, 3).T, axis=0).T,
    title="|p<sub>eef</sub> - x*|",
)
fig.show()
# fig=px.line(hist_cond_J, title='J condition number')
# fig.show()
# fig=px.line(hist_cond_Jdot, title=r'\dot{J} condition number')
# fig.show()
# %%
import pickle
import plotly.express as px
import numpy as np

with open("data_computedtorque.pkl", "rb") as f:
    data_computedtorque = pickle.load(f)
with open("data_adaptive_lyap.pkl", "rb") as f:
    data_adaptive_lyap = pickle.load(f)
with open("data_lyap_redesign.pkl", "rb") as f:
    data_lyap_redesign = pickle.load(f)


def plot_all(data):
    hist_time = data["hist_time"]
    hist_qdot = data["hist_qdot"]
    hist_qdot_des = data["hist_qdot_des"]
    hist_qddot = data["hist_qddot"]
    hist_qddot_des = data["hist_qddot_des"]
    hist_q = data["hist_q"]
    hist_peef = data["hist_peef"]
    hist_x = data["hist_x"]
    hist_torque = data["hist_torque"]
    nearest_points = data["nearest_points"]

    fig = px.line(np.linalg.norm(hist_x, axis=0).T, title="||x||")
    fig.show()
    fig = px.line(hist_qdot_des.T, title="dq<sub>des</sub>/dt")
    fig.show()
    fig = px.line((hist_qdot).T, title="dq/dt")
    fig.show()
    fig = px.line(hist_qddot_des.T, title="d<sup>2</sup>q<sub>des</sub>/dt<sup>2</sup>")
    fig.show()
    fig = px.line((hist_qddot).T, title="d<sup>2</sup>q/dt<sup>2</sup>")
    fig.show()
    fig = px.line(
        np.abs(np.array(hist_peef) - np.array(nearest_points).reshape(-1, 3).T).T,
        title="|p<sub>eef</sub> - x*|",
    )
    fig.show()
    fig = px.line(hist_torque.T, title="Torque")
    fig.show()


plot_all(data_lyap_redesign)
# %%
robot = create_jaco2(thesis_parameters=True)
light1 = PointLight(
    name="light1", color="white", intensity=2.5, htm=Utils.trn([-1, -1, 1.5])
)
light2 = PointLight(
    name="light2", color="white", intensity=2.5, htm=Utils.trn([-1, 1, 1.5])
)
light3 = PointLight(
    name="light3", color="white", intensity=2.5, htm=Utils.trn([1, -1, 1.5])
)
light4 = PointLight(
    name="light4", color="white", intensity=2.5, htm=Utils.trn([1, 1, 1.5])
)
sim = Simulation.create_sim_grid([robot])  # , light1, light2, light3, light4])
sim.set_parameters(width=800, height=600, ambient_light_intensity=4)

n = len(robot.links)

# Parametric equation definition
maxtheta = 500


def eq2(time=0):
    ## 0.5e-1, 0.025e-1, 0.6, 0.6, 0.3, 0.3 works with acceleration approx
    ## 0.5e-1, 0.025e-1, 0.6, 0.6, 0.3, 0.3 works with analytic acceleration, Kd=50, const_vel=1, alpha=5
    ## 0.5e-1, 0.025e1, 0.6, 0.6, 0.3, 0.3 works with analytic acceleration, Kd=50, const_vel=1, alpha=5
    ## Above doesnt work with alpha=1
    w1, w2, c1, c2, c3, h0 = 0.5e-1 * 0, 0.025e1 * 0, 0.6, 0.6, 0.3, 0.3
    rotz = np.matrix(
        [
            [np.cos(w1 * time), -np.sin(w1 * time), 0],
            [np.sin(w1 * time), np.cos(w1 * time), 0],
            [0, 0, 1],
        ]
    )
    theta = np.linspace(0, 2 * np.pi, num=maxtheta)
    curve = np.array(
        [
            rotz
            @ np.array(
                [
                    c1 * np.cos(s),
                    c2 * np.sin(s),
                    h0 + c3 * np.cos(w2 * time) * np.cos(s) ** 2,
                ]
            ).reshape(-1, 1)
            for s in theta
        ]
    ).reshape(3, -1)
    curve = curve.reshape(-1, 3).T

    return curve


def disturbance(t):
    if t >= 1.2:
        return 1e-1 * np.ones((n, 1))
    else:
        return np.zeros((n, 1))


# Simulation parameters
T = 1
dt = 0.001
imax = int(T / dt)
small_limit = 0.62832  # 6rpm
big_limit = 0.83776  # 8rpm
qdot_limits = 10 * np.array(
    [[big_limit], [big_limit], [big_limit], [small_limit], [small_limit], [small_limit]]
)

n = len(robot.links)
A_lqr = np.block([[np.zeros((n, n)), np.eye(n)], [np.zeros((n, n)), np.zeros((n, n))]])
B_lqr = np.block([[np.zeros((n, n))], [np.eye(n)]])
Q_lqr = np.diag([0.1] * n + [30] * n)
R_lqr = np.eye(n)
P_lqr = solve_continuous_are(A_lqr, B_lqr, Q_lqr, R_lqr)
K = np.linalg.inv(R_lqr) @ B_lqr.T @ P_lqr
Kp, Kd = np.split(K, 2, axis=1)

A = np.block([[np.zeros((n, n)), np.eye(n)], [-Kp, -Kd]])
B = B_lqr
Q = -np.eye(2 * n)
P = solve_continuous_lyapunov(A, Q)

# Add trajectory and nearest point to simulation
print("Creating point cloud")
a = None
for i in range(1):
    curve = eq2(i * dt)
    if a is None:
        a = curve
    else:
        a = np.hstack((a, curve))
traj = PointCloud(name="traj", points=a, size=12, color="cyan")
sim.add([traj])
vf = VectorField(eq2, False, alpha=5, const_vel=1.5, dt=dt)
nearest_point = Ball(
    name="nearest_point",
    radius=0.03,
    color="red",
    htm=Utils.trn([0, 0, 0]),
    opacity=0.7,
)
sim.add([nearest_point])
print("Done")
# Initializations
q = robot.q.copy()
qdot = np.zeros((n, 1))
qdot_des = np.zeros((n, 1))
qdot = np.zeros((n, 1))
L = np.eye(2) * 2
xi = np.diag([5e-1, 1e-2])
epsilon = 0.3  # (1**2) * (np.min(np.linalg.eigvals(Q)) * np.min(np.linalg.eigvals(P)) / np.max(np.linalg.eigvals(P)) )
epsilon = (1**2) * (
    np.min(np.linalg.eigvals(-Q))
    * np.min(np.linalg.eigvals(P))
    / np.max(np.linalg.eigvals(P))
)
l = 10
b0 = np.array([[7], [850]])  # 6.55737195, 854.96277489
rho0 = 110
b = b0
rho = rho0
alpha = 0.2

# Plot-related
hist_time = []
hist_qdot = np.matrix(np.zeros((n, 0)))
hist_qdot_des = np.matrix(np.zeros((n, 0)))
hist_qddot = np.matrix(np.zeros((n, 0)))
hist_qddot_des = np.matrix(np.zeros((n, 0)))
hist_q = np.matrix(np.zeros((n, 0)))
hist_error_ori = np.matrix(np.zeros((n, 0)))
hist_peef = np.zeros((3, 0))
hist_vf = np.zeros((3, 0))
hist_cond_J = []
hist_cond_Jdot = []
hist_x = np.matrix(np.zeros((2 * n, 0)))
hist_torque = np.matrix(np.zeros((n, 0)))
hist_v = np.matrix(np.zeros((n, 0)))
hist_eta = np.matrix(np.zeros((n, 0)))
hist_b = np.matrix(np.zeros((2, 0)))
hist_rho = []
hist_psbf = []


def closed_loop(
    t,
    z,
    n=6,
    l=10,
    xi=np.diag([5e-1, 1e-2]),
    epsilon=0.3,
    alpha=0.2,
    L=np.eye(2),
    disturbance=lambda t: np.zeros((n, 1)),
    save_hist=False,
):
    """z=[q, qdot, b, rho]"""
    global hist_peef, hist_qdot_des, hist_qddot, hist_qddot_des, hist_torque, hist_cond_J, hist_v, hist_eta, hist_psbf
    q = z[:n]
    qdot = z[n : 2 * n]
    b = z[2 * n : 2 * n + 2]
    rho = z[2 * n + 2].item()
    jac_eef, htm_eef = robot.jac_geo(q=q)
    p_eef = htm_eef[0:3, 3]
    target = vf(p_eef, t)
    jac_target = jac_eef[0:3, :]
    qdot_des = np.linalg.pinv(jac_target) @ target
    q_des = q + dt * qdot_des
    a_des = vf.acceleration(p_eef, target, t)  # change qdot to qdot_des
    Jdot = dot_J(robot, qdot, q)[:3, :]
    qddot_des = np.linalg.pinv(jac_target) @ (a_des - Jdot @ qdot)

    x = np.block(
        [
            [q - q_des],
            [qdot - qdot_des],
        ]
    )
    M_, C_, G_ = robot.dyn_model(q, qdot)
    M = (1 + alpha) * M_  # 1.1
    w = B.T @ P @ x
    w_norm = np.linalg.norm(w)
    kappa = np.block([[1], [np.linalg.norm(x)]])
    gamma = (kappa.T @ b).item()
    alpha = 0.2  # 0.1
    psbf_active = False

    if w_norm >= (epsilon / 2):
        v = -w - gamma * w / w_norm - rho * w / (w_norm**2)
    else:
        v = -psbf(w, epsilon) * (w / w_norm)
        psbf_active = True

    a = qddot_des - Kp @ (q - q_des) - Kd @ (qdot - qdot_des) + v
    torque = C_ + G_ + (M_ @ a)
    # eta = np.linalg.inv(M) @ ((M_ - M) @ v + (C_ - C) + (G_ - G) + disturbance)
    eta = np.linalg.inv(M) @ ((M_ - M) @ a + disturbance(t))
    xdot = A @ x + B @ (v + eta) + np.block([[qdot_des], [qddot_des]])
    lims = np.block([[qdot_limits * np.inf], [qdot_limits]])
    xdot = np.clip(xdot, -lims, lims)
    bdot = L @ (kappa * w_norm - xi @ b)
    rhodot = l - rho
    zdot = np.block([[xdot], [bdot], [rhodot]])

    qdot = xdot[:n]
    qddot = xdot[n : 2 * n]

    if save_hist:
        hist_peef = np.block([hist_peef, p_eef])
        # hist_vf = np.block([hist_vf, target[0:3]])
        hist_qdot_des = np.block([hist_qdot_des, qdot_des])
        hist_qddot = np.block([hist_qddot, qddot])
        hist_qddot_des = np.block([hist_qddot_des, qddot_des])
        hist_torque = np.block([hist_torque, torque])
        hist_cond_J.append(np.linalg.cond(jac_target))
        hist_v = np.block([hist_v, v])
        hist_eta = np.block([hist_eta, eta])
        hist_psbf.append(psbf_active)

    return zdot


z = np.block([[q], [qdot], [b], [rho]])

for i in range(1, imax):
    progress_bar(i, imax)
    t = i * dt
    z = rk4(
        closed_loop,
        t,
        z,
        0.001,
        n=n,
        l=l,
        xi=xi,
        epsilon=epsilon,
        alpha=alpha,
        L=L,
        disturbance=disturbance,
    )

    q = z[:n]
    qdot = z[n : 2 * n]
    b = z[2 * n : 2 * n + 2]
    rho = z[2 * n + 2].item()

    robot.add_ani_frame(time=t, q=q)
    # nearest_point.add_ani_frame(time=t, htm=Utils.trn(vf.nearest_points[-1]))

    hist_time.append(t)
    hist_q = np.block([hist_q, q])
    # hist_peef = np.block([hist_peef, p_eef])
    hist_qdot = np.block([hist_qdot, qdot])
    # hist_qdot_des = np.block([hist_qdot_des, qdot_des])
    # hist_torque = np.block([hist_torque, torque])
    hist_x = np.block([hist_x, np.block([[q], [qdot]])])
    hist_b = np.block([hist_b, b])
    hist_rho.append(rho)
    # hist_v = np.block([hist_v, v])
    # hist_eta = np.block([hist_eta, eta])

# %%
