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
from uaibot.robot import Robot
from uaibot.simobjects.pointcloud import PointCloud
from uaibot.simulation import Simulation
from uaibot.simobjects.pointlight import PointLight
from uaibot.simobjects.ball import Ball
from uaibot.robot._vector_field import _compute_ntd as compute_ntd
from uaibot.graphics.meshmaterial import MeshMaterial
from uaibot.graphics.model3d import Model3D
from uaibot.simobjects.rigidobject import RigidObject
from uaibot.simobjects.frame import Frame
from uaibot.robot.links import Link
from cvxopt import matrix, solvers, spmatrix
from uaibot_addons.create_jaco import create_jaco2
from plotly.subplots import make_subplots
from itertools import product
from uaibot_addons.vectorfield import VectorField, vector_field_plot
from uaibot_addons.math import dot_J
from scipy.linalg import solve_continuous_are, solve_continuous_lyapunov
from scipy.optimize import root

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

def pelican_dyn_factory(m1=6.5225, m2=2.0458, l1=0.26, l2=0.26, lc1=0.0983, lc2=0.0229, I1=0.1213, I2=0.0116, g=9.81, name='pelican_robot', color='#3e3f42', opacity=1, scale=1):
    DH_theta = np.array([0, 0])
    DH_d = np.array([0, 0])
    DH_a = np.array([l1, l2])
    DH_alpha = np.array([0, 0])

    DH_type = np.array([0, 0])
    link_info = np.array([DH_theta, DH_d, DH_alpha, DH_a, DH_type])

    def pelican_dynamics(q, qdot):
        m11 = m1 * lc1 ** 2 + m2 * (l1 ** 2 + lc2 ** 2 + 2 * l1 * lc2 * np.cos(q[1, :].item())) + I1 + I2
        m12 = m2 * (lc2 ** 2 + l1 * lc2 * np.cos(q[1, :].item())) + I2
        m21 = m12
        m22 = m2 * lc2 ** 2 + I2

        c11 = -m2 * l1 * lc2 * np.sin(q[1, :].item()) * qdot[1, :].item()
        c12 = -m2 * l1 * lc2 * np.sin(q[1, :].item()) * (qdot[0, :].item() + qdot[1, :].item())
        c21 = m2 * l1 * lc2 * np.sin(q[1, :].item()) * qdot[0, :].item()
        c22 = 0

        g1 = (m1 * lc1 + m2 * l1) * g * np.sin(q[0, :].item()) + m2 * lc2 * g * np.sin(q[0, :].item() + q[1, :].item())
        g2 = m2 * lc2 * g * np.sin(q[0, :].item() + q[1, :].item())

        M = np.array([[m11, m12], [m21, m22]])
        C = np.array([[c11, c12], [c21, c22]])
        G = np.array([g1, g2])

        return M, C, G
    
    def create_pelican(name='pelican_robot', color='#3e3f42', opacity=1, scale=1):
        n = link_info.shape[1]
        base_3d_obj = []
        mesh = MeshMaterial(metalness=0.8, clearcoat=0.9, roughness=0.3,
                            normal_scale=[0.5, 0.5], color=color,
                            opacity=opacity, side="DoubleSide")
        # original model is rotated (Robot fron = plane X x Y)
    
        q0 = [-np.pi/2*0, 0]
        Q01 = Utils.rotz(DH_theta[0]) @ Utils.trn([DH_a[0], 0, 0])
        Q02 = Q01 @ Utils.rotz(DH_theta[1]) @ Utils.trn([DH_a[1], 0, 0])

        link0_mth = Utils.inv_htm(Q01)
        base_3d_obj = [
            Model3D(url='https://raw.githubusercontent.com/fbartelt/robotics-experiments/main/models/jaco/base.obj',
                    scale=scale, htm=link0_mth, mesh_material=mesh),]
        link_3d_obj = []
        # Shoulder
        link1_mth = Utils.inv_htm(Q01)
        link_3d_obj.append([
            Model3D(url='https://raw.githubusercontent.com/fbartelt/robotics-experiments/main/models/jaco/shoulder.obj',
                    scale=scale, htm=link1_mth, mesh_material=mesh),])

        # Upper arm + elbow
        link2_mth = Utils.inv_htm(Q01)
        link_3d_obj.append([
            Model3D(url='https://raw.githubusercontent.com/fbartelt/robotics-experiments/main/models/jaco/upperarm.obj',
                    scale=scale, htm=link2_mth, mesh_material=mesh),])
        
        # com_coordinates = [[0.534615, 0, 0.15], [1.5353, 0, 0.15]]
        com_coordinates = [np.array([[lc1 - l1], [0], [0]]), np.array([[lc2 - l2], [0], [0]])]
        # com_coordinates = [np.array([[lc1 - l1], [0], [0]]), np.array([[0], [0], [0]])]
        list_inertia_mat = [] #TODO: Add inertia matrix

        # Use parameters obtained through trimesh
        list_mass = [m1, m2]
        

        # Icm + Steiner theorem (Inertia mat is in respect to DH frame)
        list_inertia_mat.append(np.eye(3) * I1 - list_mass[0] * Utils.S(com_coordinates[0])**2)
        list_inertia_mat.append(np.eye(3) * I2 - list_mass[1] * Utils.S(com_coordinates[1])**2)

        links = []
        for i in range(n):
            links.append(Link(i, theta=link_info[0, i], d=link_info[1, i], alpha=link_info[2, i], a=link_info[3, i], joint_type=link_info[4, i],
                            list_model_3d=link_3d_obj[i], com_coordinates=com_coordinates[i], mass=list_mass[i], inertia_matrix=list_inertia_mat[i]))

        htm_n_eef = Utils.trn([0, 0, 0])
        htm_base_0 = Utils.rotx(np.pi/2)

        # Create joint limits
        joint_limits = np.matrix([[-3*np.pi, 3*np.pi], [-3*np.pi, 3*np.pi]])
        pelican = Robot(name=name, links=links, list_base_3d_obj=base_3d_obj, htm=np.identity(4),
                    htm_base_0=htm_base_0, htm_n_eef=htm_n_eef, q0=q0, eef_frame_visible=True, joint_limits=joint_limits)


        return pelican
    
    pelican = create_pelican(name, color, opacity, scale)

    return pelican_dynamics, pelican

dyn_fun, robot = pelican_dyn_factory()


jac, htm = robot.jac_geo(q=robot.q0, axis='dh')
f1 = Frame(htm=htm[0])
f2 = Frame(htm=htm[1])
# f1.add_ani_frame()
sim = Simulation.create_sim_grid([f1, f2])
# sim.run()
#%%
n = len(robot.links)

# Parametric equation definition
maxtheta = 500

def parametric_eq_factory2(w1, w2, c1, c2, c3, h0, maxtheta, T, dt, timedependent=True):
    theta = np.linspace(0, 2 * np.pi, num=maxtheta)
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    precomputed = []

    if timedependent:
        for time in np.arange(0, T+2*dt, dt):
            cw1t = np.cos(w1 * time)
            sw1t = np.sin(w1 * time)
            cw2t = np.cos(w2 * time)
            rotz = np.matrix([[cw1t, -sw1t, 0], [sw1t, cw1t, 0], [0, 0, 1]])

            curve = np.empty((3, len(theta)))

            for i, _ in enumerate(theta):
                curve[:, i] = rotz @ np.array([
                    c1 * cos_theta[i],
                    c2 * sin_theta[i],
                    h0 + c3 * cw2t * cos_theta[i] ** 2
                ])

            precomputed.append(curve)
    else:
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
        precomputed.append(curve)
        
    def parametric_eq2(time):
        return precomputed[int(time / dt)]
    
    def parametric_eq(time):
        return precomputed


    if timedependent:
        return parametric_eq2
    else:
        return parametric_eq

def disturbance(t):
    return 2 * np.ones((n, 1)) * np.sin(4 * t)

def test_qdotqddot(t):
    omega = 1
    qdot = np.array([omega * np.cos(omega * t), -omega * np.sin(omega * t)]).reshape(-1, 1)
    qddot = np.array([-omega**2 * np.sin(omega * t), -omega**2 * np.cos(omega * t)]).reshape(-1, 1)

    return qdot, qddot

# Simulation parameters
T = 20
dt = 1e-3
imax = int(T / dt)
small_limit = 0.62832  # 6rpm
big_limit = 0.83776  # 8rpm
qdot_limits = 100 * np.array([[big_limit], [big_limit]])
eq2 = parametric_eq_factory2(w1=0, w2=0, c1=0.5, c2=0.5, c3=0.1, h0=0.3, maxtheta=maxtheta, T=T, dt=dt)

n = len(robot.links)
A_lqr = np.block([[np.zeros((n, n)), np.eye(n)], [np.zeros((n, n)), np.zeros((n, n))]])
B_lqr = np.block([[np.zeros((n, n))], [np.eye(n)]])
Q_lqr = np.diag([2] * n + [2] * n)
R_lqr = np.eye(n) * 5.03
P_lqr = solve_continuous_are(A_lqr, B_lqr, Q_lqr, R_lqr)
K = np.linalg.inv(R_lqr) @ B_lqr.T @ P_lqr
Kp, Kd = np.split(K, 2, axis=1)

A = np.block([[np.zeros((n, n)), np.eye(n)], [-Kp, -Kd]])
# B = B_lqr
# Q = -np.eye(2 * n)
# P = solve_continuous_lyapunov(A, Q)
P = P_lqr
B = B_lqr

# Add trajectory and nearest point to simulation
# print("Creating point cloud")
# a = None
# for i in range(1):
#     curve = eq2(i * dt)
#     if a is None:
#         a = curve
#     else:
#         a = np.hstack((a, curve))
# traj = PointCloud(name="traj", points=a, size=12, color="cyan")
# sim.add([traj])
vf = VectorField(eq2, False, alpha=10, const_vel=0.5, dt=dt)
# nearest_point = Ball(
#     name="nearest_point",
#     radius=0.03,
#     color="red",
#     htm=Utils.trn([0, 0, 0]),
#     opacity=0.7,
# )
# sim.add([nearest_point])
print("Done")
# Initializations
q = robot.q.copy()
q_des = np.zeros((n, 1))
qdot = np.zeros((n, 1))
qdot_des = np.zeros((n, 1))
qdot = np.zeros((n, 1))
ratio = 1
# L = np.diag([1, 0.1, 1]) * ratio
# xi = 1 * np.diag([0.1, 10, 0.1]) / ratio
L = np.diag([0.1, 0.1, 1]) * ratio
xi = np.diag([0.01, 0.1, 0.01]) / ratio
epsilon = 0.035  # (1**2) * (np.min(np.linalg.eigvals(Q)) * np.min(np.linalg.eigvals(P)) / np.max(np.linalg.eigvals(P)) )
epsilon = (1**2) * (np.min(np.linalg.eigvals(Q_lqr)) * np.min(np.linalg.eigvals(P_lqr)) / np.max(np.linalg.eigvals(P_lqr)) )
epsilon = 0.05
l = 10
b0 = np.array([[1e-2], [1e-2], [1e-2]])  # 7, 850
rho0 = 2
b = b0
rho = rho0

# Plot-related
hist_time = []
hist_qdot = np.matrix(np.zeros((n, 0)))
hist_qdot_des = np.matrix(np.zeros((n, 0)))
hist_qddot = np.matrix(np.zeros((n, 0)))
hist_qddot_des = np.matrix(np.zeros((n, 0)))
hist_q = np.matrix(np.zeros((n, 0)))
hist_q_des = np.matrix(np.zeros((n, 0)))
hist_error_ori = np.matrix(np.zeros((n, 0)))
hist_peef = np.zeros((3, 0))
hist_vf = np.zeros((3, 0))
hist_cond_J = []
hist_cond_Jdot = []
hist_x = np.matrix(np.zeros((2 * n, 0)))
hist_torque = np.matrix(np.zeros((n, 0)))
hist_v = np.matrix(np.zeros((n, 0)))
hist_eta = np.matrix(np.zeros((n, 0)))
hist_b = np.matrix(np.zeros((b0.shape[0], 0)))
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
    global hist_peef, hist_qdot_des, hist_qddot, hist_qddot_des, hist_torque, hist_cond_J, hist_v, hist_eta, hist_psbf, hist_x, q_des_old, hist_q_des
    q = z[:n]
    qdot = z[n : 2 * n]
    b = z[2 * n : 2 * n + b0.shape[0]]
    rho = z[2 * n + b0.shape[0]].item()
    q_des = z[2 * n + b0.shape[0] + 1 : 2 * n + b0.shape[0] + 1 + n]
    jac_eef, htm_eef = robot.jac_geo(q=q)
    p_eef = htm_eef[0:3, 3]
    target = vf(p_eef, t)
    jac_target = jac_eef[0:3, :]
    # qdot_des = Utils.dp_inv(jac_target) @ target
    qdot_des, qddot_des = test_qdotqddot(t)
    # q_des = q_des_old + dt * qdot_des
    a_des = vf.acceleration(p_eef, jac_target @ qdot, t)  # change qdot to qdot_des
    Jdot = dot_J(robot, qdot, q)[:3, :]
    # qddot_des = Utils.dp_inv(jac_target) @ (a_des - Jdot @ qdot_des)

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
    w_bar = B.T @ P @ B @ w
    w_bar_norm = np.linalg.norm(w_bar)
    K = np.block([Kp, Kd])
    kappa = np.block([[1], [np.linalg.norm(x)], [x.T @ x]])
    # b = np.array([[1750], [8.071], [0]]) #TODO remove this
    gamma = (kappa.T @ b).item()
    psbf_active = False

    if w_norm > (epsilon / 2):
        v = -w_bar - gamma * w_bar / w_bar_norm - rho * w_bar / (w_bar_norm**2) 
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
    bdot = L @ (kappa * w_bar_norm - xi @ b)
    rhodot = l - rho
    # jac_eef_next, htm_eef_next = robot.jac_geo(q=q_des)
    # p_eef_next = htm_eef_next[0:3, 3]
    # qdot_des_next = np.linalg.pinv(jac_eef_next[0:3, :]) @ vf(p_eef_next, t + dt)
    zdot = np.block([[xdot], [bdot], [rhodot], [qdot_des]])

    qdot = xdot[:n]
    qddot = xdot[n : 2 * n]

    if save_hist:
        hist_peef = np.block([hist_peef, p_eef])
        # hist_vf = np.block([hist_vf, target[0:3]])
        hist_qdot_des = np.block([hist_qdot_des, qdot_des])
        hist_qddot = np.block([hist_qddot, qddot])
        hist_qddot_des = np.block([hist_qddot_des, qddot_des])
        hist_x = np.block([hist_x, x])
        hist_torque = np.block([hist_torque, torque])
        hist_cond_J.append(np.linalg.cond(jac_target))
        hist_v = np.block([hist_v, v])
        hist_eta = np.block([hist_eta, eta])
        hist_psbf.append(psbf_active)
        hist_q_des = np.block([hist_q_des, q_des])

    return zdot

def run():
    global hist_time, hist_qdot, hist_qdot_des, hist_qddot, hist_qddot_des, hist_q, hist_peef, hist_vf, hist_cond_J, hist_cond_Jdot, hist_x, hist_torque, hist_v, hist_eta, hist_b, hist_rho, hist_psbf
    global q, qdot, b, rho, q_des, l, xi, epsilon, alpha, L, disturbance
    z = np.block([[q], [qdot], [b], [rho], [q_des]])

    for i in range(1, imax):
        progress_bar(i, imax)
        t = i * dt
        z = rk4(
            closed_loop,
            t,
            z,
            dt,
            n=n,
            l=l,
            xi=xi,
            epsilon=epsilon,
            alpha=0.2*0,
            L=L,
            disturbance=disturbance,
        )

        q = z[:n]
        qdot = z[n : 2 * n]
        b = z[2 * n : 2 * n + b0.shape[0]]
        rho = z[2 * n + b0.shape[0]].item()
        q_des = z[2 * n + b0.shape[0] + 1 : 2 * n + b0.shape[0] + 1 + n]

        robot.add_ani_frame(time=t, q=q)
        _, htms = robot.jac_geo(q=q, axis='dh')
        f1.add_ani_frame(time=t, htm=htms[0])
        f2.add_ani_frame(time=t, htm=htms[1])
        # nearest_point.add_ani_frame(time=t, htm=Utils.trn(vf.nearest_points[-1]))

        hist_time.append(t)
        hist_q = np.block([hist_q, q])
        # hist_peef = np.block([hist_peef, p_eef])
        hist_qdot = np.block([hist_qdot, qdot])
        # hist_qdot_des = np.block([hist_qdot_des, qdot_des])
        # hist_torque = np.block([hist_torque, torque])
        # hist_x = np.block([hist_x, np.block([[q], [qdot]])])
        hist_b = np.block([hist_b, b])
        hist_rho.append(rho)
        # hist_v = np.block([hist_v, v])
        # hist_eta = np.block([hist_eta, eta])

run()

fig = px.line(np.linalg.norm(hist_x, axis=0).T, title="|x|")
fig.add_scatter(x=list(range(imax)), y=np.array(hist_psbf)*3, fill='tozeroy')
# fig.write_image("figures/histx.pdf")
fig.show()
fig = px.line(
    np.linalg.norm(hist_qdot - hist_qdot_des, axis=0).T,
    title="|dq/dt - dq<sub>des</sub>/dt|",
)
fig.show()
# fig.write_image("figures/qdoterrNorm.pdf")
fig = px.line(
    np.linalg.norm(hist_q - hist_q_des, axis=0).T,
    title="|q - q<sub>des</sub>|",
)
# fig.write_image("figures/qdoterr.pdf")
fig.show()

hist_w_norm = np.linalg.norm(np.array([B.T @ P @ x.T for x in hist_x.T]).reshape(-1, n), axis=1)
fig = px.line(hist_w_norm, title='||w||')
fig.add_scatter(x=list(range(imax)), y=np.array(hist_psbf)*np.max(hist_w_norm), fill='tozeroy')
fig.show()
# %%
