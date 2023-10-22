# %%
import sys
sys.path.insert(1, '/home/fbartelt/Documents/UFMG/TCC/Sim/uaibot/uaibot')
import time
import numpy as np
import robot as rb
import plotly.graph_objects as go
import plotly.express as px
from simulation import Simulation
from utils import Utils
from simobjects.box import Box
from simobjects.ball import Ball
from simobjects.frame import Frame
from simobjects.pointcloud import PointCloud
from cvxopt import matrix, solvers
from graphics.meshmaterial import MeshMaterial, Texture
from graphics.model3d import Model3D
from robot.links import Link
from plotly.subplots import make_subplots
from demo import Demo
from IPython.display import clear_output


def _create_2dof(color='#3e3f42', opacity=1):

    DH_theta = np.array([0, 0,])
    DH_d = np.array([0, 0,])
    DH_a = np.array([1, 0,])
    DH_alpha = np.array([0.0, -np.pi/2*0, ])

    DH_type = np.array([0, 0])
    link_info = np.array([DH_theta, DH_d, DH_alpha, DH_a, DH_type])

    scale = 1
    n = link_info.shape[1]
    base_3d_obj = []
    mesh = MeshMaterial(metalness=0.5, clearcoat=0, roughness=0.5,
                        normal_scale=[0.5, 0.5], color=color,
                        opacity=opacity, side="DoubleSide")

    htm_base_0 = Utils.trn([0, 0, 0.15])

    Q00 = Utils.rotz(np.pi/2) * Utils.rotx(np.pi/2) * Utils.trn(
        [0, 0.15, -0.15]) * Utils.rotz(np.pi/2) * Utils.trn([1, 0, 0.15])
    Q01 = Q00
    Q02 = Q01

    link_3d_obj = []
    link1_mth = Utils.inv_htm(Q01)

    link_3d_obj.append([
        Model3D(url='https://raw.githubusercontent.com/fbartelt/robotics-experiments/main/models/l1.obj',
                scale=scale, htm=link1_mth, mesh_material=mesh,),
    ])

    link2_mth = Utils.inv_htm(Q02)
    link_3d_obj.append([
        Model3D(url='https://raw.githubusercontent.com/fbartelt/robotics-experiments/main/models/l2.obj',
                scale=scale, htm=link2_mth, mesh_material=mesh,),
    ])

    #com_coordinates = [[0.534615, 0, 0.15], [1.5353, 0, 0.15]]
    com_coordinates = [[0.534615 - 1, 0, 0], [1.5353 - 1, 0, 0]]
    list_inertia_mat = []

    # Icm + parallel axis theorem (Inertia mat is in respect to DH frame)
    list_inertia_mat.append(np.diag([0.010533, 0.010573, 0.001061]) + 
                            (np.array([-(0.534615 - 1), 0, 0]).reshape(1, -1) @ np.array([-(0.534615 - 1), 0, 0]).reshape(-1, 1)) * np.eye(3) - (np.array([-(0.534615 - 1), 0, 0]).reshape(-1, 1) @ np.array([-(0.534615 - 1), 0, 0]).reshape(1, -1)))
    list_inertia_mat.append(np.diag([0.010573, 0.010613, 0.001061]) + 
                            (np.array([-(1.5353 - 1), 0, 0]).reshape(1, -1) @ np.array([-(1.5353 - 1), 0, 0]).reshape(-1, 1)) * np.eye(3) - (np.array([-(1.5353 - 1), 0, 0]).reshape(-1, 1) @ np.array([-(1.5353 - 1), 0, 0]).reshape(1, -1)))
    list_mass = [159.779, 159.949]
    links = []
    for i in range(n):
        links.append(Link(i, theta=link_info[0, i], d=link_info[1, i], alpha=link_info[2, i], a=link_info[3, i], joint_type=link_info[4, i],
                          list_model_3d=link_3d_obj[i], com_coordinates=com_coordinates[i], mass=list_mass[i], inertia_matrix=list_inertia_mat[i]))

    q0 = [0, np.pi/2*0]
    htm_n_eef = Utils.trn([1.05, 0, 0]) @ Utils.roty(np.pi/2)

    # Create joint limits
    joint_limits = np.matrix([[-3*np.pi, 3*np.pi], [-1.8, 1.8]])

    return links, base_3d_obj, htm_base_0, htm_n_eef, q0, joint_limits


def create_2dof(name='jaco_robot', opacity=1):
    links, base_3d_obj, htm_base_0, htm_n_eef, q0, joint_limits = _create_2dof(opacity=opacity)
    jaco = rb.Robot(name=name, links=links, list_base_3d_obj=base_3d_obj, htm=np.identity(4),
                    htm_base_0=htm_base_0, htm_n_eef=htm_n_eef, q0=q0, eef_frame_visible=True, joint_limits=joint_limits)
    return jaco

def error(robot, htm_des, orientation_error=True, q=None, htm=None):
    jac_eef, htm_eef = robot.jac_geo(q, "eef", htm)
    p_des = htm_des[0:3, 3]
    x_des = htm_des[0:3, 0]
    y_des = htm_des[0:3, 1]
    z_des = htm_des[0:3, 2]

    p_eef = htm_eef[0:3, 3]
    x_eef = htm_eef[0:3, 0]
    y_eef = htm_eef[0:3, 1]
    z_eef = htm_eef[0:3, 2]

    err = np.matrix(np.zeros((6,1)))
    err[0:3,0] = p_eef - p_des
    eo = -0.5 * (Utils.S(x_eef)*x_des + Utils.S(y_eef)*y_des + Utils.S(z_eef)*z_des) # Siciliano cap 3.7.3 p.139
    # err[3:, 0] = eo
    err[3] = max(1 - x_des.T * x_eef, 0)
    err[4] = max(1 - y_des.T * y_eef, 0)
    err[5] = max(1 - z_des.T * z_eef, 0)

    n = len(robot.links)
    jac_err = np.matrix(np.zeros((6, n)))
    jac_err[0:3, :] = jac_eef[0:3, :]
    L = 0.5 * (Utils.S(x_des)*Utils.S(x_eef) + Utils.S(y_des)*Utils.S(y_eef) + Utils.S(z_des)*Utils.S(z_eef))
    # jac_err[3:, :] = L @ jac_eef[3:, :]
    jac_err[3, :] = x_des.T * Utils.S(x_eef) * jac_eef[3:6, :]
    jac_err[4, :] = y_des.T * Utils.S(y_eef) * jac_eef[3:6, :]
    jac_err[5, :] = z_des.T * Utils.S(z_eef) * jac_eef[3:6, :]
    return err, jac_err

def pseudo_inv_control(robot, htm_des, K=1, T=10, dt=0.05, frames=[]):
    for i in range(int(T//dt)):
        e, Je = error(robot, htm_des)
        qdot = - K @ np.linalg.pinv(Je) @ e
        q = robot.q + qdot * dt
        robot.add_ani_frame(time=i*dt, q=q)
        htms = robot.fkm(q=q, axis='dh')
        for j, frame in enumerate(frames):
            frame.add_ani_frame(time=i*dt, htm=htms[j])

def QP_control(robot, htm_des, limit_joints=True, K=1, xi=1, T=10, dt=0.05, eps = 0.001, frames=[], axis='dh'):
    qhist, qdothist = [], []
    solvers.options['show_progress'] = False
    n = len(robot.links)
    q = robot.q
    qhist.append(q)
    qdothist.append(q)
    for i in range(int(T//dt)):
        if limit_joints:
            qdot_max = robot.joint_limit[:, 1] - q.reshape(-1, 1)
            qdot_min = robot.joint_limit[:, 0] - q.reshape(-1, 1)
            A = np.block([[np.identity(n)], [-np.identity(n)]])
            b = np.block([[xi * qdot_max], [-xi * qdot_min]])
        else:
            A = np.identity(n)*0
            b = np.block([[xi * qdot_max]]) * 1e35
        e, Je = error(robot, htm_des)
        H = 2*(Je.T * Je + eps * np.identity(n))
        f = K * 2 *(e.T @ Je).T
        try:
            qdot = solvers.qp(matrix(H), matrix(f), matrix(A), matrix(b))['x']
        except:
            qdot = np.matrix(np.zeros((n, 1)))
            error_qp = True
            print(error_qp)
        
        q = robot.q + qdot * dt
        robot.add_ani_frame(time=i*dt, q=q)
        htms = robot.fkm(q=q, axis=axis)
        for j, frame in enumerate(frames):
            frame.add_ani_frame(time=i*dt, htm=htms[j])
        qhist.append(q)
        qdothist.append(qdot)
    return qhist, qdothist

htm_des = Utils.trn([-2.05, 0, 0.15]) @ Utils.roty(np.pi/2) @ Utils.rotx(np.pi)
htm_des = np.matrix([[-0.   , -0.881, -0.472, -1.292],
                     [ 0.   , -0.472,  0.881,  0.32 ],
                     [-1.   ,  0.   ,  0.   ,  0.15 ],
                     [ 0.   ,  0.   ,  0.   ,  1.   ]])
K = np.matrix([[2, 0], [0, 1]])
dt = 0.05
T = 20
axis = 'com'

double_pendulum = create_2dof(opacity=0.7)
n = len(double_pendulum.links)
frames_ref = double_pendulum.fkm(axis=axis)
sim = Simulation.create_sim_grid([double_pendulum])
frame_des = Frame(name='frame_des', htm=htm_des)
sim.add(Frame(name='base', htm=double_pendulum.htm_base_0))
sim.add(frame_des)
frames = []
for i, htm in enumerate(frames_ref):
    frames.append(Frame(name=f'{axis}_{i}', htm=htm))

for frame in frames:
    sim.add(frame)

#pseudo_inv_control(double_pendulum, htm_des, T=T, dt=dt, K=K, frames=frames)
qhist, qdot_hist = QP_control(double_pendulum, htm_des, T=T, dt=dt, K=1, frames=frames, axis=axis)
sim.run()
# %%
fig = px.line(np.array(qhist).reshape(-1, 2))
m = len(qhist)
fig.add_scatter(x=list(range(m)), y=[-1.8]*m)
fig.show()

# %%
fig = px.line(np.array(qdot_hist).reshape(-1, 2))
fig.show()

# %%
