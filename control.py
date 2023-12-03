#%%
import sys
import warnings
sys.path.insert(1, '/home/fbartelt/Documents/UFMG/TCC/Sim/uaibot/uaibot')
import numpy as np
import robot as rb
import plotly.graph_objects as go
import plotly.express as px
from copy import deepcopy
from simulation import Simulation
from utils import Utils
from simobjects.box import Box
from simobjects.ball import Ball
from simobjects.frame import Frame
from simobjects.pointcloud import PointCloud
from cvxopt import matrix, solvers
from plotly.subplots import make_subplots
from IPython.display import clear_output
from planar_manipulator import create_2dof
from create_jaco import create_jaco2
from simobjects.pointlight import PointLight
from scipy.linalg import block_diag, solve_continuous_are
solvers.options['show_progress'] = False
# 6 DoF Curved Wrist
class Controller():
    def __init__(self, robot, htm_des, control_type='pinv'):
        self.robot = robot
        self.htm_des = htm_des
        self.control_type = control_type
        self.control = self._select_controller(control_type)

    def _select_controller(self, control_type):
        match control_type.lower():
            case "pinv":
                return self.pseudo_inverse_control
            case "qp":
                return self.qp_control
            case "torque":
                return self.computed_torque
    
    def error_func(self, htm_des=None):
        if htm_des is None:
            htm_des = self.htm_des
        jac_eef, htm_eef = self.robot.jac_geo(q=None, axis="eef", htm=None)
        p_des = htm_des[0:3, 3]
        x_des = htm_des[0:3, 0]
        y_des = htm_des[0:3, 1]
        z_des = htm_des[0:3, 2]

        p_eef = htm_eef[0:3, 3]
        x_eef = htm_eef[0:3, 0]
        y_eef = htm_eef[0:3, 1]
        z_eef = htm_eef[0:3, 2]

        err = np.matrix(np.zeros((6, 1)))
        err[0:3, 0] = p_eef - p_des
        eo = -0.5 * (Utils.S(x_eef)*x_des + Utils.S(y_eef)*y_des +
                    Utils.S(z_eef)*z_des)  # Siciliano cap 3.7.3 p.139
        # err[3:, 0] = eo
        err[3] = max(1 - x_des.T * x_eef, 0)
        err[4] = max(1 - y_des.T * y_eef, 0)
        err[5] = max(1 - z_des.T * z_eef, 0)

        n = len(self.robot.links)
        jac_err = np.matrix(np.zeros((6, n)))
        jac_err[0:3, :] = jac_eef[0:3, :]
        L = 0.5 * (Utils.S(x_des)*Utils.S(x_eef) + Utils.S(y_des)
                * Utils.S(y_eef) + Utils.S(z_des)*Utils.S(z_eef))
        # jac_err[3:, :] = L @ jac_eef[3:, :]
        jac_err[3, :] = x_des.T * Utils.S(x_eef) * jac_eef[3:6, :]
        jac_err[4, :] = y_des.T * Utils.S(y_eef) * jac_eef[3:6, :]
        jac_err[5, :] = z_des.T * Utils.S(z_eef) * jac_eef[3:6, :]
        return err, jac_err
    
    def pseudo_inverse_control(self, K=1):
        n = len(self.robot.links)
        if not isinstance(K, np.ndarray):
            if isinstance(K, (list, tuple)):
                K = np.diag(K)
            else:
                K = np.diag([K]*n)
        e, Je = self.error_func(self.htm_des)
        qdot = - K @ np.linalg.pinv(Je) @ e
        return qdot, e
    
    def qp_control(self, ignore_joint_limits=False, xi=1, K=1, eps=0.001):
        n = len(self.robot.links)
        error_qp = False

        if not ignore_joint_limits:
            qdot_max = self.robot.joint_limit[:, 1] - self.robot.q.reshape(-1, 1)
            qdot_min = self.robot.joint_limit[:, 0] - self.robot.q.reshape(-1, 1)
            A = np.block([[np.identity(n)], [-np.identity(n)]])
            b = np.block([[xi * qdot_max], [-xi * qdot_min]])
        else:
            A = np.identity(n)*0
            b = np.block([[xi * self.robot.joint_limit[:, 1]]]) * 1e35

        e, Je = self.error_func(self.htm_des)
        H = 2*(Je.T * Je + eps * np.identity(n))
        f = K * 2 * (e.T @ Je).T
        try:
            qdot = solvers.qp(matrix(H), matrix(f), matrix(A), matrix(b))['x']
        except:
            qdot = np.matrix(np.zeros((n, 1)))
            error_qp = True
            warnings.warn("Quadratic Programming did not converge")
        
        qdot = np.array(qdot).reshape(-1, 1)
        return qdot, e, error_qp
    
    def computed_torque(self, qdot, qdot_des, sigma=None, q_des=None, Kp=1, Kd=1, Ki=1, disturbance=0):
        n = len(self.robot.links)

        if not isinstance(Kp, np.ndarray):
            if isinstance(Kp, (list, tuple)):
                Kp = np.diag(Kp)
            else:
                Kp = np.diag([Kp]*n)
        if not isinstance(Kd, np.ndarray):
            if isinstance(Kd, (list, tuple)):
                Kd = np.diag(Kd)
            else:
                Kd = np.diag([Kd]*n)
        if not isinstance(Ki, np.ndarray):
            if isinstance(Ki, (list, tuple)):
                Ki = np.diag(Ki)
            else:
                Ki = np.diag([Ki]*n)

        if isinstance(disturbance, (float, int)):
            disturbance = np.array([disturbance] * n).reshape(-1, 1)

        if sigma is None:
            if np.any(Ki):
                warnings.warn("Missing parameter 'sigma', assuming sigma=0")
            sigma=np.zeros((n, 1))
        
        if q_des is None:
            q_des = self.robot.ikm(self.htm_des)
        
        q = self.robot.q
        M, C, G = self.robot.dyn_model(q, qdot)
        torque = C + G + (M @ (Kp @ (q_des - q) + Kd @ (qdot_des - qdot) + Ki @ sigma) )
        qddot = np.linalg.inv(M) @ (-C -G + torque)  + disturbance
        return qddot, torque
    
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

    err = np.matrix(np.zeros((6, 1)))
    err[0:3, 0] = p_eef - p_des
    eo = -0.5 * (Utils.S(x_eef)*x_des + Utils.S(y_eef)*y_des +
                 Utils.S(z_eef)*z_des)  # Siciliano cap 3.7.3 p.139
    # err[3:, 0] = eo
    err[3] = max(1 - x_des.T * x_eef, 0)
    err[4] = max(1 - y_des.T * y_eef, 0)
    err[5] = max(1 - z_des.T * z_eef, 0)

    n = len(robot.links)
    jac_err = np.matrix(np.zeros((6, n)))
    jac_err[0:3, :] = jac_eef[0:3, :]
    L = 0.5 * (Utils.S(x_des)*Utils.S(x_eef) + Utils.S(y_des)
               * Utils.S(y_eef) + Utils.S(z_des)*Utils.S(z_eef))
    # jac_err[3:, :] = L @ jac_eef[3:, :]
    jac_err[3, :] = x_des.T * Utils.S(x_eef) * jac_eef[3:6, :]
    jac_err[4, :] = y_des.T * Utils.S(y_eef) * jac_eef[3:6, :]
    jac_err[5, :] = z_des.T * Utils.S(z_eef) * jac_eef[3:6, :]
    return err, jac_err

def pseudo_inv_control(robot, htm_des, K=1, T=10, dt=0.05, frames=[]):
    q = robot.q.copy()
    qhist, qdothist = [], []
    qhist.append(q)
    qdothist.append(q)
    for i in range(int(T//dt)):
        e, Je = error(robot, htm_des)
        qdot = - K @ np.linalg.pinv(Je) @ e
        q = robot.q + qdot * dt
        robot.add_ani_frame(time=i*dt, q=q)
        htms = robot.fkm(q=q, axis='dh')
        for j, frame in enumerate(frames):
            frame.add_ani_frame(time=i*dt, htm=htms[j])

    return qhist, qdothist


def computed_torque(robot, htm_des, q_des=None, Kp=1, Kd=1, Ki=1, K=1, T=10, dt=0.05, tol=1e-3, useQP=False, frames=[], axis='com'):
    n = len(robot.links)
    if not isinstance(Kp, np.ndarray):
        if isinstance(Kp, (list, tuple)):
            Kp = np.diag(Kp)
        else:
            Kp = np.diag([Kp]*n)
    if not isinstance(Kd, np.ndarray):
        if isinstance(Kd, (list, tuple)):
            Kd = np.diag(Kd)
        else:
            Kd = np.diag([Kd]*n)
    if not isinstance(Ki, np.ndarray):
        if isinstance(Ki, (list, tuple)):
            Ki = np.diag(Ki)
        else:
            Ki = np.diag([Ki]*n)
    imax = int(T//dt)
    i = 0
    e = np.inf
    q = robot.q.copy()
    sigma = np.zeros((n, 1))
    qdot = np.zeros((n, 1))
    # virtual_robot = deepcopy(robot)
    if q_des is None:
        q_des = robot.ikm(htm_des)
    # print(q_des)
    qhist, qdothist, e_hist, torque_hist = [], [], [], []
    qhist.append(q)
    qdothist.append(q)

    while (i < imax) and (np.linalg.norm(e) > tol):
        e, Je = error(robot, htm_des)
        qdot_des = - K * np.linalg.pinv(Je) @ e
        M, C, G = robot.dyn_model(robot.q, qdot)
        torque = C + G + (M @ (Kp @ (q_des - q) + Kd @ (qdot_des - qdot) + Ki @ sigma) )
        sigma = sigma + dt*(q_des - q)
        q = robot.q + qdot * dt
        qdot = qdot + dt * (np.linalg.inv(M) @ (-C -G + torque))
        robot.add_ani_frame(time=i*dt, q=q)
        # virtual_robot.add_ani_frame(time=i*dt, q=q)
        htms = robot.fkm(q=q, axis=axis)
        for j, frame in enumerate(frames):
            frame.add_ani_frame(time=i*dt, htm=htms[j])
        qhist.append(q)
        qdothist.append(qdot)
        e_hist.append(e)
        torque_hist.append(torque)
        i += 1
    return qhist, qdothist, e_hist, torque_hist


def QP_control(robot, htm_des, limit_joints=True, K=1, xi=1, T=10, dt=0.05, tol=1e-3, eps=0.001, frames=[], axis='dh'):
    qhist, qdothist = [], []
    solvers.options['show_progress'] = False
    n = len(robot.links)
    imax = int(T//dt)
    i = 0
    error_qp = False
    e = np.inf
    q = robot.q
    qhist.append(q)
    qdothist.append(q)
    while (i < imax) and (np.linalg.norm(e) > tol) and (not error_qp):
        if limit_joints:
            qdot_max = robot.joint_limit[:, 1] - q.reshape(-1, 1)
            qdot_min = robot.joint_limit[:, 0] - q.reshape(-1, 1)
            A = np.block([[np.identity(n)], [-np.identity(n)]])
            b = np.block([[xi * qdot_max], [-xi * qdot_min]])
        else:
            A = np.identity(n)*0
            b = np.block([[xi * robot.joint_limit[:, 1]]]) * 1e35
        e, Je = error(robot, htm_des)
        H = 2*(Je.T * Je + eps * np.identity(n))
        f = K * 2 * (e.T @ Je).T
        try:
            qdot = solvers.qp(matrix(H), matrix(f), matrix(A), matrix(b))['x']
        except:
            qdot = np.matrix(np.zeros((n, 1)))
            error_qp = True

        q = robot.q + qdot * dt
        robot.add_ani_frame(time=i*dt, q=q)
        htms = robot.fkm(q=q, axis=axis)
        for j, frame in enumerate(frames):
            frame.add_ani_frame(time=i*dt, htm=htms[j])
        qhist.append(q)
        qdothist.append(qdot)
        i += 1
    return qhist, qdothist

def tester(robot, htm_des=None, control_type='pinv', T=20, dt=0.05, tol = 1e-3, frames=[]):
    if htm_des is None:
        htm_des = np.matrix([[-0., -0.881, -0.472, -1.292],
                             [0., -0.472,  0.881,  0.32],
                             [-1.,  0.,  0.,  0.15],
                             [0.,  0.,  0.,  1.]])
    controller = Controller(robot, htm_des, control_type)
    imax = int(T//dt)
    e = np.inf
    q_hist, qdot_hist, e_hist = [], [], []
    i = 1
    while (i < imax) and (np.linalg.norm(e) > tol):
        qdot, e, *_ = controller.control()
        q = robot.q + qdot * dt
        robot.add_ani_frame(time=i*dt, q=q)
        htms = robot.fkm(q=q, axis='dh')
        for j, frame in enumerate(frames):
            frame.add_ani_frame(time=i*dt, htm=htms[j])
        i+=1
        q_hist.append(q)
        qdot_hist.append(qdot)
        e_hist.append(e)
    return q_hist, qdot_hist, e_hist
#%%

htm_des = Utils.trn([-2.05, 0, 0.15]) @ Utils.roty(np.pi/2) @ Utils.rotx(np.pi)
htm_des = np.matrix([[-0., -0.881, -0.472, -1.292],
                     [0., -0.472,  0.881,  0.32],
                     [-1.,  0.,  0.,  0.15],
                     [0.,  0.,  0.,  1.]])
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
#%%
# qhist, qdot_hist = pseudo_inv_control(double_pendulum, htm_des, T=T, dt=dt, K=K, frames=frames)
# qhist, qdot_hist = QP_control(double_pendulum, htm_des, T=T, dt=dt, K=1, frames=frames, axis=axis)
Ts = 5
omega0 = 5.8339/Ts
qhist, qdot_hist, e_hist = computed_torque(double_pendulum, htm_des, K=1, Kp=3, Kd=2, Ki=2, T=T, dt=dt, frames=frames)
# tester(double_pendulum, 'qp')
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
htm_des = Utils.trn([0.2, 0.4, 0.7]) @ Utils.rotx(np.deg2rad(-100)) @ Utils.rotz(np.deg2rad(33))
frame_des = Frame(name='frame_des', htm=htm_des)
robot = create_jaco2()
light1 = PointLight(name="light1", color="white", intensity=2.5, htm=Utils.trn([-1,-1, 1.5]))
light2 = PointLight(name="light2", color="white", intensity=2.5, htm=Utils.trn([-1, 1, 1.5]))
light3 = PointLight(name="light3", color="white", intensity=2.5, htm=Utils.trn([ 1,-1, 1.5]))
light4 = PointLight(name="light4", color="white", intensity=2.5, htm=Utils.trn([ 1, 1, 1.5]))
sim = Simulation.create_sim_grid([robot, light1, light2, light3, light4])
sim.set_parameters(width=1200, height=600, ambient_light_intensity=4)
axis='dh'
frames_ref = robot.fkm(axis=axis)
frames = []
for i, htm in enumerate(frames_ref):
    frames.append(Frame(name=f'{axis}_{i}', htm=htm, size=0.1))
for frame in frames:
    sim.add(frame)
sim.add(frame_des)
q_hist, qdot_hist, e_hist = tester(robot, htm_des=htm_des, control_type='qp', frames=frames)
sim.run()
# %%
""" PID computed torque"""
htm_des = Utils.trn([0.2, 0.4, 0.7]) @ Utils.rotx(np.deg2rad(-100)) @ Utils.rotz(np.deg2rad(33))
frame_des = Frame(name='frame_des', htm=htm_des)
robot = create_jaco2()
light1 = PointLight(name="light1", color="white", intensity=2.5, htm=Utils.trn([-1,-1, 1.5]))
light2 = PointLight(name="light2", color="white", intensity=2.5, htm=Utils.trn([-1, 1, 1.5]))
light3 = PointLight(name="light3", color="white", intensity=2.5, htm=Utils.trn([ 1,-1, 1.5]))
light4 = PointLight(name="light4", color="white", intensity=2.5, htm=Utils.trn([ 1, 1, 1.5]))
sim = Simulation.create_sim_grid([robot, light1, light2, light3, light4])
sim.set_parameters(width=800, height=600, ambient_light_intensity=4)
axis='com'
frames_ref = robot.fkm(axis=axis)
frames = []
for i, htm in enumerate(frames_ref):
    frames.append(Ball(name=f'{axis}_{i}', htm=htm, radius=0.01, color='magenta'))
for frame in frames:
    sim.add(frame)
sim.add(frame_des)
# Kp=4, Kd=5, Ki=1, K=1
# Kp=8, Kd=5, Ki=8
# Ziegler-Nichols: Pu = 124s, Ku=1 => Kp=0.6Ku, Kd=KpPu/8, Ki=2Kp/Ku
# Pu, Ku = 6.2, 4
# Kp =0.6*Ku
# Ki, Kd = 2*Kp/Ku, Kp*Pu/8
Kp, Kd, Ki = 8, 5, 8
q_des = np.array([[ 0.7262458 ], [ 1.61760955], [ 0.11582987], [-1.14679451], [ 2.16399157], [ 2.76812822]])
q_hist, qdot_hist, e_hist = computed_torque(robot, htm_des=htm_des, q_des=q_des, Kp=Kp, Kd=Kd, Ki=Ki, K=1, T=20, frames=frames, axis=axis)
sim.run()
fig = px.line(np.array(e_hist).reshape(-1, 6))
fig.show()
fig = px.line(np.array(qdot_hist).reshape(-1, 6))
fig.show()
# %%
""" LQR computed torque"""
htm_des = Utils.trn([0.2, 0.4, 0.7]) @ Utils.rotx(np.deg2rad(-100)) @ Utils.rotz(np.deg2rad(33))
frame_des = Frame(name='frame_des', htm=htm_des)
robot = create_jaco2(thesis_parameters=True)
light1 = PointLight(name="light1", color="white", intensity=2.5, htm=Utils.trn([-1,-1, 1.5]))
light2 = PointLight(name="light2", color="white", intensity=2.5, htm=Utils.trn([-1, 1, 1.5]))
light3 = PointLight(name="light3", color="white", intensity=2.5, htm=Utils.trn([ 1,-1, 1.5]))
light4 = PointLight(name="light4", color="white", intensity=2.5, htm=Utils.trn([ 1, 1, 1.5]))
sim = Simulation.create_sim_grid([robot, light1, light2, light3, light4])
sim.set_parameters(width=800, height=600, ambient_light_intensity=4)
axis='com'
frames_ref = robot.fkm(axis=axis)
frames = []
for i, htm in enumerate(frames_ref):
    frames.append(Ball(name=f'{axis}_{i}', htm=htm, radius=0.01, color='magenta'))
for frame in frames:
    sim.add(frame)
sim.add(frame_des)

n = len(robot.links)
A = np.block([[np.block([np.zeros((2*n,n)), block_diag(np.eye(n), np.eye(n))])], [np.zeros((n, 3*n))]])
B = np.block([[np.zeros((2*n, n))], [np.eye(n)]])
qmax = np.diag(np.array(1/np.max(np.abs(robot.joint_limit), axis=1)).flatten())
qdotmax = np.diag(1/(np.array([6, 6, 6, 8, 8, 8])*2*np.pi/60)) #Max RPM of actuators
# Q = np.eye(3*n)
Q = 2*np.diag([15]*n + [30]*n + [25]*n)
Q = 20*np.diag([600]*n + [30]*n + [30]*n) # Q = 4*np.diag([40]*n + [30]*n + [30]*n)
R = np.eye(n)
# R = np.diag([1/3]*3 + [1]*3)
# Bryson rule
# Q = block_diag(qmax, qmax, qdotmax)
# R = np.diag([1/30.5, 1/30.5, 1/30.5, 1/6.8, 1/6.8, 1/6.8]) # 1/Maximum torque for motors KA 75, 75, 75, 58, 58, 58, [32, 32, 32]

P = solve_continuous_are(A,B,Q,R)
K = np.linalg.inv(R) @ B.T @ P
Ki, Kp, Kd = np.split(K, 3, axis=1)

q_des = np.array([[ 0.7262458 ], [ 1.61760955], [ 0.11582987], [-1.14679451], [ 2.16399157], [ 2.76812822]])
dyn_controller = Controller(robot, htm_des, 'torque')
kin_controller = Controller(robot, htm_des, 'qp')
e = [np.inf]*3
q = robot.q.copy()
sigma = np.zeros((n, 1))
qdot = np.zeros((n, 1))
q_hist, qdot_hist, e_hist, qdotdes_hist, sigma_hist, torque_hist = [], [], [], [], [], []
T, dt, tol = 20, 0.05, 1e-3
i = 1
imax = int(T//dt)
disturbance=0

while (i < imax) and (np.linalg.norm(e) > tol):
    qdot_des, e, *_ = kin_controller.control(K=5)
    q_des = q + dt*qdot_des
    if i >= 200:
        disturbance = 40
    qddot, torque  = dyn_controller.control(qdot, qdot_des, sigma=sigma, q_des=q_des, Ki=Ki, Kd=Kd, Kp=Kp, disturbance=disturbance)
    e, _ = dyn_controller.error_func(htm_des, )
    sigma = sigma + dt*(q_des - q)
    q = robot.q + qdot * dt
    qdot = qdot + dt * qddot
    robot.add_ani_frame(time=i*dt, q=q)
    htms = robot.fkm(q=q, axis='dh')
    for j, frame in enumerate(frames):
        frame.add_ani_frame(time=i*dt, htm=htms[j])
    i+=1
    q_hist.append(q)
    qdot_hist.append(qdot)
    qdotdes_hist.append(qdot_des)
    e_hist.append(e)
    sigma_hist.append(sigma)
    torque_hist.append(torque)


sim.run()
fig = px.line(np.array(e_hist).reshape(-1, 6))
fig.show()
fig = px.line(np.array(sigma_hist).reshape(-1, 6))
fig.show()
fig = px.line(np.array(torque_hist).reshape(-1, 6))
fig.show()
# fig = px.line(np.array(qdot_hist).reshape(-1, 6))
# fig.show()
# fig = px.line(np.array(qdotdes_hist).reshape(-1, 6))
# fig.show()
# %%