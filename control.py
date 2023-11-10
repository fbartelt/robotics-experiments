#%%
import sys
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
    
    def pseudo_inverse_control(self):
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
            raise UserWarning("Quadratic Programming did not converge")
        
        return qdot, e, error_qp
    
    def computed_torque(self, qdot, qdot_des, sigma=None, q_des=None, Kp=1, Kd=1, Ki=1, K=1):
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
        if sigma is None:
            sigma=np.zeros((n, 1))
        
        q = self.robot.q
        if q_des is None:
            q_des = self.robot.ikm(self.htm_des)
        
        M, C, G = self.robot.dyn_model(q, qdot)
        torque = C + G + (M @ (Kp @ (q_des - q) + Kd @ (qdot_des - qdot) + Ki @ sigma) )
        qddot = np.linalg.inv(M) @ (-C -G + torque)
        return qddot
    
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


def computed_torque(robot, htm_des, Kp=1, Kd=1, Ki=1, K=1, T=10, dt=0.05, tol=1e-3, useQP=False, frames=[]):
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
    virtual_robot = deepcopy(robot)
    q_des = robot.ikm(htm_des)
    qhist, qdothist = [], []
    qhist.append(q)
    qdothist.append(q)

    while (i < imax) and (np.linalg.norm(e) > tol):
        e, Je = error(virtual_robot, htm_des)
        qdot_des = - K * np.linalg.pinv(Je) @ e
        M, C, G = robot.dyn_model(double_pendulum.q, qdot)
        torque = C + G + (M @ (Kp @ (q_des - q) + Kd @ (qdot_des - qdot) + Ki @ sigma) )
        sigma += dt*(q_des - q)
        q = robot.q + qdot * dt
        qdot += dt * (np.linalg.inv(M) @ (-C -G + torque))
        robot.add_ani_frame(time=i*dt, q=q)
        virtual_robot.add_ani_frame(time=i*dt, q=q)
        htms = robot.fkm(q=q, axis='dh')
        for j, frame in enumerate(frames):
            frame.add_ani_frame(time=i*dt, htm=htms[j])
        qhist.append(q)
        qdothist.append(qdot)
        i += 1
    return qhist, qdothist


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

def tester(robot, control_type='pinv'):
    htm_des = np.matrix([[-0., -0.881, -0.472, -1.292],
                     [0., -0.472,  0.881,  0.32],
                     [-1.,  0.,  0.,  0.15],
                     [0.,  0.,  0.,  1.]])
    controller = Controller(robot, htm_des, control_type)
    imax = int(T//dt)
    tol = 1e-3
    e = np.inf
    i = 0
    while (i < imax) and (np.linalg.norm(e) > tol):
        qdot, e, *_ = controller.control()
        q = robot.q + qdot * dt
        robot.add_ani_frame(time=i*dt, q=q)
        htms = robot.fkm(q=q, axis='dh')
        for j, frame in enumerate(frames):
            frame.add_ani_frame(time=i*dt, htm=htms[j])
        i+=1


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
qhist, qdot_hist = computed_torque(double_pendulum, htm_des, K=1, Kp=3, Kd=2, Ki=2, T=T, dt=dt, frames=frames)
tester(double_pendulum, 'qp')
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