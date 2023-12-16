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
from scipy.linalg import block_diag, solve_continuous_are, solve_continuous_lyapunov
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
            case "lyapunov":
                return self.lyapunov_redesign
            case _:
                warnings.warn(f'{control_type} is not a valid controller.')
    
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
    
    def qp_control(self, ignore_joint_limits=False, xi=1, K=1, eps=0.001, htm_des=None):
        n = len(self.robot.links)
        error_qp = False
        small_limit = 0.62832e10 # 6rpm
        big_limit = 0.83776e10 # 8rpm
        qdot_limits = np.array([[big_limit],[big_limit],[big_limit],[small_limit],[small_limit],[small_limit]])

        if not ignore_joint_limits:
            qdot_max = np.minimum(self.robot.joint_limit[:, 1] - self.robot.q.reshape(-1, 1), qdot_limits)
            qdot_min = np.maximum(self.robot.joint_limit[:, 0] - self.robot.q.reshape(-1, 1), -qdot_limits)
            # qdot_max = self.robot.joint_limit[:, 1] - self.robot.q.reshape(-1, 1)
            # qdot_min = self.robot.joint_limit[:, 0] - self.robot.q.reshape(-1, 1)
            A = np.block([[np.identity(n)], [-np.identity(n)]])
            b = np.block([[xi * qdot_max], [-xi * qdot_min]])
        else:
            A = np.identity(n)
            b = np.block([[xi * self.robot.joint_limit[:, 1]]]) * 1e10

        if htm is None:
            htm_des = self.htm_des
        e, Je = self.error_func(htm_des)
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
    
    def computed_torque(self, qdot, qdot_des, sigma=None, q_des=None, qddot_des=None, Kp=1, Kd=1, Ki=1, disturbance=0):
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
        if qddot_des is None:
            qddot_des = np.zeros((n,1))
        
        q = self.robot.q
        M, C, G = self.robot.dyn_model(q, qdot)
        torque = C + G + (M @ (qddot_des + Kp @ (q_des - q) + Kd @ (qdot_des - qdot) + Ki @ sigma) )
        qddot = np.linalg.inv(M) @ (-C -G + torque + disturbance)
        return qddot, torque
    
    def lyapunov_redesign(self, qdot, qdot_des, real_robot, P, sigma=None, q_des=None, Kp=1, Kd=1, Ki=0, disturbance=0, epsilon=1e-3):
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
        if np.any(Ki):
            x = np.block([[qdot - qdot_des], [q - q_des], [sigma]])
            # x = np.block([[-(qdot - qdot_des)], [-(q - q_des)],])
            B = np.block([[np.eye(n)], [np.zeros((2*n, n))]])
        else:
            x = np.block([[qdot - qdot_des], [q - q_des],])
            # x = np.block([[-(qdot - qdot_des)], [-(q - q_des)],])
            B = np.block([[np.eye(n)], [np.zeros((n, n))]])
        Px = B.T @ P @ x
        Px_norm = np.linalg.norm(Px)
        # K = np.block([-Kd, -Kp])
        K = np.block([Kd, Kp])
        phi = 18.19441564410192 + 10 + 1.1210095929589532*np.linalg.norm(x) # Bound G, disturbance and ||C\dot{q}||
        phi = 5*np.linalg.norm(x)**2 + 1# Bound G, disturbance and ||C\dot{q}||
        phi = 1.12 * np.linalg.norm(x)
        alpha = 0.8 # (Mmax - Mmin)/(Mmax + Mmin) <= alpha
        rho = 1/(1 - alpha)  * (alpha*np.linalg.norm(K, 2) * np.linalg.norm(x) + 1/0.1640573139501184*phi)
        if Px_norm >= epsilon:
            delta_v = -rho * (Px/Px_norm)
            print(Px_norm)
            print(np.linalg.norm(x))
        else:
            delta_v = -rho * (Px/epsilon)
            print(Px_norm)
            print(np.linalg.norm(x))
        
        M_, C_, G_ = self.robot.dyn_model(q, qdot)
        M, C, G = real_robot.dyn_model(q, qdot)
        M = np.eye(6) * 2/((1/0.1640573139501184) + (1/1.4397118507671303))
        rng = np.random.default_rng()
        # M = M_ + rng.uniform(high=0.01, size=(6,6))
        
        v = -Kp @ (q - q_des) - Kd @ (qdot - qdot_des) - Ki @ sigma + delta_v
        # v = Kp @ (-(q - q_des)) + Kd @ (-(qdot - qdot_des)) + Ki @ sigma + delta_v
        torque = C_ + G_ + (M_ @ v )
        # eta = np.linalg.inv(M) @ ((M_ - M) @ v + (C_ - C) + (G_ - G) + disturbance)
        eta = np.linalg.inv(M_) @ ((M_ - M_) @ v + (C_ - C_) + (G_ - G_) + disturbance)
        qddot = v + eta
        return qddot, torque
    
    def lyapunov_redesign_test(self, qdot, qdot_des, qddot_des, real_robot, P, sigma=None, q_des=None, Kp=1, Kd=1, Ki=0, disturbance=0, epsilon=1e-3, test=False):
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
        if np.any(Ki):
            x = np.block([[qdot - qdot_des], [q - q_des], [sigma]])
            # x = np.block([[-(qdot - qdot_des)], [-(q - q_des)],])
            B = np.block([[np.eye(n)], [np.zeros((2*n, n))]])
        else:
            x = np.block([[qdot - qdot_des], [q - q_des],])
            # x = np.block([[-(qdot - qdot_des)], [-(q - q_des)],])
            B = np.block([[np.eye(n)], [np.zeros((n, n))]])
        Px = B.T @ P @ x
        Px_norm = np.linalg.norm(Px)
        # K = np.block([-Kd, -Kp])
        # K = np.block([Kd, Kp])
        K = np.block([Kd])
        phi = 18.19441564410192 + 10 + 1.1210095929589532*np.linalg.norm(x) # Bound G, disturbance and ||C\dot{q}||
        phi = 1.12 * np.linalg.norm(x)
        alpha = 0.8 # (Mmax - Mmin)/(Mmax + Mmin) <= alpha
        rho = 1/(1 - alpha)  * (alpha*np.linalg.norm(K, 2) * np.linalg.norm(x) + 1/0.1640573139501184*phi)
        if Px_norm >= epsilon:
            delta_v = -rho * (Px/Px_norm)
        else:
            delta_v = -rho * (Px/epsilon)
        
        M_, C_, G_ = self.robot.dyn_model(q, qdot)
        M, C, G = real_robot.dyn_model(q, qdot)
        M = np.eye(6) * 2/((1/0.1640573139501184) + (1/1.4397118507671303))
        rng = np.random.default_rng()
        M = M_ + rng.uniform(high=0.01, size=(6,6))
        
        # v = qddot_des -Kp @ (q - q_des) - Kd @ (qdot - qdot_des) - Ki @ sigma + delta_v
        v = qddot_des - Kd @ (qdot - qdot_des) - Ki @ sigma + delta_v
        torque = C_ + G_ + (M_ @ v )
        # eta = np.linalg.inv(M) @ ((M_ - M) @ v + (C_ - C) + (G_ - G) + disturbance)
        eta = np.linalg.inv(M_) @ ((M_ - M_) @ v + (C_ - C_) + (G_ - G_) + disturbance)
        qddot = v + eta
        return qddot, torque

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
#%%
""" LQR computed torque MODIFIED"""
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
A = np.block([[np.zeros((n, 3*n))], [np.block([np.eye(n), np.zeros((n, 2*n))])],[np.block([np.zeros((n, n)), np.eye(n), np.zeros((n, n))])]])
B = np.block([[np.eye(n)], [np.zeros((2*n, n))]])
Q = 24*np.diag([30]*n + [30]*n + [880]*n)
R = np.eye(n)
P = solve_continuous_are(A, B, Q, R)
K = np.linalg.inv(R) @ B.T @ P
Kd, Kp, Ki = np.split(K, 3, axis=1)

q_des = np.array([[ 0.7262458 ], [ 1.61760955], [ 0.11582987], [-1.14679451], [ 2.16399157], [ 2.76812822]])
dyn_controller = Controller(robot, htm_des, 'torque')
kin_controller = Controller(robot, htm_des, 'qp')
e = [np.inf]*3
q = robot.q.copy()
sigma = np.zeros((n, 1))
qdot = np.zeros((n, 1))
q_hist, qdot_hist, e_hist, qdotdes_hist, sigma_hist, torque_hist = [], [], [], [], [], []
T, dt, tol = 30, 0.05, 1e-2
i = 1
imax = int(T//dt)
disturbance=0

while (i < imax) and (np.linalg.norm(e) > tol):
    qdot_des, e, *_ = kin_controller.control(K=2, ignore_joint_limits=True)
    q_des = q + dt*qdot_des
    if i >= 200:
        disturbance = 1.5E-3
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

#%%
""" PD - LQR computed torque ---- Working"""
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
A = np.block([[np.zeros((n, 2*n))], [np.block([np.eye(n), np.zeros((n, n))])]])
B = np.block([[np.eye(n)], [np.zeros((n, n))]])
Q = 24*np.diag([30]*n + [30]*n)
# Q = 0.01*np.diag([30]*n + [30]*n)
R = np.eye(n)
P = solve_continuous_are(A, B, Q, R)
K = np.linalg.inv(R) @ B.T @ P
Kd, Kp = np.split(K, 2, axis=1)

q_des = np.array([[ 0.7262458 ], [ 1.61760955], [ 0.11582987], [-1.14679451], [ 2.16399157], [ 2.76812822]])
dyn_controller = Controller(robot, htm_des, 'torque')
kin_controller = Controller(robot, htm_des, 'qp')
e = [np.inf]*3
q = robot.q.copy()
sigma = np.zeros((n, 1))
qdot = np.zeros((n, 1))
q_hist, qdot_hist, e_hist, qdotdes_hist, sigma_hist, torque_hist = [], [], [], [], [], []
T, dt, tol = 30, 0.05, 1e-3
i = 1
imax = int(T//dt)
disturbance=0

# while (i < imax) and (np.linalg.norm(e) > tol):
while (i < imax):
    qdot_des, e, *_ = kin_controller.control(K=2, ignore_joint_limits=False)
    q_des = q + dt*qdot_des
    if i >= 200:
        disturbance = 5E-3 * 0
        if i>= 300:
            disturbance = 10e-3 * 0
    qddot, torque  = dyn_controller.control(qdot, qdot_des, q_des=q_des, Ki=0, Kd=Kd, Kp=Kp, disturbance=disturbance)
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
# %%
"""Lyapunov Redesign --- WORKING"""
htm_des = Utils.trn([0.2, 0.4, 0.7]) @ Utils.rotx(np.deg2rad(-100)) @ Utils.rotz(np.deg2rad(33))
frame_des = Frame(name='frame_des', htm=htm_des)
robot = create_jaco2(thesis_parameters=False)
real_robot = create_jaco2(thesis_parameters=True)
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

#PD
n = len(robot.links)
A = np.block([[np.zeros((n, 2*n))], [np.block([np.eye(n), np.zeros((n, n))])]])
B = np.block([[np.eye(n)], [np.zeros((n, n))]])
Q = 0.01*np.diag([30]*n + [30]*n)
R = np.eye(n)
P = solve_continuous_are(A, B, Q, R)
K = np.linalg.inv(R) @ B.T @ P
Kd, Kp = np.split(K, 2, axis=1)

A_lyap = np.block([[np.block([-Kd, -Kp])], [np.block([np.eye(n), np.zeros((n, n))])]])
Q_lyap = -np.eye(2*n)
P_lyap = solve_continuous_lyapunov(A_lyap, Q_lyap)

#PID
# n = len(robot.links)
# A = np.block([[np.zeros((n, 3*n))], [np.block([np.eye(n), np.zeros((n, 2*n))])],[np.block([np.zeros((n, n)), np.eye(n), np.zeros((n, n))])]])
# B = np.block([[np.eye(n)], [np.zeros((2*n, n))]])
# Q = 24*np.diag([30]*n + [30]*n + [880]*n)
# R = np.eye(n)
# P = solve_continuous_are(A, B, Q, R)
# K = np.linalg.inv(R) @ B.T @ P
# Kd, Kp, Ki = np.split(K, 3, axis=1)

# A_lyap = np.block([[np.block([-Kd, -Kp, -Ki])], [np.block([np.eye(n), np.zeros((n, 2*n))])], [np.block([np.zeros((n, n)), np.eye(n), np.zeros((n, n))])]])
# Q_lyap = -np.eye(3*n)
# P_lyap = solve_continuous_lyapunov(A_lyap, Q_lyap)

# wn = 0.05
# Kp = np.diag([wn**2]*n)
# Kd = np.diag([wn]*n)

q_des = np.array([[ 0.7262458 ], [ 1.61760955], [ 0.11582987], [-1.14679451], [ 2.16399157], [ 2.76812822]])
dyn_controller = Controller(robot, htm_des, 'lyapunov')
kin_controller = Controller(robot, htm_des, 'qp')
e = [np.inf]*3
q = robot.q.copy()
sigma = np.zeros((n, 1))
qdot = np.zeros((n, 1))
q_hist, qdot_hist, qddot_hist, e_hist, qdotdes_hist, sigma_hist, torque_hist = [], [], [], [], [], [], []
T, dt, tol = 30, 0.05, 1e-3
i = 1
imax = int(T//dt)
disturbance=0
small_limit = 0.62832 # 6rpm
big_limit = 0.83776 # 8rpm
qdot_limits = np.array([[big_limit],[big_limit],[big_limit],[small_limit],[small_limit],[small_limit]])

# while (i < imax) and (np.linalg.norm(e) > tol):
while (i < imax):
    qdot_des, e, *_ = kin_controller.control(K=2, ignore_joint_limits=False)
    q_des = q + dt*qdot_des
    if i >= 200:
        disturbance = 5E-3
        if i>= 300:
            disturbance = 10e-3*np.sin(i*dt)#10e-3
    qddot, torque  = dyn_controller.control(qdot, qdot_des, real_robot=real_robot, P=P_lyap, q_des=q_des, Ki=0, Kd=Kd, Kp=Kp, disturbance=disturbance, epsilon=1)
    e, _ = dyn_controller.error_func(htm_des, )
    sigma = sigma + dt*(q - q_des)
    q = robot.q + qdot * dt
    # qdot = qdot + qddot * dt
    qdot = np.clip(qdot + qddot * dt, -qdot_limits, qdot_limits)
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
    qddot_hist.append(qddot)
    torque_hist.append(torque)


sim.run()
fig = px.line(np.array(e_hist).reshape(-1, 6))
fig.show()
fig = px.line(np.array(qdot_hist).reshape(-1, 6))
fig.show()
fig = px.line(np.array(qdotdes_hist).reshape(-1, 6))
fig.show()

# %%
"""Lyapunov Redesign PID"""
htm_des = Utils.trn([0.2, 0.4, 0.7]) @ Utils.rotx(np.deg2rad(-100)) @ Utils.rotz(np.deg2rad(33))
frame_des = Frame(name='frame_des', htm=htm_des)
robot = create_jaco2(thesis_parameters=False)
real_robot = create_jaco2(thesis_parameters=True)
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

#PD
#PID
n = len(robot.links)
A = np.block([[np.zeros((n, 3*n))], [np.block([np.eye(n), np.zeros((n, 2*n))])],[np.block([np.zeros((n, n)), np.eye(n), np.zeros((n, n))])]])
B = np.block([[np.eye(n)], [np.zeros((2*n, n))]])
Q = 0.01*np.diag([30]*n + [30]*n + [30]*n)
R = np.eye(n)
P = solve_continuous_are(A, B, Q, R)
K = np.linalg.inv(R) @ B.T @ P
Kd, Kp, Ki = np.split(K, 3, axis=1)

A_lyap = np.block([[np.block([-Kd, -Kp, -Ki])], [np.block([np.eye(n), np.zeros((n, 2*n))])], [np.block([np.zeros((n, n)), np.eye(n), np.zeros((n, n))])]])
Q_lyap = -np.eye(3*n)
P_lyap = solve_continuous_lyapunov(A_lyap, Q_lyap)

# wn = 0.05
# Kp = np.diag([wn**2]*n)
# Kd = np.diag([wn]*n)

q_des = np.array([[ 0.7262458 ], [ 1.61760955], [ 0.11582987], [-1.14679451], [ 2.16399157], [ 2.76812822]])
dyn_controller = Controller(robot, htm_des, 'lyapunov')
kin_controller = Controller(robot, htm_des, 'qp')
e = [np.inf]*3
q = robot.q.copy()
sigma = np.zeros((n, 1))
qdot = np.zeros((n, 1))
q_hist, qdot_hist, qddot_hist, e_hist, qdotdes_hist, sigma_hist, torque_hist = [], [], [], [], [], [], []
T, dt, tol = 30, 0.05, 1e-3
i = 1
imax = int(T//dt)
disturbance=0
small_limit = 0.62832 # 6rpm
big_limit = 0.83776 # 8rpm
qdot_limits = np.array([[big_limit],[big_limit],[big_limit],[small_limit],[small_limit],[small_limit]])

# while (i < imax) and (np.linalg.norm(e) > tol):
while (i < imax):
    qdot_des, e, *_ = kin_controller.control(K=1, ignore_joint_limits=False)
    q_des = q + dt*qdot_des
    if i >= 200:
        disturbance = 5E-3
        if i>= 300:
            disturbance = 10e-3
    qddot, torque  = dyn_controller.control(qdot, qdot_des, q_des=q_des, sigma=sigma, real_robot=real_robot, P=P_lyap, Ki=Ki, Kd=Kd, Kp=Kp, disturbance=disturbance, epsilon=1)
    e, _ = dyn_controller.error_func(htm_des, )
    sigma = sigma + dt*(q - q_des)
    q = robot.q + qdot * dt
    # qdot = qdot + qddot * dt
    qdot = np.clip(qdot + qddot * dt, -qdot_limits, qdot_limits)
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
    qddot_hist.append(qddot)
    torque_hist.append(torque)


sim.run()
fig = px.line(np.array(e_hist).reshape(-1, 6))
fig.show()
fig = px.line(np.array(qdot_hist).reshape(-1, 6))
fig.show()
fig = px.line(np.array(qdotdes_hist).reshape(-1, 6))
fig.show()
# %%
""" Extrema for Lyapunov Redesign"""
rng = np.random.default_rng()
robot = create_jaco2(thesis_parameters=False)
joint_limits = robot.joint_limit
small_limit = 0.62832 # 6rpm
big_limit = 0.83776 # 8rpm
qdot_limits = np.array([[big_limit],[big_limit],[big_limit],[small_limit],[small_limit],[small_limit]])
M_norm = []
C_norm = []
G_norm = []
imax=100000
for i in range(imax):
    x = int(60 * i / (imax - 1))
    print(f"[{u'█'*x}{('.'*(60-x))}] {i}/{imax}", end='\r', file=sys.stdout, flush=True)
    q_test = rng.uniform(joint_limits[:, 0], joint_limits[:, 1], (6, 1))
    qdot_test = rng.uniform(-qdot_limits[:], qdot_limits[:], (6, 1))
    M, C, G = robot.dyn_model(q_test, qdot_test)
    M_norm.append(np.linalg.norm(M, 2))
    C_norm.append(np.linalg.norm(C, 2))
    G_norm.append(np.linalg.norm(G, 2))

print(np.max(M_norm))
print(np.min(M_norm))
print(np.max(C_norm))
print(np.min(C_norm))
print(np.max(G_norm))
print(np.min(G_norm))
with open('/home/fbartelt/Documents/Projetos/robotics-experiments/MCGlimits.txt', 'w') as file:
    file.write(f'Mlim {np.min(M_norm)} {np.max(M_norm)}\n')
    file.write(f'Clim {np.min(C_norm)} {np.max(C_norm)}\n')
    file.write(f'Glim {np.min(G_norm)} {np.max(G_norm)}\n')
# %%
"""Lyapunov Redesign TESTING"""
T, dt, tol = 30, 0.05, 1e-3
omega = 0.8
radius = 0.2
traj = [radius * np.cos(0*dt * omega), 0.5, radius * np.sin(0*dt * omega) + 0.5]
htm_des = Utils.trn(traj)
# htm_des = Utils.trn([0.2, 0.4, 0.7]) @ Utils.rotx(np.deg2rad(-100)) @ Utils.rotz(np.deg2rad(33))
frame_des = Frame(name='frame_des', htm=htm_des)
robot = create_jaco2(thesis_parameters=False)
real_robot = create_jaco2(thesis_parameters=True)
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

#PD
n = len(robot.links)
A = np.block([[np.zeros((n, 2*n))], [np.block([np.eye(n), np.zeros((n, n))])]])
B = np.block([[np.eye(n)], [np.zeros((n, n))]])
Q = 0.01*np.diag([30]*n + [30]*n)
Q = 0.01*np.diag([30]*n + [0]*n)
R = np.eye(n)
P = solve_continuous_are(A, B, Q, R)
K = np.linalg.inv(R) @ B.T @ P
Kd, Kp = np.split(K, 2, axis=1)

A_lyap = np.block([[np.block([-Kd, -Kp])], [np.block([np.eye(n), np.zeros((n, n))])]])
A_lyap = np.block([[-K], [np.block([np.eye(n), np.zeros((n, n))])]])
Q_lyap = -np.eye(2*n)
P_lyap = solve_continuous_lyapunov(A_lyap, Q_lyap)

q_des = np.array([[ 0.7262458 ], [ 1.61760955], [ 0.11582987], [-1.14679451], [ 2.16399157], [ 2.76812822]])
dyn_controller = Controller(robot, htm_des, 'lyapunov')
kin_controller = Controller(robot, htm_des, 'qp')
e = [np.inf]*3
q = robot.q.copy()
sigma = np.zeros((n, 1))
qdot = np.zeros((n, 1))
qdot_des = np.zeros((n, 1))
q_hist, qdot_hist, qddot_hist, e_hist, qdotdes_hist, sigma_hist, torque_hist = [], [], [], [], [], [], []
T, dt, tol = 30, 0.05, 1e-3
i = 1
imax = int(T//dt)
disturbance=0
small_limit = 0.62832 # 6rpm
big_limit = 0.83776 # 8rpm
qdot_limits = np.array([[big_limit],[big_limit],[big_limit],[small_limit],[small_limit],[small_limit]])

# while (i < imax) and (np.linalg.norm(e) > tol):
# alpha=4
# xt = (3/2* radius + radius * np.cos(t * omega * alpha))*np.cos(t*omega)
# yt = (3/2* radius + radius * np.cos(t * omega * alpha))*np.sin(t*omega)
# zt = radius * np.sin(t*omega*alpha)

while (i < imax):
    omega = 0.8
    radius = 0.2
    traj = [radius * np.cos(i*dt * omega), 0.5, radius * np.sin(i*dt * omega) + 0.5]
    htm_des = Utils.trn(traj)
    old_qdot_des = qdot_des
    qdot_des, e, *_ = kin_controller.control(K=2, ignore_joint_limits=False, htm_des=htm_des)
    q_des = q + dt*qdot_des
    qddot_des = (qdot_des - old_qdot_des)/dt
    if i >= 300:
        disturbance = 5E-3
        if i>= 400:
            disturbance = 10e-3*np.sin(i*dt)#10e-3
    qddot, torque  = dyn_controller.lyapunov_redesign_test(qdot, qdot_des, qddot_des=qddot_des, real_robot=real_robot, P=P_lyap, q_des=q_des, Ki=0, Kd=Kd, Kp=Kp, disturbance=disturbance, epsilon=1)
    e, _ = dyn_controller.error_func(htm_des, )
    sigma = sigma + dt*(q - q_des)
    q = robot.q + qdot * dt
    # qdot = qdot + qddot * dt
    qdot = np.clip(qdot + qddot * dt, -qdot_limits, qdot_limits)
    robot.add_ani_frame(time=i*dt, q=q)
    htms = robot.fkm(q=q, axis='dh')
    for j, frame in enumerate(frames):
        frame.add_ani_frame(time=i*dt, htm=htms[j])
    frame_des.add_ani_frame(time=(i+1)*dt, htm=htm_des)
    i+=1
    q_hist.append(q)
    qdot_hist.append(qdot)
    qdotdes_hist.append(qdot_des)
    e_hist.append(e)
    sigma_hist.append(sigma)
    qddot_hist.append(qddot)
    torque_hist.append(torque)

pos_hist = [robot.fkm(q=x)[:3, -1] for x in q_hist]
traj_points = PointCloud(name='traj', points=np.array(pos_hist).reshape(-1, 3).T, color='cyan', size=8)
sim.add(traj_points)
for i in range(len(pos_hist)):
    traj_points.add_ani_frame((i+1)*dt, 0, i)


sim.run()
# fig = px.line(np.array(e_hist).reshape(-1, 6))
# fig.show()
# fig = px.line(np.array(qdot_hist).reshape(-1, 6))
# fig.show()
# fig = px.line(np.array(qdotdes_hist).reshape(-1, 6))
# fig.show()

# traj = [[radius * np.cos(i*dt * omega), 0.5, radius * np.sin(i*dt * omega) + 0.5] for i in range(1, imax)]
# pos_hist = [robot.fkm(q=x)[:3, -1] for x in q_hist]
# px.line(np.array(traj).reshape(-1, 3) - np.array(pos_hist).reshape(-1, 3))
# %%
alpha=4
xt = (4/2* radius + radius * np.cos(t * omega * alpha))*np.cos(t*omega)
yt = (4/2* radius + radius * np.cos(t * omega * alpha))*np.sin(t*omega)
zt = radius * np.sin(t*omega*alpha)

fig = go.Figure(data=[go.Scatter3d(x=xt, y=yt, z=zt,
                                   mode='markers')])
fig.show()
#%%
"""PD TESTING"""
T, dt, tol = 30, 0.05, 1e-3
omega = 0.8
radius = 0.2
traj = [radius * np.cos(0*dt * omega), 0.5, radius * np.sin(0*dt * omega) + 0.5]
htm_des = Utils.trn(traj)
# htm_des = Utils.trn([0.2, 0.4, 0.7]) @ Utils.rotx(np.deg2rad(-100)) @ Utils.rotz(np.deg2rad(33))
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
A = np.block([[np.zeros((n, 2*n))], [np.block([np.eye(n), np.zeros((n, n))])]])
B = np.block([[np.eye(n)], [np.zeros((n, n))]])
Q = 0.01*np.diag([30]*n + [0]*n)
# Q = 0.01*np.diag([30]*n + [30]*n)
R = np.eye(n)
P = solve_continuous_are(A, B, Q, R)
K = np.linalg.inv(R) @ B.T @ P
Kd, Kp = np.split(K, 2, axis=1)

q_des = np.array([[ 0.7262458 ], [ 1.61760955], [ 0.11582987], [-1.14679451], [ 2.16399157], [ 2.76812822]])
dyn_controller = Controller(robot, htm_des, 'torque')
kin_controller = Controller(robot, htm_des, 'qp')
e = [np.inf]*3
q = robot.q.copy()
sigma = np.zeros((n, 1))
qdot = np.zeros((n, 1))
qdot_des = np.zeros((n, 1))
q_hist, qdot_hist, e_hist, qdotdes_hist, sigma_hist, torque_hist = [], [], [], [], [], []
i = 1
imax = int(T//dt)
disturbance=0
small_limit = 0.62832 # 6rpm
big_limit = 0.83776 # 8rpm
qdot_limits = np.array([[big_limit],[big_limit],[big_limit],[small_limit],[small_limit],[small_limit]])
# while (i < imax) and (np.linalg.norm(e) > tol):
while (i < imax):
    traj = [radius * np.cos(i*dt * omega), 0.5, radius * np.sin(i*dt * omega) + 0.5]
    htm_des = Utils.trn(traj)
    old_qdot_des = qdot_des
    qdot_des, e, *_ = kin_controller.control(K=2, ignore_joint_limits=False, htm_des=htm_des)
    qddot_des = (qdot_des - old_qdot_des)/dt
    q_des = q + dt*qdot_des
    if i >= 300:
        disturbance = 5E-3
        if i>= 400:
            disturbance = 10e-3*np.sin(i*dt)#10e-3
    qddot, torque  = dyn_controller.control(qdot, qdot_des, q_des=q_des, qddot_des=qddot_des, Ki=0, Kd=Kd, Kp=Kp, disturbance=disturbance)
    e, _ = dyn_controller.error_func(htm_des, )
    sigma = sigma + dt*(q_des - q)
    q = robot.q + qdot * dt
    # qdot = qdot + dt * qddot
    qdot = np.clip(qdot + qddot * dt, -qdot_limits, qdot_limits)
    robot.add_ani_frame(time=i*dt, q=q)
    htms = robot.fkm(q=q, axis='dh')
    for j, frame in enumerate(frames):
        frame.add_ani_frame(time=i*dt, htm=htms[j])
    frame_des.add_ani_frame(time=(i+1)*dt, htm=htm_des)
    i+=1
    q_hist.append(q)
    qdot_hist.append(qdot)
    qdotdes_hist.append(qdot_des)
    e_hist.append(e)
    sigma_hist.append(sigma)
    torque_hist.append(torque)


pos_hist = [robot.fkm(q=x)[:3, -1] for x in q_hist]
traj_points = PointCloud(name='traj', points=np.array(pos_hist).reshape(-1, 3).T, color='cyan', size=8)
sim.add(traj_points)
for i in range(len(pos_hist)):
    traj_points.add_ani_frame((i+1)*dt, 0, i)

sim.run()
#Norm of e
# fig = px.line(x=np.linspace(0, T, imax-1), y=np.linalg.norm(np.array(e_hist).reshape(-1, 6), axis=1))
# fig.update_xaxes(title=r'$t$')
# fig.update_yaxes(title=r'$\| e(q)\|$')
# fig.update_layout(margin=dict(t=10))
# fig.show()

# #qdot_des
# fig = go.Figure()
# qdotdes_hist = np.array(qdotdes_hist).reshape(-1, 6)
# for i in range(n):
#     fig.add_scatter(x=np.linspace(0, T, imax-1), y=qdotdes_hist[:, i], name=fr'$\dot{{q}}_{{{i+1}, des}}$')
# fig.update_xaxes(title=r'$t$')
# fig.update_yaxes(title=r'$\dot{{q}}_\text{des}$')
# fig.update_layout(margin=dict(t=10))
# fig.show()

# #torque
# fig = go.Figure()
# torque_hist = np.array(torque_hist).reshape(-1, 6)
# for i in range(n):
#     fig.add_scatter(x=np.linspace(0, T, imax-1), y=torque_hist[:, i], name=fr'$\tau_{i+1}$')
# fig.update_xaxes(title=r'$t$')
# fig.update_yaxes(title=r'$\tau$')
# fig.update_layout(margin=dict(t=10))
# fig.show()

# # qdot_des - qdot
# fig = go.Figure()
# qdotdes_hist = np.array(qdotdes_hist).reshape(-1, 6)
# qdot_hist = np.array(qdot_hist).reshape(-1, 6)
# for i in range(n):
#     fig.add_scatter(x=np.linspace(0, T, imax-1), y=qdotdes_hist[:, i]-qdot_hist[:, i], name=fr'${i}$')
# fig.update_xaxes(title=r'$t$')
# fig.update_yaxes(title=r'$\dot{{q}}_\text{des} - \dot{q}$')
# fig.update_layout(margin=dict(t=10))


# traj = [[radius * np.cos(i*dt * omega), 0.5, radius * np.sin(i*dt * omega) + 0.5] for i in range(1, imax)]
# pos_hist = [robot.fkm(q=x)[:3, -1] for x in q_hist]
# px.line(np.array(traj).reshape(-1, 3) - np.array(pos_hist).reshape(-1, 3))

# ### fig.write_image('./pd_img.pdf')
# %%
