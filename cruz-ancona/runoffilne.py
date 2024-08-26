import sys
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import plotly.figure_factory as ff

sys.path.insert(1, "/home/fbartelt/Documents/UFMG/TCC/Sim/uaibot")
import uaibot as ub
from uaibot.utils import Utils
from uaibot.simobjects.pointcloud import PointCloud
from uaibot.simulation import Simulation
from uaibot.simobjects.pointlight import PointLight
from uaibot.simobjects.ball import Ball
from uaibot.robot._vector_field import _compute_ntd as compute_ntd
from cvxopt import matrix, solvers, spmatrix
from uaibot_addons.create_jaco import create_jaco2
from plotly.subplots import make_subplots
from itertools import product
from uaibot_addons.vectorfield import VectorField, vector_field_plot
from uaibot_addons.math import dot_J
from scipy.linalg import solve_continuous_are

solvers.options["show_progress"] = False

"""
VectorField acceleration causes RuntimeWarning: invalid value encountered in sqrt
  eta = (-psi_s.T @ psi_t + np.sqrt((psi_s.T @ psi_t)**2 + self.const_vel**2 - psi_t.T @ psi_t))[0, 0]

Probably due to tangent field with norm greater than vr.
"""

def pseudo_inverse_control(robot, htm_des, K=1):
        n = len(robot.links)
        if not isinstance(K, np.ndarray):
            if isinstance(K, (list, tuple)):
                K = np.diag(K)
            else:
                K = np.diag([K]*n)
        e, Je = error_func(htm_des)
        qdot = - K @ np.linalg.pinv(Je) @ e
        return qdot, e

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

""" VECTOR FIELD Adaptive Lyapunov Redesign"""
robot = create_jaco2(thesis_parameters=True)
light1 = PointLight(name="light1", color="white", intensity=2.5, htm=Utils.trn([-1,-1, 1.5]))
light2 = PointLight(name="light2", color="white", intensity=2.5, htm=Utils.trn([-1, 1, 1.5]))
light3 = PointLight(name="light3", color="white", intensity=2.5, htm=Utils.trn([ 1,-1, 1.5]))
light4 = PointLight(name="light4", color="white", intensity=2.5, htm=Utils.trn([ 1, 1, 1.5]))
sim = Simulation.create_sim_grid([robot])#, light1, light2, light3, light4])
sim.set_parameters(width=800, height=600, ambient_light_intensity=4)

n = len(robot.links)

# Parametric equation definition
maxtheta = 500
def eq2(time=0):
    ## 0.5e-1, 0.025e-1, 0.6, 0.6, 0.3, 0.3 works with acceleration approx
    ## 0.5e-1, 0.025e-1, 0.6, 0.6, 0.3, 0.3 works with analytic acceleration, Kd=50, const_vel=1, alpha=5
    ## 0.5e-1, 0.025e1, 0.6, 0.6, 0.3, 0.3 works with analytic acceleration, Kd=50, const_vel=1, alpha=5
    ## Above doesnt work with alpha=1
    w1, w2, c1, c2, c3, h0 = 0.5e-1*0, 0.025e1*0, 0.6, 0.6, 0.3, 0.3
    rotz = np.matrix([[np.cos(w1*time), -np.sin(w1*time), 0],
                      [np.sin(w1*time), np.cos(w1*time), 0],
                      [0, 0, 1]])
    theta = np.linspace(0, 2 * np.pi, num=maxtheta)
    curve = np.array([rotz @ np.array([c1*np.cos(s), c2*np.sin(s), h0 + c3*np.cos(w2*time)*np.cos(s)**2]).reshape(-1, 1) for s in theta]).reshape(3, -1)
    curve = curve.reshape(-1, 3).T

    return curve

# Simulation parameters and initializations
T = 2
dt = 1e-6
Kd = np.eye(n) * 50
imax = int(T/dt)
qdot = np.zeros((n, 1))

# Add trajectory and nearest point to simulation
print('Creating point cloud')
a = None
for i in range(int(T/0.01)):
    curve = eq2(i*dt)
    if a is None:
        a = curve
    else:
        a = np.hstack((a, curve))
traj = PointCloud(name='traj', points=a, size=12, color='cyan')
sim.add([traj])
vf = VectorField(eq2, False, alpha=5, const_vel=1)
nearest_point = Ball(name='nearest_point', radius=0.03, color='red', htm=Utils.trn([0, 0, 0]), opacity=0.7)
sim.add([nearest_point])
print('Done')

def psbf(z, epsilon):
    return np.linalg.norm(z) / (epsilon - np.linalg.norm(z))

def qqdotqddot(t):
    q_d = np.block([np.array([np.sin(0.2*t), np.cos(0.2*t)]), np.ones(4)*0.2]).reshape(-1, 1)
    qdot_d = np.block([np.array([np.cos(0.2*t), -np.sin(0.2*t)]), np.zeros(4)]).reshape(-1, 1)
    qddot_d = np.block([np.array([-np.sin(0.2*t), -np.cos(0.2*t)]), np.zeros(4)]).reshape(-1, 1)
    return q_d, qdot_d, qddot_d

#%%
# Plot-related
hist_time = []
hist_qdot = np.matrix(np.zeros((6,0)))
hist_qdot_des = np.matrix(np.zeros((6,0)))
hist_qddot = np.matrix(np.zeros((6,0)))
hist_qddot_des = np.matrix(np.zeros((6,0)))
hist_q = np.matrix(np.zeros((6,0)))
hist_error_ori = np.matrix(np.zeros((3,0)))
hist_peef = np.zeros((3, 0))
hist_vf = np.zeros((3, 0))
hist_cond_J = []
hist_cond_Jdot = [] 
hist_x = np.matrix(np.zeros((12,0)))

# Adaptive Lyapunov Redesign parameters
A = np.block([[np.zeros((n, n)), np.eye(n)], [np.zeros((n, n)), np.zeros((n, n))]])
B = np.block([[np.zeros((n, n))], [np.eye(n)]])
varrho = 5e-1
P = solve_continuous_are(A, B, 2*np.eye(2*n), varrho * np.eye(n))
epsilon = 1
l = 10
L = np.eye(3) * 1e-1
Xi = np.diag([1, 1e-1, 1e-3])
rho0 = 11
b0 = np.array([1, 1, 1]).reshape(-1, 1) * 1e-1

rho = rho0
b = b0


for i in range(imax):
    # if i % 50 == 0 or i == imax - 1:
    sys.stdout.write('\r')
    sys.stdout.write("[%-20s] %d%%" % ('=' * round(20 * i / (imax - 1)), round(100 * i / (imax - 1))))
    sys.stdout.flush()
    q = robot.q.copy()
    jac_eef, htm_eef = robot.jac_geo()
    p_eef = htm_eef[0:3, 3]

    # target = np.matrix(np.zeros((3, 1)))
    # target[0:3] = vf(p_eef, i*dt)
    # jac_target = np.matrix(np.zeros((3, n)))
    # jac_target[0:3, :] = jac_eef[0:3, :]
    
    # q_des = robot.ikm(Utils.trn(vf.nearest_points[-1]), ignore_orientation=True)
    q_des, qdot_des, qddot_des = qqdotqddot(i*dt)
    # qdot_des = np.linalg.pinv(jac_target) @ target
    x1 = q - q_des
    x2 = qdot - qdot_des
    x = np.block([[x1], [x2]])
    # a_des = vf.acceleration(p_eef, jac_target @ qdot, i*dt)
    Jdot = dot_J(robot, qdot, q)[:3, :]
    # qddot_des = np.linalg.pinv(jac_target) @ (a_des - Jdot @ qdot)

    M, C_bar, g = robot.dyn_model(q, qdot)
    rng = np.random.default_rng()
    Delta = rng.normal(0, 0.1, (n, n)) * 0
    Mtilde = M + Delta
    delta = 2*np.sin(i*dt) * 0
    G = np.linalg.inv(M)
    DeltaG = (M @ np.linalg.inv(Mtilde) - np.eye(n))
    h = G @ (np.eye(n) + DeltaG) @ (-C_bar - g + delta - (Mtilde @ qddot_des))
    psi = -varrho * B.T @ P @ x
    w = B.T @ P @ x
    B_bar = B.T @ P @ B @ G
    w_bar = B_bar.T @ w
    w_norm = np.linalg.norm(w)
    w_bar_norm = np.linalg.norm(w_bar)
    kappa = np.block([[1], [np.linalg.norm(x)], [x.T @ x]]).reshape(-1, 1)
    Gamma_bar = kappa.T @ b
    if w_norm > (epsilon/2):
        k = (w_bar_norm + Gamma_bar + rho/w_bar_norm)[0, 0]
    else:
        k = psbf(w, epsilon)
    v = -k * w_bar / w_bar_norm
    tau = psi + v


    xdot = A @ x + B @ (G @ (np.eye(n) + DeltaG) @ tau + h)
    bdot = L @ (kappa * w_bar_norm - Xi @ b)
    rhodot = l - rho

    x = x + xdot * dt
    b = b + bdot * dt
    rho = rho + rhodot * dt

    q = q_des + x[:n, :]
    qdot = qdot_des + x[n:, :]
    qddot = xdot[n:, :]

    robot.add_ani_frame(time=i*dt, q=q)
    # traj.add_ani_frame(time=i*dt, initial_ind=maxtheta*i, final_ind=maxtheta*(i+1))
    # nearest_point.add_ani_frame(time=i*dt, htm=Utils.trn(vf.nearest_points[i]))

    hist_time.append(i * dt)
    hist_q = np.block([hist_q, robot.q])
    hist_peef = np.block([hist_peef, p_eef])
    # hist_vf = np.block([hist_vf, target[0:3]])
    hist_qdot = np.block([hist_qdot, qdot])
    hist_qdot_des = np.block([hist_qdot_des, qdot_des])
    hist_qddot = np.block([hist_qddot, qddot])
    hist_qddot_des = np.block([hist_qddot_des, qddot_des])
    # hist_cond_J.append(np.linalg.cond(jac_target))
    # hist_cond_Jdot.append(np.linalg.cond(Jdot))
    hist_x = np.block([hist_x, x])

sim.save('figures', 'cbf_lyapredesign')
# sim.run()
# hist_vf = np.array(hist_vf)
hist_peef = np.array(hist_peef)
# fig = vector_field_plot(hist_peef, hist_vf, add_lineplot=True, sizemode="absolute", sizeref=2.5, anchor='tail')
# fig.write_image("figures/vectorfield.pdf")
# fig.show()
fig=px.line(np.linalg.norm(hist_x, axis=0).T, title='|x|')
fig.write_image("figures/histx.pdf")
# fig.show()
fig=px.line(np.linalg.norm(hist_qdot-hist_qdot_des, axis=0).T, title='|dq/dt - dq<sub>des</sub>/dt|')
# fig.show()
fig.write_image("figures/qdoterrNorm.pdf")
fig=px.line(np.abs(hist_qdot-hist_qdot_des).T, title='abs(dq/dt - dq<sub>des</sub>/dt)')
fig.write_image("figures/qdoterr.pdf")
# fig.show()
fig=px.line(np.abs(hist_peef-np.array(vf.nearest_points).reshape(-1, 3).T).T, title='|p<sub>eef</sub> - x*|')
fig.write_image("figures/poserr.pdf")
# fig.show()
