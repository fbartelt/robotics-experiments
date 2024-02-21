#%%
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
from create_jaco import create_jaco2
from plotly.subplots import make_subplots
from itertools import product
from uaibot_addons.vectorfield import VectorField, vector_field_plot
from uaibot_addons.math import dot_J

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

# %%
""" JACO TEST Vector Field KINEMATIC CONTROL """
robot = create_jaco2(thesis_parameters=True)
light1 = PointLight(name="light1", color="white", intensity=2.5, htm=Utils.trn([-1,-1, 1.5]))
light2 = PointLight(name="light2", color="white", intensity=2.5, htm=Utils.trn([-1, 1, 1.5]))
light3 = PointLight(name="light3", color="white", intensity=2.5, htm=Utils.trn([ 1,-1, 1.5]))
light4 = PointLight(name="light4", color="white", intensity=2.5, htm=Utils.trn([ 1, 1, 1.5]))
sim = Simulation.create_sim_grid([robot])#, light1, light2, light3, light4])
sim.set_parameters(width=800, height=600, ambient_light_intensity=4)

n = len(robot.links)

T = 10
dt = 0.01
imax = int(T/dt)
K = 1

points = np.linspace(0, 0.5, 200)
curve = [[0, i, 0] for i in points]
curve.extend([[i, 0.5, 0] for i in points])
curve.extend([[0.5, i, 0] for i in points[::-1]])
curve.extend([[i, 0, 0] for i in points[::-1]])
curve = np.matrix(curve).T

theta = np.linspace(0, 2 * np.pi, num=300)
curve = np.matrix(np.zeros(  (3, len(theta))))
for i in range(len(theta)):
    t = theta[i]
    # curve[:,i] = np.matrix([ [0.56], [0.2 * np.cos(t)], [0.2 * np.sin(t) + 0.5]])
    curve[:,i] = np.matrix([ [0.5/3*(np.sin(t) + 2*np.sin(2*t))], [0.5/3*(np.cos(t) - 2*np.cos(2*t))], [0.45*(-np.sin(3*t)) + 0.45]])

traj = PointCloud(name='traj', points=curve, size=12, color='cyan')
sim.add([traj])
vf = robot.vector_field(curve, alpha=5, const_vel=2)

# Desired axis
x_des = np.matrix([0, 0, 1]).reshape((3, 1))
y_des = np.matrix([0, 1, 0]).reshape((3, 1))
z_des = np.matrix([1, 0, 0]).reshape((3, 1))

hist_time = []
hist_qdot = np.matrix(np.zeros((6,0)))
hist_q = np.matrix(np.zeros((6,0)))
hist_error_ori = np.matrix(np.zeros((3,0)))
hist_peef = np.zeros((3, 0))
hist_vf = np.zeros((3, 0))
for i in range(imax):
    jac_eef, htm_eef = robot.jac_geo()

    p_eef = htm_eef[0:3, 3]
    x_eef = htm_eef[0:3, 0]
    y_eef = htm_eef[0:3, 1]
    z_eef = htm_eef[0:3, 2]

    target = np.matrix(np.zeros((3, 1)))
    target[0:3] = vf(p_eef)
    # target[3] = -K * np.sqrt(max(1 - x_des.T * x_eef, 0))
    # target[4] = -K * np.sqrt(max(1 - y_des.T * y_eef, 0))
    # target[5] = -K * np.sqrt(max(1 - z_des.T * z_eef, 0))

    jac_target = np.matrix(np.zeros((3, n)))
    jac_target[0:3, :] = jac_eef[0:3, :]
    # jac_target[3, :] = x_des.T * Utils.S(x_eef) * jac_eef[3:6, :]
    # jac_target[4, :] = y_des.T * Utils.S(y_eef) * jac_eef[3:6, :]
    # jac_target[5, :] = z_des.T * Utils.S(z_eef) * jac_eef[3:6, :]

    qdot = Utils.dp_inv(jac_target, 0.002) * target

    q = robot.q + qdot * dt
    robot.add_ani_frame(time=i*dt, q=q)
    traj.add_ani_frame(time=i*dt, initial_ind=0, final_ind=curve.shape[1])

    hist_time.append(i * dt)
    hist_q = np.block([hist_q, robot.q])
    hist_peef = np.block([hist_peef, p_eef])
    hist_vf = np.block([hist_vf, target[0:3]])
    # error_ori = np.matrix([(180 / (np.pi)) * np.arccos(1 - min(num * num / (K * K),2)) for num in target[3:6]]).reshape((3,1))
    # hist_error_ori = np.block([hist_error_ori, error_ori])
    hist_qdot = np.block([hist_qdot, qdot])


sim.run()
hist_vf = np.array(hist_vf)
hist_peef = np.array(hist_peef)
fig = vector_field_plot(hist_peef, hist_vf, add_lineplot=True, sizemode="absolute", sizeref=2.5, anchor='tail')
fig.show()


# %%
""" VECTOR FIELD Dynamic Control"""
robot = create_jaco2(thesis_parameters=True)
light1 = PointLight(name="light1", color="white", intensity=2.5, htm=Utils.trn([-1,-1, 1.5]))
light2 = PointLight(name="light2", color="white", intensity=2.5, htm=Utils.trn([-1, 1, 1.5]))
light3 = PointLight(name="light3", color="white", intensity=2.5, htm=Utils.trn([ 1,-1, 1.5]))
light4 = PointLight(name="light4", color="white", intensity=2.5, htm=Utils.trn([ 1, 1, 1.5]))
sim = Simulation.create_sim_grid([robot])#, light1, light2, light3, light4])
sim.set_parameters(width=800, height=600, ambient_light_intensity=4)

n = len(robot.links)

T = 10
dt = 0.01
imax = int(T/dt)
K = 1
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

def eq2_(time=0):
    theta = np.linspace(0, 2 * np.pi, num=maxtheta)
    curve = np.matrix(np.zeros(  (3, len(theta))))
    for i in range(len(theta)):
        t = theta[i]
        # curve[:,i] = np.matrix([ [0.56], [0.2 * np.cos(t)], [0.2 * np.sin(t) + 0.5]])
        curve[:,i] = np.matrix([ [0.5/3*(np.sin(t) + 2*np.sin(2*t))], [0.5/3*(np.cos(t) - 2*np.cos(2*t))], [0.45*(-np.sin(3*t)) + 0.45]])

    return curve

a = None
for i in range(imax):
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
# Desired axis
# x_des = np.matrix([0, 0, 1]).reshape((3, 1))
# y_des = np.matrix([0, 1, 0]).reshape((3, 1))
# z_des = np.matrix([1, 0, 0]).reshape((3, 1))

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
qdot = np.zeros((n, 1))
Kd = np.eye(n) * 50

for i in range(imax):
    if i % 50 == 0 or i == imax - 1:
        sys.stdout.write('\r')
        sys.stdout.write("[%-20s] %d%%" % ('=' * round(20 * i / (imax - 1)), round(100 * i / (imax - 1))))
        sys.stdout.flush()
    q = robot.q.copy()
    jac_eef, htm_eef = robot.jac_geo()

    p_eef = htm_eef[0:3, 3]
    x_eef = htm_eef[0:3, 0]
    y_eef = htm_eef[0:3, 1]
    z_eef = htm_eef[0:3, 2]

    target = np.matrix(np.zeros((3, 1)))
    target[0:3] = vf(p_eef, i*dt)
    # target[3] = -K * np.sqrt(max(1 - x_des.T * x_eef, 0))
    # target[4] = -K * np.sqrt(max(1 - y_des.T * y_eef, 0))
    # target[5] = -K * np.sqrt(max(1 - z_des.T * z_eef, 0))

    jac_target = np.matrix(np.zeros((3, n)))
    jac_target[0:3, :] = jac_eef[0:3, :]
    # jac_target[3, :] = x_des.T * Utils.S(x_eef) * jac_eef[3:6, :]
    # jac_target[4, :] = y_des.T * Utils.S(y_eef) * jac_eef[3:6, :]
    # jac_target[5, :] = z_des.T * Utils.S(z_eef) * jac_eef[3:6, :]

    # qdot_des = Utils.dp_inv(jac_target, 0.002) * target
    qdot_des = np.linalg.pinv(jac_target) @ target
    q_des = q 
    # qddot_des = (qdot_des - qdot) / dt 
    a_des = vf.acceleration(p_eef, jac_target @ qdot, i*dt)
    Jdot = dot_J(robot, qdot, q)[:3, :]
    qddot_des = np.linalg.pinv(jac_target) @ (a_des - Jdot @ qdot)
    M, C, G = robot.dyn_model(q, qdot)
    torque = C + G + (M @ (qddot_des + Kd @ (qdot_des - qdot)) )
    qddot = np.linalg.inv(M) @ (-C -G + torque)
    qddot = Kd @ (qdot_des - qdot) + qddot_des 

    qdot = qdot + qddot * dt
    q = robot.q + qdot * dt

    robot.add_ani_frame(time=i*dt, q=q)
    # traj.add_ani_frame(time=i*dt, initial_ind=0, final_ind=curve.shape[1])
    traj.add_ani_frame(time=i*dt, initial_ind=maxtheta*i, final_ind=maxtheta*(i+1))
    nearest_point.add_ani_frame(time=i*dt, htm=Utils.trn(vf.nearest_points[i]))

    hist_time.append(i * dt)
    hist_q = np.block([hist_q, robot.q])
    hist_peef = np.block([hist_peef, p_eef])
    hist_vf = np.block([hist_vf, target[0:3]])
    # error_ori = np.matrix([(180 / (np.pi)) * np.arccos(1 - min(num * num / (K * K),2)) for num in target[3:6]]).reshape((3,1))
    # hist_error_ori = np.block([hist_error_ori, error_ori])
    hist_qdot = np.block([hist_qdot, qdot])
    hist_qdot_des = np.block([hist_qdot_des, qdot_des])
    hist_qddot = np.block([hist_qddot, qddot])
    hist_qddot_des = np.block([hist_qddot_des, qddot_des])
    hist_cond_J.append(np.linalg.cond(jac_target))
    hist_cond_Jdot.append(np.linalg.cond(Jdot))


sim.run()
hist_vf = np.array(hist_vf)
hist_peef = np.array(hist_peef)
fig = vector_field_plot(hist_peef, hist_vf, add_lineplot=True, sizemode="absolute", sizeref=2.5, anchor='tail')
fig.show()
fig=px.line(np.linalg.norm(hist_qdot-hist_qdot_des, axis=0).T, title='|dq/dt - dq<sub>des</sub>/dt|')
fig.show()
fig=px.line(np.abs(hist_qdot-hist_qdot_des).T, title='abs(dq/dt - dq<sub>des</sub>/dt)')
fig.show()
fig=px.line(np.abs(hist_peef-np.array(vf.nearest_points).reshape(-1, 3).T).T, title='|p<sub>eef</sub> - x*|')
fig.show()
fig=px.line(hist_cond_J, title='J condition number')
fig.show()
fig=px.line(hist_cond_Jdot, title=r'\dot{J} condition number')
fig.show()

# %%
lista = (hist_qdot_des - hist_qdot) / dt

for i, qddot_ in enumerate(hist_qdot.T):
    qddot_approx = (hist_qdot_des[:, i] - qddot_.reshape(-1, 1)) / dt
    j, *_ = robot.jac_geo(q=hist_q[:, i])
    j = j[:3, :]
    a = vf.acceleration(hist_peef[:, i], j @ qddot_.reshape(-1, 1), i*dt)
    dotJ = dot_J(robot, qddot_.reshape(-1, 1), hist_q[:, i])[:3, :]
    qddot = np.linalg.pinv(j) @ (a - dotJ @ qddot_.reshape(-1, 1))


# %%
""" VECTOR FIELD Adaptive Lyapunov Redesign"""
robot = create_jaco2(thesis_parameters=True)
light1 = PointLight(name="light1", color="white", intensity=2.5, htm=Utils.trn([-1,-1, 1.5]))
light2 = PointLight(name="light2", color="white", intensity=2.5, htm=Utils.trn([-1, 1, 1.5]))
light3 = PointLight(name="light3", color="white", intensity=2.5, htm=Utils.trn([ 1,-1, 1.5]))
light4 = PointLight(name="light4", color="white", intensity=2.5, htm=Utils.trn([ 1, 1, 1.5]))
sim = Simulation.create_sim_grid([robot])#, light1, light2, light3, light4])
sim.set_parameters(width=800, height=600, ambient_light_intensity=4)

n = len(robot.links)

T = 10
dt = 0.01
imax = int(T/dt)
K = 1
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

a = None
for i in range(imax):
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
# Desired axis
# x_des = np.matrix([0, 0, 1]).reshape((3, 1))
# y_des = np.matrix([0, 1, 0]).reshape((3, 1))
# z_des = np.matrix([1, 0, 0]).reshape((3, 1))

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
qdot = np.zeros((n, 1))
Kd = np.eye(n) * 50

for i in range(imax):
    if i % 50 == 0 or i == imax - 1:
        sys.stdout.write('\r')
        sys.stdout.write("[%-20s] %d%%" % ('=' * round(20 * i / (imax - 1)), round(100 * i / (imax - 1))))
        sys.stdout.flush()
    q = robot.q.copy()
    jac_eef, htm_eef = robot.jac_geo()

    p_eef = htm_eef[0:3, 3]
    x_eef = htm_eef[0:3, 0]
    y_eef = htm_eef[0:3, 1]
    z_eef = htm_eef[0:3, 2]

    target = np.matrix(np.zeros((3, 1)))
    target[0:3] = vf(p_eef, i*dt)
    # target[3] = -K * np.sqrt(max(1 - x_des.T * x_eef, 0))
    # target[4] = -K * np.sqrt(max(1 - y_des.T * y_eef, 0))
    # target[5] = -K * np.sqrt(max(1 - z_des.T * z_eef, 0))

    jac_target = np.matrix(np.zeros((3, n)))
    jac_target[0:3, :] = jac_eef[0:3, :]
    # jac_target[3, :] = x_des.T * Utils.S(x_eef) * jac_eef[3:6, :]
    # jac_target[4, :] = y_des.T * Utils.S(y_eef) * jac_eef[3:6, :]
    # jac_target[5, :] = z_des.T * Utils.S(z_eef) * jac_eef[3:6, :]

    # qdot_des = Utils.dp_inv(jac_target, 0.002) * target
    qdot_des = np.linalg.pinv(jac_target) @ target
    q_des = q 
    # qddot_des = (qdot_des - qdot) / dt 
    a_des = vf.acceleration(p_eef, jac_target @ qdot, i*dt)
    Jdot = dot_J(robot, qdot, q)[:3, :]
    qddot_des = np.linalg.pinv(jac_target) @ (a_des - Jdot @ qdot)
    M, C, G = robot.dyn_model(q, qdot)
    torque = C + G + (M @ (qddot_des + Kd @ (qdot_des - qdot)) )
    qddot = np.linalg.inv(M) @ (-C -G + torque)
    qddot = Kd @ (qdot_des - qdot) + qddot_des 

    qdot = qdot + qddot * dt
    q = robot.q + qdot * dt

    robot.add_ani_frame(time=i*dt, q=q)
    # traj.add_ani_frame(time=i*dt, initial_ind=0, final_ind=curve.shape[1])
    traj.add_ani_frame(time=i*dt, initial_ind=maxtheta*i, final_ind=maxtheta*(i+1))
    nearest_point.add_ani_frame(time=i*dt, htm=Utils.trn(vf.nearest_points[i]))

    hist_time.append(i * dt)
    hist_q = np.block([hist_q, robot.q])
    hist_peef = np.block([hist_peef, p_eef])
    hist_vf = np.block([hist_vf, target[0:3]])
    # error_ori = np.matrix([(180 / (np.pi)) * np.arccos(1 - min(num * num / (K * K),2)) for num in target[3:6]]).reshape((3,1))
    # hist_error_ori = np.block([hist_error_ori, error_ori])
    hist_qdot = np.block([hist_qdot, qdot])
    hist_qdot_des = np.block([hist_qdot_des, qdot_des])
    hist_qddot = np.block([hist_qddot, qddot])
    hist_qddot_des = np.block([hist_qddot_des, qddot_des])
    hist_cond_J.append(np.linalg.cond(jac_target))
    hist_cond_Jdot.append(np.linalg.cond(Jdot))


sim.run()
hist_vf = np.array(hist_vf)
hist_peef = np.array(hist_peef)
fig = vector_field_plot(hist_peef, hist_vf, add_lineplot=True, sizemode="absolute", sizeref=2.5, anchor='tail')
fig.show()
fig=px.line(np.linalg.norm(hist_qdot-hist_qdot_des, axis=0).T, title='|dq/dt - dq<sub>des</sub>/dt|')
fig.show()
fig=px.line(np.abs(hist_qdot-hist_qdot_des).T, title='abs(dq/dt - dq<sub>des</sub>/dt)')
fig.show()
fig=px.line(np.abs(hist_peef-np.array(vf.nearest_points).reshape(-1, 3).T).T, title='|p<sub>eef</sub> - x*|')
fig.show()
fig=px.line(hist_cond_J, title='J condition number')
fig.show()
fig=px.line(hist_cond_Jdot, title=r'\dot{J} condition number')
fig.show()