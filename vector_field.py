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
from uaibot.robot._vector_field import _compute_ntd as compute_ntd
from cvxopt import matrix, solvers, spmatrix
from create_jaco import create_jaco2
from plotly.subplots import make_subplots
from itertools import product

solvers.options["show_progress"] = False
_INVHALFPI = 0.63660
"""
VectorField acceleration causes RuntimeWarning: invalid value encountered in sqrt
  eta = (-psi_s.T @ psi_t + np.sqrt((psi_s.T @ psi_t)**2 + self.const_vel**2 - psi_t.T @ psi_t))[0, 0]

Probably due to tangent field with norm greater than vr.
"""
class VectorField():
    def __init__(self, parametric_equation, time_dependent, alpha=1, const_vel=1, dt=1e-3):
        self.parametric_equation = parametric_equation
        self.alpha = alpha
        self.const_vel = const_vel
        self.time_dependent = time_dependent
        self.dt = dt
        self.nearest_points = []

    def __call__(self, position, time=0):
        return self.psi(position, time)
    
    def _add_nearest_point(self, point):
        self.nearest_points.append(point)

    def psi(self, position, time=0, store_points=True):
        psi_s = self._psi_s(position, time, store_points=store_points)
        if self.time_dependent:
            psi_s = psi_s / abs(self.const_vel)
            psi_t = self._psi_t(position, time, store_points=False)
            eta = (-psi_s.T @ psi_t + np.sqrt((psi_s.T @ psi_t)**2 + self.const_vel**2 - psi_t.T @ psi_t))[0, 0]
            return eta * psi_s + psi_t
        else:
            return psi_s
    def _psi_s(self, position, time=0, store_points=True):
        #TODO change _add_nearest_point to apply here instead of psi t.
        # this implies copying the uaibot vector field code into here.
        p = np.array(position).reshape(-1, 1)
        curve = np.matrix(self.parametric_equation(time=time))
        # return ub.Robot.vector_field(curve, self.alpha, self.const_vel)(p)
        return self._vector_field_vel(p, curve, self.alpha, self.const_vel, np.shape(curve)[0], store_points=store_points)
    
    def _psi_t(self, position, time, store_points=True):
        #TODO implement time derivative computation of Distance vector. The component PsiT,
        # which is the null space projection of the time derivative.
        p = np.array(position).reshape(-1, 1)
        curve = np.matrix(self.parametric_equation(time=time))
        next_curve = np.matrix(self.parametric_equation(time=time+self.dt))
        min_dist = float('inf')
        min_dist_next = float('inf')
        ind_min = -1
        ind_min_next = -1

        pr = np.matrix(p).reshape((3,1))

        for i in range(np.shape(curve)[1]):
            dist_temp = np.linalg.norm(pr - curve[:,i])
            if dist_temp < min_dist:
                min_dist = dist_temp
                ind_min = i
            
            dist_temp = np.linalg.norm(pr - next_curve[:,i])
            if dist_temp < min_dist_next:
                min_dist_next = dist_temp
                ind_min_next = i

        if ind_min == np.shape(curve)[1] - 1:
            vec_t = (next_curve[:,1] - curve[:,ind_min]) / self.dt
        else:
            vec_t = (next_curve[:,ind_min_next] - curve[:,ind_min]) / self.dt

        if store_points:
            self._add_nearest_point(curve[:,ind_min])
        vec_t = vec_t
        Tstar = compute_ntd(curve, p)[1]
        nullspace = np.eye(p.shape[0]) - Tstar @ Tstar.T
        vec_t = -nullspace @ vec_t

        return vec_t
    
    def _psi_t_wrong(self, position, time):
        #TODO implement time derivative computation of Distance vector. The component PsiT,
        # which is the null space projection of the time derivative.
        p = np.array(position).reshape(-1, 1)
        curve = np.matrix(self.parametric_equation(time=time))
        next_curve = np.matrix(self.parametric_equation(time=time+self.dt))
        min_dist = float('inf')
        ind_min = -1

        pr = np.matrix(p).reshape((3,1))

        for i in range(np.shape(curve)[1]):
            dist_temp = np.linalg.norm(pr - curve[:,i])
            if dist_temp < min_dist:
                min_dist = dist_temp
                ind_min = i

        if ind_min == np.shape(curve)[1] - 1:
            vec_t = (next_curve[:,1] - curve[:,ind_min]) / self.dt
        else:
            vec_t = (next_curve[:,ind_min] - curve[:,ind_min]) / self.dt

        vec_t = vec_t
        Tstar = compute_ntd(curve, p)[1]
        nullspace = np.eye(p.shape[0]) - Tstar @ Tstar.T
        vec_t = -nullspace @ vec_t

        return vec_t

    def acceleration(self, position, velocity, time=0):
        position = np.array(position).reshape(-1, 1)
        velocity = np.array(velocity).reshape(-1, 1)
        current_vf = self.psi(position, time, store_points=False)
        # \partial{vf}/\partial{t}
        dvfdt = (self.psi(position, time+self.dt, store_points=False) - current_vf) / self.dt 
        # \partial{vf}/\partial{x} \dot{x}
        dvfdx = np.array([
                 self.psi(position + np.array([self.dt, 0, 0]).reshape(-1, 1), time, store_points=False) - current_vf,
                 self.psi(position + np.array([0, self.dt, 0]).reshape(-1, 1), time, store_points=False) - current_vf,
                 self.psi(position + np.array([0, 0, self.dt]).reshape(-1, 1), time, store_points=False) - current_vf
                ]).reshape(3, 3).T / self.dt
        a = dvfdx @ velocity + dvfdt
        return a
    
    def _vector_field_vel(self, p, curve, alpha, const_vel, vector_size, store_points=True):
        vec_n, vec_t, min_dist = self._compute_ntd(curve, p, store_points=store_points)
        fun_g = _INVHALFPI * np.arctan(alpha * min_dist)
        fun_h = np.sqrt(max(1 - fun_g ** 2, 0))
        abs_const_vel = abs(const_vel)
        sgn = const_vel / (abs_const_vel + 0.00001)

        return abs_const_vel * (fun_g * vec_n + sgn * fun_h * vec_t)
    
    def _compute_ntd(self, curve, p, store_points=True):
        min_dist = float('inf')
        ind_min = -1

        pr = np.matrix(p).reshape((3,1))

        for i in range(np.shape(curve)[1]):
            dist_temp = np.linalg.norm(pr - curve[:,i])
            if dist_temp < min_dist:
                min_dist = dist_temp
                ind_min = i

        vec_n = curve[:,ind_min] - pr
        vec_n = vec_n / (np.linalg.norm(vec_n) + 0.0001)

        if ind_min == np.shape(curve)[1] - 1:
            vec_t = curve[:,1] - curve[:,ind_min]
        else:
            vec_t = curve[:,ind_min + 1] - curve[:,ind_min]

        vec_t = vec_t / (np.linalg.norm(vec_t) + 0.0001)
        if store_points:
            self._add_nearest_point(curve[:,ind_min])

        return vec_n, vec_t, min_dist

def dot_J(robot, qdot, q=None):
    """Compute the end effetctor jacobian matrix time derivative.
    robot.jac_jac_geo returns a list of the jacobians of the i-th jacobian column
    by the configurations q. The jacobian of the jacobian (jac_jac_geo) is a 
    nx6xn tensor. Since \dot{J} = \frac{\partial J}{\partial q}\dot{q}, we can 
    compute the jacobian time derivative as jac_jac_geo @ qdot.
    """
    if q is None:
        q = robot.q
    jj_geo, *_ = robot.jac_jac_geo(q=q, axis='eef')
    dotJ = np.array(jj_geo) @ np.array(qdot).reshape(-1, 1)
    dotJ = dotJ[:, :, 0].T
    
    return dotJ

def vector_field_plot(coordinates, field_values, add_lineplot=False, **kwargs):
    """ Plot a vector field in 3D. The vectors are represented as cones and the 
    auxiliary lineplot is used to represent arrow tails. The kwargs are passed
    to the go.Cone function.

    Parameters
    ----------
    coordinates : list or np.array
        3xM array of coordinates of the vectors. Each row corresponds to x,y,z 
        respectively. The column entries are the respective coordinates.
    field_values : list or np.array
        3xM array of field values of the vectors. Each row corresponds to u,v,w 
        respectively, i.e. the velocity of the field in each direction. 
        The column entries are the respective values.
    add_lineplot : bool, optional
        Whether to add a lineplot of the field coordinates. The default is False.
    """
    coordinates = np.array(coordinates)
    field_values = np.array(field_values)
    fig = go.Figure(
    go.Cone(x=coordinates[0, :], y=coordinates[1, :], z=coordinates[2, :], 
             u=field_values[0, :], v=field_values[1, :], w=field_values[2, :], 
             **kwargs))
            #  sizemode=sizemode, sizeref=2.5, anchor='tail'))
    if add_lineplot:
        fig.add_scatter3d(x=coordinates[0, :], y=coordinates[1, :], z=coordinates[2, :], mode="lines")

    return fig

def vector_field_animation(coordinates, field_values, add_lineplot=False, **kwargs):
    coordinates = np.array(coordinates)
    fig = vector_field_plot(coordinates, field_values, add_lineplot, **kwargs)
    frames = []
    for i, p in enumerate(coordinates.T):
        frames.append(go.Frame(data=[go.Scatter3d(x=[p[0]], y=[p[1]], z=[p[2]], mode="markers",
                                                marker=dict(color="red", size=5))]))
    fig.add_trace(fig.data[0])
    fig.frames = frames
    fig.update_layout(
         title='Slices in volumetric data',
         updatemenus = [
            {
                "buttons": [
                    {
                        "args": [None, {"frame": {"duration": 0, "redraw": True},
                                "fromcurrent": True, "transition": {"duration": 0,
                                                                    "easing": "quadratic-in-out"}}],
                        "label": "&#9654;", # play symbol
                        "method": "animate",
                    },
                    {
                        "args": [[None], {"frame": {"duration": 0, "redraw": False},
                                  "mode": "immediate",
                                  "transition": {"duration": 0}}],
                        "label": "&#9616;&#9616;", # pause symbol
                        "method": "animate",
                    },
                ],
                "direction": "left",
                "pad": {"r": 10, "t": 0},
                "type": "buttons",
                "x": 0.07,
                "y": -0.1,
            }
         ], 
         margin=dict(l=10, r=10, b=10, t=50),
        #  uirevision=True
)
    return fig

def curve_animation(parametric_equation, time_range, dt, **kwargs):
    t0, tf = time_range
    time = np.arange(t0, tf, dt)
    fig = go.Figure(data=[go.Scatter3d(x=[], y=[], z=[],
                     mode="markers",marker=dict(color="red", size=10))])
    frames = []
    for t in time:
        p = parametric_equation(time=t)
        frames.append(go.Frame(data=[go.Scatter3d(x=p[0, :], y=p[1, :], z=p[2, :], mode="markers")]))

    fig.update_layout(
         title='Slices in volumetric data',
         updatemenus = [
            {
                "buttons": [
                    {
                        "args": [None, {"frame": {"duration": 0, "redraw": True},
                                "fromcurrent": True, "transition": {"duration": 0,
                                                                    "easing": "quadratic-in-out"}}],
                        "label": "&#9654;", # play symbol
                        "method": "animate",
                    },
                    {
                        "args": [[None], {"frame": {"duration": 0, "redraw": False},
                                  "mode": "immediate",
                                  "transition": {"duration": 0}}],
                        "label": "&#9616;&#9616;", # pause symbol
                        "method": "animate",
                    },
                ],
                "direction": "left",
                "pad": {"r": 10, "t": 0},
                "type": "buttons",
                "x": 0.07,
                "y": -0.1,
            }
         ], 
         margin=dict(l=10, r=10, b=10, t=50),
        #  uirevision=True
    )
    
    fig.update(frames=frames)
    return fig


def quiver_plot3d(vector_field, x_range, y_range, z_range, N):
    x = np.linspace(x_range[0], x_range[1], N)
    y = np.linspace(y_range[0], y_range[1], N)
    z = np.linspace(z_range[0], z_range[1], N)
    # fig = make_subplots(rows=1, cols=3)

    vf, xs, ys = [], [], []
    for xy in product(x, y):
        xs.append(xy[0])
        ys.append(xy[1])
        coords = np.array([xy[0], xy[1], 0])
        vf.append(vector_field(coords).T)
    vf = np.array(vf).reshape(-1, 3)
    fig1 = ff.create_quiver(xs, ys, vf[:, 0], vf[:, 1])

    vf, xs, zs = [], [], []
    for xz in product(x, z):
        xs.append(xz[0])
        zs.append(xz[1])
        coords = np.array([xz[0], 0, xz[1]])
        vf.append(vector_field(coords).T)
    vf = np.array(vf).reshape(-1, 3)
    fig2 = ff.create_quiver(xs, zs, vf[:, 0], vf[:, 2])

    vf, ys, zs = [], [], []
    for yz in product(y, z):
        ys.append(yz[0])
        zs.append(yz[1])
        coords = np.array([0 ,yz[0], yz[1]])
        vf.append(vector_field(coords).T)
    vf = np.array(vf).reshape(-1, 3)
    fig3 = ff.create_quiver(ys, zs, vf[:, 1], vf[:, 2])

    subplots = make_subplots(rows=1, cols=3)
    fig1, fig2, fig3 = fig

    for d in fig1.data:
        subplots.add_trace(go.Scatter(x=d['x'], y=d['y']), row=1, col=1)

    for d in fig2.data:
        subplots.add_trace(go.Scatter(x=d['x'], y=d['y']), row=1, col=2)

    for d in fig3.data:
        subplots.add_trace(go.Scatter(x=d['x'], y=d['y']), row=1, col=3)

    return subplots

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
""" KUKA TEST KINEMATIC CONTROL """
T = 10
dt = 0.01
imax = int(T/dt)
K = 1

robot = ub.Robot.create_kuka_kr5(name="jacotest")
sim = ub.Simulation.create_sim_grid([robot])
n = len(robot.links)
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
# fig = go.Figure(
#      go.Cone(x=hist_peef[0, :], y=hist_peef[1, :], z=hist_peef[2, :], 
#              u=hist_vf[0, :], v=hist_vf[1, :], w=hist_vf[2, :], 
#              sizemode="absolute", sizeref=2.5, anchor='tail'))
# fig.add_scatter3d(x=hist_peef[0, :], y=hist_peef[1, :], z=hist_peef[2, :], mode="lines")
fig = vector_field_plot(hist_peef, hist_vf, add_lineplot=True, sizemode="absolute", sizeref=2.5, anchor='tail')
fig.show()
# %%
""" JACO TEST KINEMATIC CONTROL """
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
""" JACO TEST COMPUTED TORQUE CONTROL """
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
Kp, Kd = np.diag([2]*n), np.diag([10]*n)

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

qdot = np.zeros((n, 1))
hist_time = []
hist_qdot = np.matrix(np.zeros((6,0)))
hist_q = np.matrix(np.zeros((6,0)))
hist_error_ori = np.matrix(np.zeros((3,0)))
hist_peef = np.zeros((3, 0))
hist_vf = np.zeros((3, 0))
for i in range(imax):
    q = robot.q
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

    qdot_des = Utils.dp_inv(jac_target, 0.002) * target
    q_des = q 
    qddot_des = (qdot_des - qdot) / dt
    M, C, G = robot.dyn_model(q, qdot)
    torque = C + G + (M @ (qddot_des + Kp @ (q_des - q) * 0 + Kd @ (qdot_des - qdot)) )
    qddot = np.linalg.inv(M) @ (-C -G + torque)

    qdot = qdot + qddot * dt
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
fig = vector_field_plot(hist_peef, hist_vf, add_lineplot=True, sizemode="absolute", sizeref=2.5, anchor='tail', colorscale=[[0, px.colors.qualitative.Plotly[0]], [1, px.colors.qualitative.Plotly[0]]], showscale=False)
fig.show()
# %%
# %%
""" JACO TEST TASK SPACE COMPUTED TORQUE """
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
Kp, Kd = np.diag([2]*n), np.diag([10]*n)

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

qdot = np.zeros((n, 1))
hist_time = []
hist_qdot = np.matrix(np.zeros((6,0)))
hist_q = np.matrix(np.zeros((6,0)))
hist_error_ori = np.matrix(np.zeros((3,0)))
hist_peef = np.zeros((3, 0))
hist_vf = np.zeros((3, 0))
for i in range(imax):
    q = robot.q
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

    qdot_des = Utils.dp_inv(jac_target, 0.002) * target
    q_des = q 
    qddot_des = (qdot_des - qdot) / dt
    M, C, G = robot.dyn_model(q, qdot)
    torque = C + G + (M @ (qddot_des + Kp @ (q_des - q) * 0 + Kd @ (qdot_des - qdot)) )
    qddot = np.linalg.inv(M) @ (-C -G + torque)

    qdot = qdot + qddot * dt
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
fig = vector_field_plot(hist_peef, hist_vf, add_lineplot=True, sizemode="absolute", sizeref=2.5, anchor='tail', colorscale=[[0, px.colors.qualitative.Plotly[0]], [1, px.colors.qualitative.Plotly[0]]], showscale=False)
fig.show()

#%%
def eq1(time=0):
    theta = np.linspace(0, 2 * np.pi, num=300)
    curve = np.matrix(np.zeros(  (3, len(theta))))
    for i in range(len(theta)):
        t = theta[i]
        # curve[:,i] = np.matrix([ [0.56], [0.2 * np.cos(t)], [0.2 * np.sin(t) + 0.5]])
        curve[:,i] = np.matrix([ [0.5/3*(np.sin(t) + 2*np.sin(2*t))], [0.5/3*(np.cos(t) - 2*np.cos(2*t))], [0.45*(-np.sin(3*t)) + 0.45]])

    return curve

def eq2(time=0):
    w1, w2, c1, c2, c3, h0 = 0.05, 0.025, 5, 5, 3.5, 7
    rotz = np.matrix([[np.cos(w1*time), -np.sin(w1*time), 0],
                      [np.sin(w1*time), np.cos(w1*time), 0],
                      [0, 0, 1]])
    theta = np.linspace(0, 2 * np.pi, num=300)
    curve = np.array([rotz @ np.array([c1*np.cos(s), c2*np.sin(s), h0 + c3*np.cos(w2*time)*np.cos(s)**2]).reshape(-1, 1) for s in theta]).reshape(3, -1)
    curve = curve.reshape(-1, 3).T

    return curve

vf1 = VectorField(eq1, False)
vf2 = VectorField(eq2, True)
# %%
""" JACO TEST TASK SPACE COMPUTED TORQUE TRAJECTORY"""
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
Kp, Kd = np.diag([2]*n), np.diag([10]*n)

def eq2_(time=0):
    w1, w2, c1, c2, c3, h0 = 0.005, 0.0025, 0.6, 0.6, 0.3, 0.3
    rotz = np.matrix([[np.cos(w1*time), -np.sin(w1*time), 0],
                      [np.sin(w1*time), np.cos(w1*time), 0],
                      [0, 0, 1]])
    theta = np.linspace(0, 2 * np.pi, num=300)
    curve = np.array([rotz @ np.array([c1*np.cos(s), c2*np.sin(s), h0 + c3*np.cos(w2*time)*np.cos(s)**2]).reshape(-1, 1) for s in theta]).reshape(3, -1)
    curve = curve.reshape(-1, 3).T

    return curve

def eq2__(time=0):
    w1, w2, c1, c2, c3, h0 = 0, 0, 0.6, 0.6, 0.3, 0.3
    rotz = np.matrix([[np.cos(w1*time), -np.sin(w1*time), 0],
                      [np.sin(w1*time), np.cos(w1*time), 0],
                      [0, 0, 1]])
    theta = np.linspace(0, 2 * np.pi, num=300)
    curve = np.array([rotz @ np.array([c1*np.cos(s), c2*np.sin(s), h0 + c3*np.cos(w2*time)*np.cos(s)**2]).reshape(-1, 1) for s in theta]).reshape(3, -1)
    curve = curve.reshape(-1, 3).T

    return curve

def eq2(time=0):
    theta = np.linspace(0, 2 * np.pi, num=300)
    curve = np.matrix(np.zeros(  (3, len(theta))))
    for i in range(len(theta)):
        t = theta[i]
        # curve[:,i] = np.matrix([ [0.56], [0.2 * np.cos(t)], [0.2 * np.sin(t) + 0.5]])
        curve[:,i] = np.matrix([ [0.5/3*(np.sin(t) + 2*np.sin(2*t))], [0.5/3*(np.cos(t) - 2*np.cos(2*t))], [0.45*(-np.sin(3*t)) + 0.45]])

    return curve

maxtheta = 300

a = None
for i in range(imax):
    curve = eq2(i*dt)
    if a is None:
        a = curve
    else:
        a = np.hstack((a, curve))
traj = PointCloud(name='traj', points=a, size=12, color='cyan')
sim.add([traj])
vf = VectorField(eq2, False, alpha=1, const_vel=1)

# Desired axis
x_des = np.matrix([0, 0, 1]).reshape((3, 1))
y_des = np.matrix([0, 1, 0]).reshape((3, 1))
z_des = np.matrix([1, 0, 0]).reshape((3, 1))

qdot = np.zeros((n, 1))
hist_time = []
hist_qdot = np.matrix(np.zeros((6,0)))
hist_qdot_des = np.matrix(np.zeros((6,0)))
hist_q = np.matrix(np.zeros((6,0)))
hist_error_ori = np.matrix(np.zeros((3,0)))
hist_peef = np.zeros((3, 0))
hist_vf = np.zeros((3, 0))
for i in range(imax):
    q = robot.q
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
    a_des = np.nan_to_num(vf.acceleration(p_eef, jac_target @ qdot, i*dt))
    Jdot = dot_J(robot, qdot, q)[:3, :]
    qddot_des = np.linalg.pinv(jac_target) @ (a_des - Jdot @ qdot)
    M, C, G = robot.dyn_model(q, qdot)
    torque = C + G + (M @ (qddot_des + Kp @ (q_des - q) * 0 + Kd @ (qdot_des - qdot)) )
    qddot = np.linalg.inv(M) @ (-C -G + torque)
    

    qdot = qdot + qddot * dt
    q = robot.q + qdot * dt
    robot.add_ani_frame(time=i*dt, q=q)
    traj.add_ani_frame(time=i*dt, initial_ind=maxtheta*i, final_ind=maxtheta*(i+1))

    hist_time.append(i * dt)
    hist_q = np.block([hist_q, robot.q])
    hist_peef = np.block([hist_peef, p_eef])
    hist_vf = np.block([hist_vf, target[0:3]])
    # error_ori = np.matrix([(180 / (np.pi)) * np.arccos(1 - min(num * num / (K * K),2)) for num in target[3:6]]).reshape((3,1))
    # hist_error_ori = np.block([hist_error_ori, error_ori])
    hist_qdot = np.block([hist_qdot, qdot])
    hist_qdot_des = np.block([hist_qdot_des, qdot_des])


sim.run()
hist_vf = np.array(hist_vf)
hist_peef = np.array(hist_peef)
fig = vector_field_plot(hist_peef, hist_vf, add_lineplot=True, sizemode="absolute", sizeref=2.5, anchor='tail', colorscale=[[0, px.colors.qualitative.Plotly[0]], [1, px.colors.qualitative.Plotly[0]]], showscale=False)
fig.show()

fig=px.line(np.linalg.norm(hist_qdot-hist_qdot_des, axis=0).T, title='|dq/dt - dq<sub>des</sub>/dt|')
fig.show()
fig=px.line(np.abs(hist_qdot-hist_qdot_des).T, title='abs(dq/dt - dq<sub>des</sub>/dt)')
fig.show()
fig=px.line(np.abs(hist_peef-np.array(vf.nearest_points).reshape(-1, 3).T).T, title='|p<sub>eef</sub> - x*|')
fig.show()


# %%
""" VECTOR FIELD CLASS TEST """
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
maxtheta = 300

def eq2(time=0):
    ## 0.5e-1, 0.025e-1, 0.6, 0.6, 0.3, 0.3 works with acceleration approx
    ## 0.5e-1, 0.025e-1, 0.6, 0.6, 0.3, 0.3 works with analytic acceleration, Kd=50, const_vel=1, alpha=5
    ## 0.5e-1, 0.025e1, 0.6, 0.6, 0.3, 0.3 works with analytic acceleration, Kd=50, const_vel=1, alpha=5
    ## Above doesnt work with alpha=1
    w1, w2, c1, c2, c3, h0 = 0.5e-1, 0.025e1, 0.6, 0.6, 0.3, 0.3
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
vf = VectorField(eq2, True, alpha=5, const_vel=1)
# Desired axis
# x_des = np.matrix([0, 0, 1]).reshape((3, 1))
# y_des = np.matrix([0, 1, 0]).reshape((3, 1))
# z_des = np.matrix([1, 0, 0]).reshape((3, 1))

hist_time = []
hist_qdot = np.matrix(np.zeros((6,0)))
hist_qdot_des = np.matrix(np.zeros((6,0)))
hist_q = np.matrix(np.zeros((6,0)))
hist_error_ori = np.matrix(np.zeros((3,0)))
hist_peef = np.zeros((3, 0))
hist_vf = np.zeros((3, 0))
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
    a_des = np.nan_to_num(vf.acceleration(p_eef, jac_target @ qdot, i*dt))
    Jdot = np.nan_to_num(dot_J(robot, qdot, q)[:3, :])
    qddot_des = np.nan_to_num(np.linalg.pinv(jac_target) @ (a_des - Jdot @ qdot))
    M, C, G = robot.dyn_model(q, qdot)
    torque = C + G + (M @ (qddot_des + Kd @ (qdot_des - qdot)) )
    qddot = np.linalg.inv(M) @ (-C -G + torque)
    qddot = Kd @ (qdot_des - qdot) + qddot_des 

    qdot = qdot + qddot * dt
    q = robot.q + qdot * dt

    robot.add_ani_frame(time=i*dt, q=q)
    # traj.add_ani_frame(time=i*dt, initial_ind=0, final_ind=curve.shape[1])
    traj.add_ani_frame(time=i*dt, initial_ind=maxtheta*i, final_ind=maxtheta*(i+1))

    hist_time.append(i * dt)
    hist_q = np.block([hist_q, robot.q])
    hist_peef = np.block([hist_peef, p_eef])
    hist_vf = np.block([hist_vf, target[0:3]])
    # error_ori = np.matrix([(180 / (np.pi)) * np.arccos(1 - min(num * num / (K * K),2)) for num in target[3:6]]).reshape((3,1))
    # hist_error_ori = np.block([hist_error_ori, error_ori])
    hist_qdot = np.block([hist_qdot, qdot])
    hist_qdot_des = np.block([hist_qdot_des, qdot_des])


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

# %%
lista = (hist_qdot_des - hist_qdot) / dt

for i, qddot_ in enumerate(hist_qdot.T):
    qddot_approx = (hist_qdot_des[:, i] - qddot_.reshape(-1, 1)) / dt
    j, *_ = robot.jac_geo(q=hist_q[:, i])
    j = j[:3, :]
    a = vf.acceleration(hist_peef[:, i], j @ qddot_.reshape(-1, 1), i*dt)
    dotJ = dot_J(robot, qddot_.reshape(-1, 1), hist_q[:, i])[:3, :]
    qddot = np.linalg.pinv(j) @ (a - dotJ @ qddot_.reshape(-1, 1))


