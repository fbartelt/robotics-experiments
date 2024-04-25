#%%
from scipy.spatial.transform import Rotation
import numpy as np
from uaibot_addons.vfcomplete import VectorField
from uaibot_addons.math import skew, vee
from create_jaco import create_jaco2
from numpy import cos, sin, sqrt, arccos
from scipy.linalg import logm, expm

def progress_bar(i, imax):
    sys.stdout.write("\r")
    sys.stdout.write(
        "[%-20s] %d%%" % ("=" * round(20 * i / (imax - 1)), round(100 * i / (imax - 1)))
    )
    sys.stdout.flush()

# Parametric equation definition
maxtheta = 1500

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
            1/8*(sin_theta[i] + 2*np.sin(2*theta[i])),
            1/8*(cos_theta[i] - 2*np.cos(2*theta[i])),
            0.4 + 1/8*(-np.sin(3*theta[i]))
        ])
    orientations = np.empty((len(theta), 3, 3))
    for i, ang in enumerate(theta):
        orientations[i, :, :] = Rotation.from_euler('z', 2*ang).as_matrix() @ rotz
        # orientations[i, :, :] = Rotation.from_euler('z', np.pi).as_matrix()
    
    precomputed = ((curve.T, orientations))
    
    def parametric_eq(time):
        return precomputed


    return parametric_eq

eq = parametric_eq_factory(w1=0, w2=0, c1=0.3, c2=0.3, c3=0, h0=0.6, maxtheta=maxtheta, T=2, dt=1e-2, timedependent=False)
# vf = VectorField(eq, False, kf=5, vr=0.8, wr=1, beta=1, dt=1e-2)

#%%
import sys
sys.path.insert(1, "/home/fbartelt/Documents/UFMG/TCC/Sim/uaibot")
from uaibot import robot as rb
from uaibot.utils import Utils
from uaibot.simobjects.ball import Ball
from uaibot.simobjects.box import Box
from uaibot.simobjects.cylinder import Cylinder
from uaibot.simobjects.pointcloud import PointCloud
from uaibot.simobjects.pointlight import PointLight
from uaibot.simobjects.frame import Frame
from uaibot.simulation import Simulation
from uaibot.graphics.meshmaterial import MeshMaterial
import plotly.express as px
import plotly.graph_objects as go

def to_htm(p, R):
    if p is None:
        p = np.zeros((3, 1))
    if R is None:
        R = np.eye(3)
    p = p.ravel()
    htm = np.array([[R[0,0], R[0,1], R[0,2], p[0]],
                    [R[1,0], R[1,1], R[1,2], p[1]],
                    [R[2,0], R[2,1], R[2,2], p[2]],
                    [0, 0, 0, 1]])
    return htm

T = 10
dt = 1e-2
eq = parametric_eq_factory(w1=0, w2=0, c1=0.3, c2=0.3, c3=0, h0=0.6, maxtheta=maxtheta, T=T, dt=dt, timedependent=False)
vf = VectorField(eq, False, kf=5, vr=0.5, wr=2, beta=0.5, dt=1e-3) #kf=5, vr=0.5, beta=1 ;;; vr=1.5 erro menor 

curve = eq(0)
curve_points = curve[0]
curve_ori = curve[1]
p = np.array([0.3, 0.3, 0.1]).reshape(-1, 1)
R = Rotation.from_euler('z', np.deg2rad(45)).as_matrix() @ Rotation.from_euler('x', np.deg2rad(12)).as_matrix()
obj = Ball(htm=Utils.trn(p), radius=0.05, color="#8a2be2")
frame_ball = Frame(to_htm(p, R), 'axis', 0.2, ['magenta', 'green', 'cyan'])
light1 = PointLight(name="light1", color="white",
                        intensity=2.5, htm=Utils.trn([-1, -1, 1.5]))
light2 = PointLight(name="light2", color="white",
                    intensity=2.5, htm=Utils.trn([-1, 1, 1.5]))
light3 = PointLight(name="light3", color="white",
                    intensity=2.5, htm=Utils.trn([1, -1, 1.5]))
light4 = PointLight(name="light4", color="white",
                intensity=2.5, htm=Utils.trn([1, 1, 1.5]))
curve_draw = PointCloud(name="curve", points=curve_points.T, size=8)
curve_frames = []
for i, c in enumerate(zip(curve_points, curve_ori)):
    pos, ori = c
    if i % 20 == 0:
        curve_frames.append(Frame(to_htm(pos, ori), f'curveframe{i}', 0.1))

imax = int(T / dt)
p_hist = [p]
R_hist = [R]
v_hist, w_hist = [np.zeros((3,1))], [np.zeros((3,1))]

for i in range(imax):
    progress_bar(i, imax)
    xi = vf.psi(p, R)
    vd = xi[:3, :]
    wd = xi[3:, :]
    if np.iscomplex(xi).any():
        print('Complex number found')
    p = p + vd * dt
    R = expm(dt * skew(wd)) @ R
    obj.add_ani_frame(i * dt, to_htm(p, None))
    frame_ball.add_ani_frame(i * dt, to_htm(p, R))
    curve_draw.add_ani_frame(i * dt, 0, 500)
    p_hist.append(p)
    R_hist.append(R)
    v_hist.append(vd)
    w_hist.append(wd)

sim = Simulation.create_sim_grid([obj, frame_ball, curve_draw, light1, light2, light3, light4])
sim.add([curve_frames])
sim.set_parameters(width=1200, height=600, ambient_light_intensity=4)
# sim.run()
# points = np.array(p_hist).reshape(-1, 3)
# fig = px.scatter_3d(x=points[:,0], y=points[:,1], z=points[:,2])
# for pos, rot in zip(p_hist, R_hist):
#     px, py, pz = pos
#     ux, uy, uz = rot[:, 0], rot[:, 1], rot[:, 2]
#     fig.add_trace(go.Scatter3d(x=[px, px+ux], y=[py, py+uy], z=[pz, pz+uz], mode='lines'))
# fig.show()
#%%
"""""######################################"""
import plotly.colors as pc
def vector_field_plot(coordinates, field_values, orientations, skip_coord=1, skip_ori=1, add_lineplot=False, **kwargs):
    """Plot a vector field in 3D. The vectors are represented as cones and the
    auxiliary lineplot is used to represent arrow tails. The kwargs are passed
    to the go.Cone function.

    Parameters
    ----------
    coordinates : list or np.array
        Mx3 array of coordinates of the vectors. Each row corresponds to x,y,z
        respectively. The column entries are the respective coordinates.
    field_values : list or np.array
        Mx3 array of field values of the vectors. Each row corresponds to u,v,w
        respectively, i.e. the velocity of the field in each direction.
        The column entries are the respective values.
    add_lineplot : bool, optional
        Whether to add a lineplot of the field coordinates. The default is False.
    """
    coordinates = np.array(coordinates).reshape(-1, 3)
    coord_field = coordinates[::skip_coord].T
    coordinates = coordinates.T
    field_values = np.array(field_values).reshape(-1, 3)[::skip_coord].T
    npoints = coordinates.shape[1]
    plasma_cscale = pc.make_colorscale(pc.sequential.Plasma)

    # Path
    fig = go.Figure(go.Scatter3d(x=coordinates[0, :], y=coordinates[1, :], 
                                 z=coordinates[2, :], mode="lines", line=dict(width=5)))
    
    # Vector field
    fig.add_trace(
        go.Cone(
            x=coord_field[0, :],
            y=coord_field[1, :],
            z=coord_field[2, :],
            u=field_values[0, :],
            v=field_values[1, :],
            w=field_values[2, :],
            # colorscale=[[i / max(index), c[1]] for i, c in zip(index, plasma_cscale)],
            colorscale=[[0, plasma_cscale[1][1]], [1, plasma_cscale[1][1]]],  # Set the colorscale
            showscale=False,
            **kwargs,
        )
    )

    # Orientation frames
    scale_frame = 0.05
    if orientations is not None:
        for i, ori in enumerate(orientations[::skip_ori]):
            px, py, pz = coordinates[:, i*skip_ori]
            ux, uy, uz = scale_frame*(ori[:, 0])
            vx, vy, vz = scale_frame*(ori[:, 1])
            wx, wy, wz = scale_frame*(ori[:, 2])
            fig.add_trace(go.Scatter3d(x=[px, px+ux], y=[py, py+uy], z=[pz, pz+uz], mode='lines', line=dict(color='red')))
            fig.add_trace(go.Scatter3d(x=[px, px+vx], y=[py, py+vy], z=[pz, pz+vz], mode='lines', line=dict(color='green')))
            fig.add_trace(go.Scatter3d(x=[px, px+wx], y=[py, py+wy], z=[pz, pz+wz], mode='lines', line=dict(color='blue'))
            )

    # Object
    i2 = -1
    i3 = 15 * skip_ori
    fig.add_trace(go.Scatter3d(x=[coordinates[0, 0]], y=[coordinates[1, 0]], z=[coordinates[2, 0]], mode="markers", marker=dict(size=10, color='magenta')))
    fig.add_trace(go.Scatter3d(x=[coordinates[0, i2]], y=[coordinates[1, i2]], z=[coordinates[2, i2]], mode="markers", marker=dict(size=10, color='orange')))
    # fig.add_trace(go.Scatter3d(x=[coordinates[0, i3]], y=[coordinates[1, i3]], z=[coordinates[2, i3]], mode="markers", marker=dict(size=15, color='magenta')))

    #  sizemode=sizemode, sizeref=2.5, anchor='tail'))
    if add_lineplot:
        fig.add_scatter3d(
            x=coordinates[0, :], y=coordinates[1, :], z=coordinates[2, :], mode="lines"
        )
    camera = dict(eye=dict(x=-0.3, y=2.2, z=0.5))
    camera = dict(eye=dict(x=.4, y=1.6, z=1.6))
    yticks = [-0.4, -0.2, 0, 0.2, 4]
    zticks = [0.2, 0.4, 0.6]
    fig.update_layout(margin=dict(t=0, b=10, r=0, l=0), scene_camera=camera, 
                      showlegend=False, scene_aspectmode='cube', 
                      scene_yaxis=dict(range=[-0.4, 0.4],   ticks='outside',
                                       tickvals=yticks, ticktext=yticks),
                      scene_zaxis=dict(range=[0, 0.6],   ticks='outside',
                                       tickvals=zticks, ticktext=zticks),
                      width=1080, height=1080)

    return fig

skip_ori = int(len(p_hist) / 12)  #87
skip_coord = int(len(p_hist) / 14) #57
coords = np.array(p_hist).reshape(-1, 3)
vf_values = np.array(v_hist).reshape(-1, 3)
orientations = R_hist
fig = vector_field_plot(coords, vf_values, orientations, skip_coord=skip_coord, 
                  skip_ori=skip_ori, sizemode="scaled", sizeref=.25, anchor='tail')
# fig.show(width=1080, height=1080)
#%%
near_p, near_R = zip(*vf.nearest_points)
near_p = np.array(near_p).reshape(-1, 3)
coords = np.array(p_hist).reshape(-1, 3)
# fig1 = px.line(np.linalg.norm(near_p - coords[1::], axis=1)**2)
# fig1.show()
fro_norms = []
for rot, rot_d in zip(R_hist, near_R):
    fro_norms.append(0.5 * np.linalg.norm(np.eye(3) - rot_d.T @ rot)**2)
fig2 = px.line(x=np.arange(0, T, dt), y=np.array(fro_norms) + np.linalg.norm(near_p - coords[1::], axis=1)**2)
# fig2.update_xaxes(title="Time (s)")
# fig2.update_yaxes(title="Distance Function")
fig2.update_layout(xaxis_title="Time (s)", yaxis_title="Distance Function", width=1200, height=600, margin=dict(t=0, b=0, r=0, l=5),
                   xaxis_title_font=dict(size=18), yaxis_title_font=dict(size=18))
fig2.update_xaxes(tickfont=dict(size=14), tickprefix="\t")
fig2.update_yaxes(tickfont=dict(size=14))
fig2.show()
# %%
import sys
sys.path.insert(1, "/home/fbartelt/Documents/UFMG/TCC/Sim/uaibot")
from uaibot import robot as rb
from uaibot.utils import Utils
from uaibot.simobjects.ball import Ball
from uaibot.simobjects.box import Box
from uaibot.simobjects.cylinder import Cylinder
from uaibot.simobjects.pointcloud import PointCloud
from uaibot.simobjects.frame import Frame
from uaibot.simulation import Simulation
from uaibot.graphics.meshmaterial import MeshMaterial

# def _control_demo_1():
    # Create simulation and add objects to the scene
# robot = rb.Robot.create_abb_crb(Utils.trn([-0.2,0,0.3]), "robo")
robot = rb.Robot.create_abb_crb(name="robo")
n = len(robot.links)

mesh_board = MeshMaterial(roughness=1, metalness=0.9)
board = Box(htm=Utils.trn([0.6, 0, 0.5]), width=0.05, depth=0.9, height=0.8, color="white",
            mesh_material=mesh_board)
material_box = MeshMaterial(color="#242526", roughness=1, metalness=1)
base = Cylinder(htm=Utils.trn([-0.2, 0, 0.15]), radius=0.1, height=0.3, mesh_material = material_box)
sim = Simulation.create_sim_factory([robot, board, base])
frame_aux = Frame(Utils.rotz(np.pi) @ Utils.trn([0, 0, .5]), 'axis', 0.5, ['magenta', 'green', 'cyan'])
sim.add([frame_aux])


# Create curve
# theta = np.linspace(0, 2 * np.pi, num=300)
curve = eq(0)[0]

# Create vector field
# vf = rb.Robot.vector_field(np.matrix(curve.T), 10, 0.3) #TODO

# Parameters
dt = 0.01
time_max = 20
K = 0.03
imax = round(time_max / dt)

# Initializations
hist_time = []
hist_qdot = np.matrix(np.zeros((6,0)))
hist_q = np.matrix(np.zeros((6,0)))
hist_error_ori = np.matrix(np.zeros((3,0)))

# x_des = np.matrix([0, 0, 1]).reshape((3, 1))
# y_des = np.matrix([0, -1, 0]).reshape((3, 1))
# z_des = np.matrix([1, 0, 0]).reshape((3, 1))
x_des = np.matrix([1, 0, 0]).reshape((3, 1))
y_des = np.matrix([0, 1, 0]).reshape((3, 1))
z_des = np.matrix([0, 0, 1]).reshape((3, 1))
Rd = Rotation.from_euler('z', np.pi).as_matrix()

# Main loop
draw_points = np.zeros((3, 0))
reached_board = False

for i in range(imax):
    x_des, y_des, z_des = Rd[:, 0], Rd[:, 1], Rd[:, 2]

    if i % 50 == 0 or i == imax - 1:
        sys.stdout.write('\r')
        sys.stdout.write("[%-20s] %d%%" % ('=' * round(20 * i / (imax - 1)), round(100 * i / (imax - 1))))
        sys.stdout.flush()

    jac_eef, htm_eef = robot.jac_geo()

    p_eef = htm_eef[0:3, 3]
    x_eef = htm_eef[0:3, 0]
    y_eef = htm_eef[0:3, 1]
    z_eef = htm_eef[0:3, 2]

    target = np.real(vf.psi(p_eef, htm_eef[0:3, 0:3])) #TODO
    # target = np.matrix(np.zeros((6,1)))
    # target[0:3] = vf(p_eef)
    # target[3] = -K * sqrt(max(1 - x_des.T * x_eef, 0))
    # target[4] = -K * sqrt(max(1 - y_des.T * y_eef, 0))
    # target[5] = -K * sqrt(max(1 - z_des.T * z_eef, 0))

    jac_target = np.matrix(np.zeros((6, n)))
    jac_target[0:3, :] = jac_eef[0:3, :]
    jac_target[3, :] = x_des.T * Utils.S(x_eef) * jac_eef[3:6, :]
    jac_target[4, :] = y_des.T * Utils.S(y_eef) * jac_eef[3:6, :]
    jac_target[5, :] = z_des.T * Utils.S(z_eef) * jac_eef[3:6, :]

    qdot = Utils.dp_inv(jac_target, 0.002) * target

    q_prox = robot.q + qdot * dt
    Rd = expm(dt * skew(target[3:6])) @ Rd

    robot.add_ani_frame(i * dt, q_prox)

    hist_time.append(i * dt)
    hist_q = np.block([hist_q, robot.q])
    # error_ori = np.array([(180 / (np.pi)) * arccos(1 - min(num.item() * num.item() / (K * K),2)) for num in target[3:6]]).reshape((3,1))
    error_ori = (1 - np.diagonal(Rd.T @ htm_eef[:3, :3])).reshape(-1, 1)
    hist_error_ori = np.block([hist_error_ori, error_ori])
    hist_qdot = np.block([hist_qdot, qdot])

    # See if the end-effector is close to the board to add to the point cloud
    draw_points = np.block([draw_points, p_eef])


    # if (not reached_board) and (abs(p_eef[0,0] - board.htm[0,3]) < board.width / 2 + 0.001):
    #     reached_board = True
    ind_reached = 0

# Set up the cloud of points
point_cloud = PointCloud(name="drawing", points=draw_points, size=0.025)
sim.add(point_cloud)
for i in range(imax):
    if i < ind_reached:
        point_cloud.add_ani_frame(i * dt, 0, 0)
    else:
        point_cloud.add_ani_frame(i * dt, ind_reached, i)

# Run simulation
sim.run()

# Plot graphs
Utils.plot(hist_time, hist_q, "", "Time (s)", "Joint configuration  (rad)", "q")
Utils.plot(hist_time, hist_qdot, "", "Time (s)", "Joint speed (rad/s)", "u")
Utils.plot(hist_time, hist_error_ori, "", "Time (s)", "Orientation error (degrees)", ['x', 'y', 'z'])

    # return sim

# sim = _control_demo_1()
# %%
