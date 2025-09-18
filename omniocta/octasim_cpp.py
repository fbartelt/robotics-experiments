import uaibot as ub
import numpy as np
import uaibot_cpp_bind as uaibot_cpp
from uaibot_cpp_bind import expSO3, SmapSO3, SmapSE3, expSE3
import plotly.graph_objects as go
import plotly.colors as pc
from uaibot.robot import Robot
from uaibot.utils import Utils
from scipy.linalg import block_diag
from plotly.subplots import make_subplots

def pose2htm(p, R):
    """Homogeneous transformation matrix from position and rotation."""
    p = np.array(p)
    htm = np.eye(4)
    htm[0:3, 0:3] = R
    htm[0:3, 3] = p.ravel()
    return htm


def vector_field_plot(
    coordinates,
    field_values,
    orientations,
    curve,
    num_arrows=10,
    init_ball=0,
    final_ball=None,
    num_balls=10,
    add_lineplot=False,
    colorscale=None,
    show_curve=True,
    ball_size=5,
    curve_width=2,
    path_width=5,
    frame_scale=0.05,
    frame_width=2,
    curr_path_style="solid",
    prev_path_style="dash",
    **kwargs,
):
    """Plot a vector field in 3D. The vectors are represented as cones and the
    auxiliary lineplot is used to represent arrow tails. The kwargs are passed
    to the go.Cone function. Also plots the target curve, and the path of the
    object. The object is represented as a sphere. The orientations are represented
    as frames with the x, y and z axis of the frame.

    Parameters
    ----------
    coordinates : list or np.array
        Mx3 array of coordinates of the vectors. Each row corresponds to x,y,z
        respectively. The column entries are the respective coordinates.
    field_values : list or np.array
        Mx3 array of field values of the vectors. Each row corresponds to u,v,w
        respectively, i.e. the LINEAR velocity of the field in each direction.
        The column entries are the respective values.
    orientations : list or np.array
        Mx3x3 array of orientations of the object. Each row corresponds to the
        orientation of the object at that point. The 'column' entries are the
        respective 3x3 rotation matrices.
    curve : np.array
        Nx3 array of the curve points. Each row corresponds to x,y,z respectively.
    num_arrows : int, optional
        Number of vector field arrows (cones) to plot. The default is 10.
    init_ball : int, optional
        Initial ball index to plot. The default is 0.
    final_ball : int, optional
        Final ball index to plot. The default is None, which plots until the end.
    num_balls : int, optional
        Number of balls to plot. The default is 10.
    add_lineplot : bool, optional
        Whether to add a lineplot of the field coordinates. The default is False.
        This is used to connect the vector field arrows.
    colorscale : list, optional
        List of colors to use in the plot. The default is None, which uses the
        Plotly default colors. The list must have at least 6 colors, which are
        used for the curve, previous path, current path, initial ball, final ball
        and the object, respectively.
    show_curve : bool, optional
        Whether to show the target curve. The default is True.
    ball_size : int, optional
        Size of the object balls. The default is 5.
    curve_width : int, optional
        Width of the curve line. The default is 2.
    path_width : int, optional
        Width of the path line. The default is 5.
    frame_scale : float or list, optional
        Scale factor for the orientation frames. The default is 0.05. If a list
        is given, the scale factor is applied to each axis of the frame.
    frame_width : int, optional
        Width of the orientation frame lines. The default is 2.
    curr_path_style : str, optional
        Style of the current path line. The default is "solid".
    prev_path_style : str, optional
        Style of the previous path line. The default is "dash".
    **kwargs
        Additional keyword arguments to pass to the go.Cone function.

    Returns
    -------
    fig : plotly.graph_objects.Figure
        Resulting plotly figure.
    """
    if final_ball is None:
        final_ball = len(coordinates) - 1

    if isinstance(frame_scale, (int, float)):
        frame_scale = [frame_scale] * 3

    coordinates = np.array(coordinates).reshape(-1, 3)
    arrows_idx = np.round(np.linspace(0, len(coordinates) - 1, num_arrows)).astype(int)
    coord_field = coordinates[arrows_idx].T
    field_values = np.array(field_values).reshape(-1, 3)[arrows_idx].T
    ball_idx = np.round(np.linspace(init_ball, final_ball, num_balls)).astype(int)
    coord_balls = coordinates[ball_idx]
    ori_balls = np.array(orientations)[ball_idx]
    coordinates = coordinates.T

    if colorscale is None:
        colorscale = pc.qualitative.Plotly

    if isinstance(curve, tuple):
        curve = curve[0]

    fig = go.Figure()

    # Curve
    if show_curve:
        fig.add_trace(
            go.Scatter3d(
                x=curve[:, 0],
                y=curve[:, 1],
                z=curve[:, 2],
                mode="lines",
                line=dict(width=curve_width, color=colorscale[1]),
            )
        )
    # Previous path
    if init_ball > 0:
        fig.add_trace(
            (
                go.Scatter3d(
                    x=coordinates[0, 0:init_ball],
                    y=coordinates[1, 0:init_ball],
                    z=coordinates[2, 0:init_ball],
                    mode="lines",
                    line=dict(
                        width=path_width, dash=prev_path_style, color=colorscale[5]
                    ),
                )
            )
        )

    # Current path
    fig.add_trace(
        go.Scatter3d(
            x=coordinates[0, init_ball:final_ball],
            y=coordinates[1, init_ball:final_ball],
            z=coordinates[2, init_ball:final_ball],
            mode="lines",
            line=dict(width=path_width, dash=curr_path_style, color=colorscale[0]),
        )
    )

    # Vector field arrows
    fig.add_trace(
        go.Cone(
            x=coord_field[0, :],
            y=coord_field[1, :],
            z=coord_field[2, :],
            u=field_values[0, :],
            v=field_values[1, :],
            w=field_values[2, :],
            colorscale=[[0, colorscale[5]], [1, colorscale[5]]],  # Set the colorscale
            showscale=False,
            **kwargs,
        )
    )

    # Orientation frames
    if orientations is not None:
        for i, ori in enumerate(ori_balls):
            px, py, pz = coord_balls[i, :]
            ux, uy, uz = ori[:, 0] / (np.linalg.norm(ori[:, 0] + 1e-6)) * frame_scale
            vx, vy, vz = ori[:, 1] / (np.linalg.norm(ori[:, 1] + 1e-6)) * frame_scale
            wx, wy, wz = ori[:, 2] / (np.linalg.norm(ori[:, 2] + 1e-6)) * frame_scale
            fig.add_trace(
                go.Scatter3d(
                    x=[px, px + ux],
                    y=[py, py + uy],
                    z=[pz, pz + uz],
                    mode="lines",
                    line=dict(color="red", width=frame_width),
                )
            )
            fig.add_trace(
                go.Scatter3d(
                    x=[px, px + vx],
                    y=[py, py + vy],
                    z=[pz, pz + vz],
                    mode="lines",
                    line=dict(color="lime", width=frame_width),
                )
            )
            fig.add_trace(
                go.Scatter3d(
                    x=[px, px + wx],
                    y=[py, py + wy],
                    z=[pz, pz + wz],
                    mode="lines",
                    line=dict(color="blue", width=frame_width),
                )
            )

    # Object
    for i, coord in enumerate(coord_balls):
        if i == 0:
            color = colorscale[3]
        elif i == len(coord_balls) - 1:
            color = colorscale[4]
        else:
            color = "rgba(172, 99, 250, 0.6)"
        fig.add_trace(
            go.Scatter3d(
                x=[coord[0]],
                y=[coord[1]],
                z=[coord[2]],
                mode="markers",
                marker=dict(size=ball_size, color=color),
            )
        )

    if add_lineplot:
        fig.add_scatter3d(
            x=coordinates[0, :], y=coordinates[1, :], z=coordinates[2, :], mode="lines"
        )

    return fig


def hd(s, r=1, b=1, d=0.2):
    """Curve parametrization used in paper. This is based on the hyperbolic
    paraboloid.

    Parameters
    ----------
    s : float
        Parameter of the curve. It must be in the interval [0, 1].
    r : float, optional
        Radius of the curve in XY plane. The default is 1.
    b : float, optional
        Height of the curve. The default is 1.
    d : float, optional
        Curvature of the curve. The default is 0.2.

    Returns
    -------
    hds : np.array
        Homogeneous transformation matrix of the curve evaluated at parameter s.
        This is a 'list' of elements of the SE(3) group.
    """
    theta = 2 * np.pi * s
    hds = np.identity(4)  # initialize the homogeneous transformation matrix
    position = [
        r * np.cos(theta),
        r * np.sin(theta),
        b + d * r**2 * (np.cos(theta) ** 2 - np.sin(theta) ** 2),
    ]
    hds[:3, 3] = np.array(position)
    angle = np.pi/6 * np.sin(2*np.pi * s)
    # angle = theta
    orientation = np.array(
        [
            [1, 0, 0],
            [0, np.cos(angle), np.sin(angle)],
            [0, -np.sin(angle), np.cos(angle)],
        ]
    )
    # axis = np.array([1, 1, 1])
    # axis = axis / np.linalg.norm(axis)
    # skew_mat = SmapSO3(axis)
    # orientation = expSO3(theta * skew_mat)
    # orientation = np.eye(3)
    hds[:3, :3] = orientation
    return hds


def hd_derivative(s, r=1, b=1, d=0.2):
    theta = 2 * np.pi * s
    dhds = np.zeros((4, 4))
    dposition_ds = [
        -r * 2 * np.pi * np.sin(theta),
        r * 2 * np.pi * np.cos(theta),
        d
        * r**2
        * 2
        * (-2 * np.cos(theta) * np.sin(theta) - 2 * np.sin(theta) * np.cos(theta))
        * 2
        * np.pi,
    ]
    dhds[:3, 3] = np.array(dposition_ds)
    angle = np.pi/6 * np.sin(2*np.pi * s)
    # angle = theta
    orientation = np.array(
        [
            [1, 0, 0],
            [0, np.cos(angle), np.sin(angle)],
            [0, -np.sin(angle), np.cos(angle)],
        ]
    )
    chain = np.pi/6 * 2 * np.pi * np.cos(2 * np.pi * s)
    # chain = 2 * np.pi
    dorientation_ds = chain * SmapSO3(np.array([1, 0, 0]))  @ orientation
    # axis = np.array([1, 1, 1])
    # axis = axis / np.linalg.norm(axis)
    # dorientation_ds = 2 * np.pi * SmapSO3(axis * theta)
    dhds[:3, :3] = dorientation_ds
 
    # dhds[:3, :3] = 2 * np.pi * np.array(
    #     [
    #         [0, 0, 0],
    #         [0, -np.sin(theta), np.cos(theta)],
    #         [0, -np.cos(theta), -np.sin(theta)],
    #     ]
    # )
    return dhds



def precomputed_hd(curve_fun, n_points, *args, **kwargs):
    """Function that precomputes the curve for each parameter s.

    Parameters
    ----------
    curve_fun : function
        Function that computes the curve. It must be a function that takes as
        first argument the parameter s, and returns a homogeneous transformation
        matrix.
    n_points : int
        Number of points in the curve.
    *args : list
        Arguments of the curve function.
    **kwargs : dict
        Keyword arguments of the curve function.

    Returns
    -------
    precomputed : np.array
        Array with the precomputed curve. The shape is (n_points, 4, 4).
    """
    s = np.linspace(0, 1, num=n_points)
    precomputed = []
    for si in s:
        precomputed.append(curve_fun(si, *args, **kwargs))
    precomputed = np.array(precomputed)
    return precomputed
m = 3.2  # mass (Kg)
M_body = 2 * 0.015 * np.eye(3)  # inertia (Kg*m^2)

w_min = 00  # [Hz]
w_max = 240  # [Hz]

# Design parameters
n = 8

k_f = 9.9016 * 10 ** (-4)  # N/RPM^2
k_d = 1.9 * 10 ** (-5)  # Nm/RPM^2
u_max = w_max**2
r = k_d / k_f  # lift force/drag moment

A = k_f * np.array(
    [
        [0.4274, 0.8596, -0.2799, -0.0000, 0.1380, 0.3770],
        [0.7600, -0.5309, 0.3748, 0.2352, 0.1504, -0.3716],
        [-0.7619, -0.3881, 0.5186, 0.2261, 0.0467, 0.3341],
        [-0.4622, -0.2558, -0.8491, -0.2267, -0.2754, 0.2745],
        [-0.0491, 0.9877, -0.1483, -0.0373, -0.0793, -0.3914],
        [-0.7823, 0.1134, 0.6124, -0.2054, -0.1104, -0.2652],
        [0.8232, -0.2046, 0.5296, -0.2187, -0.0978, 0.3444],
        [-0.0048, -0.4788, -0.8779, 0.3009, 0.3472, -0.2571],
    ]
)
A = A.T
motor_tau = 0.1  # time constant of the motor dynamics

# Simulation parameters
dt = 1e-3
dt = 10 * dt
# dt = 0.0025
T = 200
imax = int(T / dt)


# Initial conditions
n_points = 5000
r, b, d = 2.5, 1, 0.2
curve = precomputed_hd(hd, n_points, r, b, d)
curve_derivative = precomputed_hd(hd_derivative, n_points, r, b, d)
R0 = np.eye(3)
# p0 = np.array([2, -2, 0]).reshape(-1, 1)
p0 = np.array([1, 1, 0]).reshape(-1, 1)
# p0 = curve[0, :3, 3].reshape(-1, 1)  # Start at beginning of curve
p = p0.copy()
R0 = curve[0, :3, :3]  # Start with orientation of curve
R = R0.copy()
v = np.array([0, 0, 0]).reshape(-1, 1)
v_dot = np.array([0, 0, 0]).reshape(-1, 1)
omega = np.array([0, 0, 0]).reshape(-1, 1)
omega_dot = np.array([0, 0, 0]).reshape(-1, 1)
u = np.zeros((n, 1)).reshape(-1, 1)

kn1, kn2 = 1 * 0.2, 10
kt1, kt2, kt3 = kn1 * 1, 1, kn2

#
# state = uaibot_cpp.DroneState()
state = uaibot_cpp.CPP_DroneState()
state.p = p
state.Q = R
state.v = v
state.omega = omega
state.u = u

param = uaibot_cpp.CPP_ParametersSim()
param.A = A
param.pinv_A = np.linalg.pinv(A)
param.M = m
param.J = M_body[0, 0]
param.u_min = 0.0
param.u_max = u_max
param.tc = motor_tau
param.dt = dt
param.sim_time = T
# Vector field parameters
param.kt1 = kt1
param.kt2 = kt2
param.kt3 = kt3
param.kn1 = kn1
param.kn2 = kn2
param.delta = 1e-3
param.ds = 1e-3
# PID gains
# Current best: 10, 10
param.kv = 10
param.komega = 10
# Noise parameters (p, Q, v, omega, u)
param.stds = np.array([0.1, 0.01, 0.001*0, 0.02*0, 0*0.00001]).reshape(-1, 1) * 0

log_full = uaibot_cpp.vant_simulation(
    state, curve, curve_derivative, param
)

# log = log_full
steady_index = log_full[-1].steady_index
log = log_full[steady_index:]
print(steady_index)
print(len(log))
p_hist = np.array([np.array(z.p).reshape(-1, 1) for z in log]).reshape(-1, 3)
R_hist = np.array([np.array(z.Q) for z in log]).reshape(-1, 3, 3)
v_hist = [np.array(z.v).reshape(-1, 1) for z in log]
u_hist = [np.array(z.u).reshape(-1, 1) for z in log]
dist_hist = np.array([z.distance for z in log]).reshape(-1, 1)
nearest_indexes = [z.nearest_index for z in log]
nearest_htms = [curve[i] for i in nearest_indexes]

# print("AAAA")
# print(A @ np.linalg.pinv(A))
# print(A @ np.array([-0.00000,  2.32703,  3.21962,  0.00000,  0.00000,  3.80230,  3.28829,  0.00000]).reshape(-1,1))
# print(A)
#
print(u_max)

def nvim_plot():
    final_index = int(len(nearest_indexes)) - 1
    fig = vector_field_plot(
        p_hist,
        v_hist,
        R_hist,
        curve[:, :3, 3],
        num_arrows=20,
        init_ball=0,
        final_ball=final_index,
        num_balls=20,
        add_lineplot=False,
        colorscale=None,
        show_curve=True,
        ball_size=3,
        curve_width=2,
        path_width=5,
        frame_scale=0.05,
        frame_width=2,
        curr_path_style="solid",
        prev_path_style="dash",
        sizemode="absolute",
        sizeref=3e-2,
        anchor="tail",
    )
    fig.update_layout(showlegend=False)
    fig.show()

    # Plot every component of the control input u
    fig = go.Figure()
    time_vec = np.arange(0, len(u_hist) * dt, dt)
    for i in range(n):
        fig.add_trace(
            go.Scatter(
                x=time_vec,
                y=[u_hist[j][i, 0] for j in range(len(u_hist))],
                name=f"u_{i+1}",
                line=dict(width=3),
            )
        )
    fig.update_xaxes(title_text="Time (s)", gridcolor="gray", zerolinecolor="gray")
    fig.update_yaxes(
        title_text="Control inputs (RPM^2)", gridcolor="gray", zerolinecolor="gray"
    )
    fig.show()
    # Plot Distance + position/angle error
    ori_errs = []
    pos_errs = []
    states = [pose2htm(p_hist[i, :], R_hist[i, :, :]) for i in range(final_index)]
    # Compute the distance, position error, and orientation error
    for closest_point, state in zip(nearest_htms, states):
        p_near = closest_point[:3, 3]
        ori_near = closest_point[:3, :3]
        p_curr = state[:3, 3]
        ori_curr = state[:3, :3]
        pos_errs.append(np.linalg.norm(p_near - p_curr) * 100)
        trace_ = np.trace(ori_near @ np.linalg.inv(ori_curr))
        acos = np.arccos((trace_ - 1) / 2)
        # checks if acos is nan
        if np.isnan(acos):
            acos = 0
        ori_errs.append(acos * 180 / np.pi)

    # Create a figure with three plots, one above another. First the distance,
    # then position error, and the orientation error
    time_vec = np.arange(0, len(pos_errs) * dt, dt)
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.02)
    fig.add_trace(
        go.Scatter(x=time_vec, y=dist_hist[:, 0], showlegend=False, line=dict(width=3)),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(x=time_vec, y=pos_errs, showlegend=False, line=dict(width=3)),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Scatter(x=time_vec, y=ori_errs, showlegend=False, line=dict(width=3)),
        row=3,
        col=1,
    )
    fig.update_xaxes(
        title_text="Time (s)", gridcolor="gray", zerolinecolor="gray", row=3, col=1
    )
    fig.update_xaxes(
        title_text="", gridcolor="gray", zerolinecolor="gray", row=1, col=1
    )
    fig.update_xaxes(
        title_text="", gridcolor="gray", zerolinecolor="gray", row=2, col=1
    )
    fig.update_yaxes(
        title_text="Distance D",
        gridcolor="gray",
        zerolinecolor="gray",
        row=1,
        col=1,
        title_standoff=30,
    )
    fig.update_yaxes(
        title_text="Pos. error (cm)",
        gridcolor="gray",
        zerolinecolor="gray",
        row=2,
        col=1,
        title_standoff=30,
    )
    fig.update_yaxes(
        title_text="Ori. error (deg)",
        gridcolor="gray",
        zerolinecolor="gray",
        row=3,
        col=1,
        title_standoff=30,
    )
    fig.update_layout(width=718.110, height=605.9155)
    fig.show()

nvim_plot()
