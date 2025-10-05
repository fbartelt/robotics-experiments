#%%
import re
import pickle
import numpy as np
import scipy as sp
import uaibot as ub
import pandas as pd
import os
import plotly.graph_objects as go
import plotly.express as px
from uaibot_cpp_bind import expSO3, SmapSO3, SmapSE3, expSE3, ECdistance
import plotly.colors as pc
from plotly.subplots import make_subplots


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
    angle = np.pi / 6 * np.sin(2 * np.pi * s)
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
    angle = np.pi / 6 * np.sin(2 * np.pi * s)
    # angle = theta
    orientation = np.array(
        [
            [1, 0, 0],
            [0, np.cos(angle), np.sin(angle)],
            [0, -np.sin(angle), np.cos(angle)],
        ]
    )
    chain = np.pi / 6 * 2 * np.pi * np.cos(2 * np.pi * s)
    # chain = 2 * np.pi
    dorientation_ds = chain * SmapSO3(np.array([1, 0, 0])) @ orientation
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
    # precomputed = np.array(precomputed)
    return precomputed


def create_uav(
    pos,
    ori,
    arm_length=0.2,
    motor_size=4,
    body_size=5,
    color_body="black",
    color_arms="gray",
    color_motors="black",
):
    """
    Create a simple octarotor (8-propeller) model centered at pos (3,)
    and oriented according to ori (3x3 rotation matrix).
    Returns a list of plotly traces.
    """
    # Octarotor arm directions (8 evenly spaced around z)
    angles = np.linspace(0, 2 * np.pi, 8, endpoint=False)
    local_arm_dirs = np.array([[np.cos(a), np.sin(a), 0] for a in angles])

    traces = []

    # Draw central body (sphere)
    traces.append(
        go.Scatter3d(
            x=[pos[0]],
            y=[pos[1]],
            z=[pos[2]],
            mode="markers",
            marker=dict(size=body_size, color=color_body),
            showlegend=False,
        )
    )

    for i, dir_local in enumerate(local_arm_dirs):
        # Rotate direction
        dir_global = ori @ dir_local
        arm_end = pos + arm_length * dir_global

        # Arm line
        traces.append(
            go.Scatter3d(
                x=[pos[0], arm_end[0]],
                y=[pos[1], arm_end[1]],
                z=[pos[2], arm_end[2]],
                mode="lines",
                line=dict(color=color_arms, width=4),
                showlegend=False,
            )
        )

        # Motor/propeller marker
        traces.append(
            go.Scatter3d(
                x=[arm_end[0]],
                y=[arm_end[1]],
                z=[arm_end[2]],
                mode="markers",
                marker=dict(size=motor_size, color=color_motors),
                showlegend=False,
            )
        )

    return traces


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
    # for i, coord in enumerate(coord_balls):
    for i, (coord, ori) in enumerate(zip(coord_balls, ori_balls)):
        if i == 0:
            color = colorscale[3]
        elif i == len(coord_balls) - 1:
            color = colorscale[4]
        else:
            solid_color = colorscale[6]
            rgba_color = pc.hex_to_rgb(solid_color) + (0.6,)
            # color = "rgba(172, 99, 250, 0.6)"
            color = f"rgba{rgba_color}"

        # fig.add_trace(
        #     go.Scatter3d(
        #         x=[coord[0]],
        #         y=[coord[1]],
        #         z=[coord[2]],
        #         mode="markers",
        #         marker=dict(size=ball_size, color=color),
        #     )
        # )

        uav_traces = create_uav(
            coord,
            ori,
            arm_length=0.4,
            motor_size=ball_size / 2,
            body_size=ball_size,
            color_body=color,
            color_arms=color,
            color_motors=color,
        )
        for trace in uav_traces:
            fig.add_trace(trace)

    if add_lineplot:
        fig.add_scatter3d(
            x=coordinates[0, :], y=coordinates[1, :], z=coordinates[2, :], mode="lines"
        )

    fig.update_layout(showlegend=False)

    return fig


def progress_bar(i, imax):
    """Prints a progress bar in the terminal."""
    bar_len = 60
    filled_len = int(round(bar_len * i / float(imax)))

    percents = round(100.0 * i / float(imax), 1)
    bar = "=" * filled_len + "-" * (bar_len - filled_len)

    print(f"[{bar}] {percents}%\r", end="")
    if i == imax:
        print()


def pose2htm(p, R):
    """Homogeneous transformation matrix from position and rotation."""
    p = np.array(p)
    htm = np.eye(4)
    htm[0:3, 0:3] = R
    htm[0:3, 3] = p.ravel()
    return htm


def get_stable_index(distances, threshold=0.7, window_size=30):
    """Get the index where the distance to the curve is stable, i.e. the
    average of the last 30 samples is below the threshold.
    """
    for i in range(len(distances) - window_size):
        if np.mean(distances[i : i + window_size]) < threshold:
            return i
    return -1


def check_traversal(indexes, n_points):
    """Check if the system traversed the whole curve. Returns True if the
    system traversed the whole curve, False otherwise. Also returns the
    time spent in seconds, assuming a control frequency of 100 Hz.
    Accepts 98% of the points to account for noise.
    """
    indexes = np.array(indexes)
    unique_indexes = np.unique(indexes)
    success = len(unique_indexes) >= 0.98 * n_points
    # Checks the time spent
    dt = 10e-3
    time_spent = len(indexes) * dt
    return success, time_spent


# %%
path = "/home/fbartelt/Documents/Projetos/robotics-experiments/omniocta/data"
# Get all pickle files in the path
files = [f for f in os.listdir(path) if f.endswith(".pkl")]
print(files)
# File 0 = highest noise full traversal
# File 1 = highest noise failed traversal
# File 2 = medium noise full traversal

with open(os.path.join(path, files[2]), "rb") as f:
    data = pickle.load(f)

print(data.keys())

#%%
p_hist = data["p_hist"]
N = p_hist.shape[0]
final = int(N * 0.75)
R_hist = data["R_hist"]
n_points = 5000
r, b, d = 2.5, 1, 0.2
curve = precomputed_hd(hd, n_points, r, b, d)
curve = np.array([c[:3, 3] for c in curve])


def nvimplot():
    colorscale = px.colors.qualitative.Plotly
    colorscale[3] = "black"  # start ball
    colorscale[4] = "rgba(255, 200, 0, 1.0)"  # end ball
    colorscale[5] = "cyan"
    colorscale[6] = "#4d4c4c"  # intermediate balls
    # eye = {'x': -0.12256678707924387, 'y': -1.9433578647375376, 'z': 0.36193708924324386}
    # eye = {'x': -1.2296970402513263 * 1.2,
    #        'y': 1.2869397509241487 * 1.2, 
    #        'z': 0.18525174039831366 * 1.2
    #        }
    eye = {'x': -1.1593384413157246, 'y': -0.06194502710745191, 'z': 1.6230421457524087}
    camera = dict(
        eye=eye,
        # eye=dict(x=eye[0], y=eye[1], z=eye[2]),
        # center=dict(x=center[0], y=center[1], z=center[2]),
    )
    scal = 0.05
    # init_ball = 0
    # final_ball = int(N * 0.07)
    # init_ball = int(N * 0.07)
    # final_ball = int(N * 0.15)
    init_ball = int(N * 0.15)
    final_ball = int(N * 0.22)
    fig = vector_field_plot(
        p_hist,
        p_hist,
        R_hist,
        curve,
        num_arrows=0,
        init_ball=init_ball,
        final_ball=final_ball,
        num_balls=10,
        show_curve=True,
        ball_size=5,
        curve_width=5,
        path_width=2,
        frame_scale=[scal * 10, scal * 7, scal * 4],
        frame_width=4,
        prev_path_style="dot",
        colorscale=colorscale,
    )
    fig.update_layout(
        scene=dict(
            xaxis=dict(showbackground=False, gridcolor="rgba(0,0,0,0.1)"),
            yaxis=dict(showbackground=False, gridcolor="rgba(0,0,0,0.1)"),
            zaxis=dict(showbackground=False, gridcolor="rgba(0,0,0,0.1)"),
        ),
        margin=dict(t=0, b=0, r=0, l=0, pad=0),
        scene_camera=camera,
        scene_aspectmode="cube",
        scene_yaxis=dict(
            # range=[-0.4, 0.4],
            ticks="outside",
            # tickvals=yticks,
            # ticktext=yticks,
            gridcolor="rgba(148, 150, 153, 1)",
            # showticklabels=False,
            tickangle=0,
            title="y",
        ),
        scene_zaxis=dict(
            range=[np.min(p_hist[:, 2] - 0.5), np.max(p_hist[:, 2]) + 0.5],
            ticks="outside",
            # tickvals=zticks,
            # ticktext=zticks,
            gridcolor="rgba(148, 150, 153, 1)",
            # showticklabels=False,
            title="z",
        ),
        scene_xaxis=dict(
            # range=[-0.4, 0.4],
            range=[-4, 6],
            # tickvals=xticks,
            gridcolor="rgba(148, 150, 153, 1)",
            # showticklabels=False,
            tickangle=0,
            title="x",
        ),
        width=1080,
        height=1080,
        autosize=False,
        showlegend=False,
        paper_bgcolor="rgba(0,0,0,0)",  # transparent
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig
fig = nvimplot()
fig.show()

#eye {'x': -0.29883993714371976, 'y': -2.2237394500062773, 'z': 0.4590366243128926}
# {'x': -1.5514405060163505, 'y': 1.700083478236783, 'z': 0.044324498514860936}
# {'x': -1.139440140723252, 'y': -0.011558745231574939, 'z': 2.0808502028079072}
# fig.write_image("figures/vf_traj_03.svg", width=1080, height=1080, scale=2)
# f = go.FigureWidget(fig)
# f
# f.layout['scene']['camera']['eye']._props
# %%
