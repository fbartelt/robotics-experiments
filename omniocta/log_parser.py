import re
import pickle
import numpy as np
import uaibot as ub
import pandas as pd
import os
import plotly.graph_objects as go
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



def progress_bar(i, imax):
    """Prints a progress bar in the terminal."""
    bar_len = 60
    filled_len = int(round(bar_len * i / float(imax)))

    percents = round(100.0 * i / float(imax), 1)
    bar = '=' * filled_len + '-' * (bar_len - filled_len)

    print(f'[{bar}] {percents}%\r', end='')
    if i == imax:
        print()

def pose2htm(p, R):
    """Homogeneous transformation matrix from position and rotation."""
    p = np.array(p)
    htm = np.eye(4)
    htm[0:3, 0:3] = R
    htm[0:3, 3] = p.ravel()
    return htm


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
    precomputed = np.array(precomputed)
    return precomputed

def get_average_stable_errors(p_hist, R_hist, curve, threshold=0.7, n_stable=30):
    average_dist, average_pos_err, average_ori_err = -1, -1, -1
    dist_hist, pos_err_hist, ori_err_hist = [], [], []
    imax = len(p_hist)
    for i in range(imax):
        progress_bar(i, imax)
        p = p_hist[i, :].ravel()
        R = R_hist[i, :, :]
        # p = np.array(p_hist[i]).reshape(3, 1)
        # R = np.array(R_hist[i]).reshape(3, 3)
        htm = pose2htm(p, R)

        dist, idx = ECdistance(htm, curve)
        dist_hist.append(dist)
        closest_point = curve[idx]
        p_near = closest_point[:3, 3].ravel()
        ori_near = closest_point[:3, :3]
        p_curr = p.copy()
        ori_curr = R.copy()
        pos_err_hist.append(np.linalg.norm(p_near - p_curr) * 100)
        trace_ = np.trace(ori_near @ np.linalg.inv(ori_curr))
        acos = np.arccos((trace_ - 1) / 2)
        # checks if acos is nan
        if np.isnan(acos):
            acos = 0
        ori_err_hist.append(acos * 180 / np.pi)
    # Get index where average of last 30 samples is below 0.7
    dist_hist = np.array(dist_hist)
    pos_err_hist = np.array(pos_err_hist)
    ori_err_hist = np.array(ori_err_hist)
    converge_idx = -1
    for i in range(len(dist_hist) - n_stable):
        if np.mean(dist_hist[i:i+n_stable]) < threshold:
            converge_idx = i
            break
    if converge_idx == -1:
        print("Did not converge in ")
    else:
        print(converge_idx)
        average_dist = np.mean(dist_hist[converge_idx:])
        average_pos_err = np.mean(pos_err_hist[converge_idx:])
        average_ori_err = np.mean(ori_err_hist[converge_idx:])

    return average_dist, average_pos_err, average_ori_err, dist_hist, pos_err_hist, ori_err_hist


# n_points = 5000
# r, b, d = 2.5, 1, 0.2
# curve = precomputed_hd(hd, n_points, r, b, d)
# curve_derivative = precomputed_hd(hd_derivative, n_points, r, b, d)
#
# path = "/home/fbartelt/Documents/Projetos/robotics-experiments/omniocta/data"
# # Get all pickle files in the path
# files = [f for f in os.listdir(path) if f.endswith('.pkl')]
#
# stats = {}
# for f in files:
#     with open(os.path.join(path, f), 'rb') as file:
#         pos_std = re.findall(r'pos_(\d+\.\d+)', f)
#         ori_std = re.findall(r'ori_(\d+\.\d+)', f)
#         print(f"Processing file: {f} with pos_std: {pos_std} and ori_std: {ori_std}")
#         data = pickle.load(file)
#         p_hist = data['p_hist']
#         R_hist = data['R_hist']
#
#         avg_dist, avg_pos_err, avg_ori_err = get_average_stable_errors(p_hist, R_hist, curve)
#         stats[f"{pos_std[0]}_{ori_std[0]}"] = {
#             'avg_dist': avg_dist,
#             'avg_pos_err': avg_pos_err,
#             'avg_ori_err': avg_ori_err,
#         }
# # Create df and order by index
# df = pd.DataFrame.from_dict(stats, orient='index')
# df
# # Plot distances of each file
# # fig = go.Figure()
# # for i, dist in enumerate(average_dists):
# #     fig.add_trace(go.Scatter(y=dist, mode='lines+markers', name=f'Run {i+1}'))
# # fig.show()
# # print(average_dists)

path = "/home/fbartelt/Documents/Projetos/robotics-experiments/omniocta/data/grid_search_results.pkl"
with open(path, 'rb') as file:
    results = pickle.load(file)

df = pd.DataFrame.from_dict(results)
print(df.columns)
sub_df = df[["pos_std", "ori_std", "mean_avg_pos_err",  "std_avg_pos_err", "mean_avg_ori_err", "std_avg_ori_err"]]
# sub_df.sort_values(by=["mean_avg_pos_err", "std_avg_pos_err"], )
sub_df.sort_values(by=["pos_std", "ori_std"], )

np.array(df[["all_avg_pos_errs"]].iloc[-1].values[0]).max()
#%%
path = "/home/fbartelt/Documents/Projetos/robotics-experiments/omniocta/data/pos_0.1_ori_0.0_seed_0.pkl"

with open(path, 'rb') as file:
    data = pickle.load(file)

p_hist = data['p_hist']
R_hist = data['R_hist']
v_hist = data['v_hist']
n_points = 5000
r, b, d = 2.5, 1, 0.2
curve = precomputed_hd(hd, n_points, r, b, d)
curve_pos = np.array([c[:3, 3] for c in curve])
print(data.keys())
np.mean(df.iloc[0]['all_avg_pos_errs'])
fig = vector_field_plot(p_hist, v_hist, R_hist, curve_pos, num_arrows=0, init_ball=0, final_ball=len(p_hist)-1, num_balls=20, add_lineplot=False, show_curve=True, ball_size=3, frame_scale=0.1)
fig.show()
go.Figure(go.Scatter(y=data['dist_hist'].ravel(), mode='lines')).show()



#%%
mean_dist, mean_pos, mean_ori, dist_hist, pos_err_hist, ori_err_hist = get_average_stable_errors(p_hist, R_hist, curve)

def nvim_err_plot(dist_hist, pos_err_hist, ori_err_hist):
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, subplot_titles=("Distance to curve", "Position error (cm)", "Orientation error (deg)"))
    xvec = np.arange(len(dist_hist)) * 10e-3
    fig.add_trace(
        go.Scatter(
            y=dist_hist,
            mode='lines',
            name='Distance to curve',
            line=dict(color='blue'),
        )
    , row=1, col=1)
    fig.add_trace(
        go.Scatter(
            y=pos_err_hist,
            mode='lines',
            name='Position error (cm)',
            line=dict(color='orange'),
        )
    , row=2, col=1)
    fig.add_trace(
        go.Scatter(
            y=ori_err_hist,
            mode='lines',
            name='Orientation error (deg)',
            line=dict(color='green'),
        )
    , row=3, col=1)

    return fig

fig = nvim_err_plot(dist_hist, pos_err_hist, ori_err_hist)
fig.show()


