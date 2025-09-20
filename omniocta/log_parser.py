import re
import pickle
import numpy as np
import uaibot as ub
import pandas as pd
import os
import plotly.graph_objects as go
from uaibot_cpp_bind import expSO3, SmapSO3, SmapSE3, expSE3, ECdistance

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
        p = np.array(p_hist[i]).reshape(3, 1)
        R = np.array(R_hist[i]).reshape(3, 3)
        htm = pose2htm(p, R)

        dist, idx = ECdistance(htm, curve)
        dist_hist.append(dist)
        closest_point = curve[idx]
        p_near = closest_point[:3, 3]
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
        average_dist = np.mean(dist_hist[converge_idx:])
        average_pos_err = np.mean(pos_err_hist[converge_idx:])
        average_ori_err = np.mean(ori_err_hist[converge_idx:])

    return average_dist, average_pos_err, average_ori_err


n_points = 5000
r, b, d = 2.5, 1, 0.2
curve = precomputed_hd(hd, n_points, r, b, d)
curve_derivative = precomputed_hd(hd_derivative, n_points, r, b, d)

path = "/home/fbartelt/Documents/Projetos/robotics-experiments/omniocta/data"
# Get all pickle files in the path
files = [f for f in os.listdir(path) if f.endswith('.pkl')]

stats = {}
for f in files:
    with open(os.path.join(path, f), 'rb') as file:
        pos_std = re.findall(r'pos_(\d+\.\d+)', f)
        ori_std = re.findall(r'ori_(\d+\.\d+)', f)
        print(f"Processing file: {f} with pos_std: {pos_std} and ori_std: {ori_std}")
        data = pickle.load(file)
        p_hist = data['p_hist']
        R_hist = data['R_hist']

        avg_dist, avg_pos_err, avg_ori_err = get_average_stable_errors(p_hist, R_hist, curve)
        stats[f"{pos_std[0]}_{ori_std[0]}"] = {
            'avg_dist': avg_dist,
            'avg_pos_err': avg_pos_err,
            'avg_ori_err': avg_ori_err,
        }
# Create df and order by index
df = pd.DataFrame.from_dict(stats, orient='index')
df
# Plot distances of each file
# fig = go.Figure()
# for i, dist in enumerate(average_dists):
#     fig.add_trace(go.Scatter(y=dist, mode='lines+markers', name=f'Run {i+1}'))
# fig.show()
# print(average_dists)
