#%%
import sys
import pickle
import os
import plotly.graph_objects as go
import numpy as np


def plot_trajectory(p_hist, curve_htm):
    fig = go.Figure(
        go.Scatter3d(
            x=[p[0] for p in p_hist],
            y=[p[1] for p in p_hist],
            z=[p[2] for p in p_hist],
            mode="lines",
            marker=dict(
                size=2,
                colorscale="Viridis",
                colorbar=dict(title="Time (s)"),
                opacity=0.8,
            ),
            line=dict(color="darkblue", width=2),
        )
    )
    fig.add_trace(
        go.Scatter3d(
            x=[cp[0, 3] for cp in curve_htm],
            y=[cp[1, 3] for cp in curve_htm],
            z=[cp[2, 3] for cp in curve_htm],
            mode="lines",
            line=dict(color="red", width=2),
            name="Curve",
        )
    )
    return fig

def plot_distance(time, min_distances):
    fig = go.Figure(
        go.Scatter(
            x=time,
            y=min_distances,
            mode="lines",
        )
    )
    return fig


def plot_zeta(time, zeta_hist):
    fig = go.Figure(
        go.Scatter(
            x=time,
            y=zeta_hist,
            mode="lines",
        )
    )
    return fig

def plot_velocities(time, xi_hist, psi_hist):
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=time,
            y=[np.linalg.norm(xi[:3]) for xi in xi_hist],
            mode="lines",
            name="System Linear Velocity",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=time,
            y=[np.linalg.norm(xi[3:]) for xi in xi_hist],
            mode="lines",
            name="System Angular Velocity",
            )
    )
    fig.add_trace(
        go.Scatter(
            x=time,
            y=[np.linalg.norm(psi[:3]) for psi in psi_hist],
            mode="lines",
            name="Desired Linear Velocity",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=time,
            y=[np.linalg.norm(psi[3:]) for psi in psi_hist],
            mode="lines",
            name="Desired Angular Velocity",
        )
    )
    return fig

def plot_input(time, input_hist):
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=time,
            y=[inp[0] for inp in input_hist],
            mode="lines",
            name="Fx",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=time,
            y=[inp[1] for inp in input_hist],
            mode="lines",
            name="Fy",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=time,
            y=[inp[2] for inp in input_hist],
            mode="lines",
            name="Fz",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=time,
            y=[inp[3] for inp in input_hist],
            mode="lines",
            name="Tx",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=time,
            y=[inp[4] for inp in input_hist],
            mode="lines",
            name="Ty",
            )
    )
    fig.add_trace(
        go.Scatter(
            x=time,
            y=[inp[5] for inp in input_hist],
            mode="lines",
            name="Tz",
            )
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
    # position = [
    #     r * np.cos(theta),
    #     r * np.sin(theta),
    #     b + d * r**2 * (np.cos(theta) ** 2 - np.sin(theta) ** 2),
    # ]
    position = [
        r * (np.sin(theta) + 2 * np.sin(2 * theta)),
        r * (np.cos(theta) - 2 * np.cos(2 * theta)),
        b + r * (-np.sin(3 * theta)),
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
    # dposition_ds = [
    #     -r * 2 * np.pi * np.sin(theta),
    #     r * 2 * np.pi * np.cos(theta),
    #     d
    #     * r**2
    #     * 2
    #     * (-2 * np.cos(theta) * np.sin(theta) - 2 * np.sin(theta) * np.cos(theta))
    #     * 2
    #     * np.pi,
    # ]

    dposition_ds = [
        r * 2 * np.pi * (np.cos(theta) + 2 * 2 * np.cos(2 * theta)),
        r * 2 * np.pi * (-np.sin(theta) + 2 * 2 * np.sin(2 * theta)),
        r * 2 * np.pi * (-3 * np.cos(3 * theta)),
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


def pose2htm(p, R):
    """Homogeneous transformation matrix from position and rotation."""
    p = np.array(p)
    htm = np.eye(4)
    htm[0:3, 0:3] = R
    htm[0:3, 3] = p.ravel()
    return htm


def progress_bar(i, imax):
    """Prints a progress bar in the terminal.

    Parameters
    ----------
    i : int
        Current iteration.
    imax : int
        Maximum number of iterations.
    """
    sys.stdout.write("\r")
    sys.stdout.write(
        "[%-20s] %d%%" % ("=" * round(20 * i / (imax - 1)), round(100 * i / (imax - 1)))
    )
    sys.stdout.flush()


# Initial conditions
n_points = 3000
n_points = 2000
r, b, d = 0.35, 1, 0.2
r, b = 0.7, 0.4
curve = precomputed_hd(hd, n_points, r, b, d)
path = "./"
pkl_files = [f for f in os.listdir(path) if f.endswith(".pkl")]
file = pkl_files[0]  # choose the first pkl file found

with open(os.path.join(path, file), "rb") as f:
    data = pickle.load(f)

H_hist = data["H_hist"]
xi_hist = data["xi_hist"]
xi_dot_hist = data["xi_dot_hist"]
psi_hist = data["psi_hist"]
o_hat_hist = data["o_hat_hist"]
r_hat_hist = data["r_hat_hist"]
closest_indexes = data["closest_indexes"]
min_distances = data["min_distances"]
zeta_hist = data["zeta_hist"]
taui_hist = data["taui_hist"]
aprox_hist = data["aprox_hist"]
input_hist = data["input_hist"]
r_i = data["r_i"]
full_traversal = data["full_traversal"]
converge_idx = data["converge_idx"]

p_hist = [np.array(H[:3, 3]).ravel() for H in H_hist]
time = np.array(list(range(len(xi_hist)))) * 1e-2

fig = plot_trajectory(p_hist, curve)
fig.show()
fig = plot_distance(time, min_distances)
fig.show()
fig = plot_zeta(time, zeta_hist)
fig.show()

fig = plot_velocities(time, xi_hist, psi_hist)
fig.show()

fig = plot_input(time, input_hist)
fig.show()
o_hat_hist[-1]
