# %%
import sys
import numpy as np
import plotly.graph_objects as go
import plotly.colors as pc
from uaibot.robot import Robot
from uaibot.utils import Utils
from scipy.linalg import block_diag
from plotly.subplots import make_subplots
from numba import njit



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



def S(vec):
    """Skew-symmetric matrix from vector."""
    vec = np.array(vec).ravel()
    return np.array([[0, -vec[2], vec[1]], [vec[2], 0, -vec[0]], [-vec[1], vec[0], 0]])



def pose2htm(p, R):
    """Homogeneous transformation matrix from position and rotation."""
    p = np.array(p)
    htm = np.eye(4)
    htm[0:3, 0:3] = R
    htm[0:3, 3] = p.ravel()
    return htm



def exp_SO3(skew_matrix):
    """Exponential map so(3) -> SO(3) using Rodrigues' formula."""
    theta = np.linalg.norm([skew_matrix[2, 1], skew_matrix[0, 2], skew_matrix[1, 0]])
    if theta < 1e-6:
        return np.eye(3) + skew_matrix
    else:
        K = skew_matrix / theta
        return np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * (K @ K)


def motor_dynamics(u_prev, u_d, dt, tau=0.1):
    """Simple motor dynamics model with saturation."""
    alpha = np.exp(-dt / tau)
    return alpha * u_prev + (1 - alpha) * u_d



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
    orientation = np.array(
        [
            [1, 0, 0],
            [0, np.cos(theta), np.sin(theta)],
            [0, -np.sin(theta), np.cos(theta)],
        ]
    )
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
    skew_matrix = S([theta, 0, 0])
    dhds[:3, :3] = skew_matrix @ np.array(
        [
            [1, 0, 0],
            [0, np.cos(theta), np.sin(theta)],
            [0, -np.sin(theta), np.cos(theta)],
        ]
    )
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


def task_func(p_, R_, sd=None, Rd=None):
    if Rd is None:
        xd, yd, zd = (
            np.array([1, 0, 0]).reshape(-1, 1),
            np.array([0, 1, 0]).reshape(-1, 1),
            np.array([0, 0, 1]).reshape(-1, 1),
        )
    else:
        xd, yd, zd = Rd[:, 0].reshape(-1, 1), Rd[:, 1].reshape(-1, 1), Rd[:, 2].reshape(-1, 1)
    if sd is None:
        sd = np.array([0, 0, 0.8]).reshape(-1, 1)
    else:
        sd = np.array(sd).reshape(-1, 1)
    xe, ye, ze = R_[:, 0], R_[:, 1], R_[:, 2]
    se = p_
    AA = np.vstack((xd.T @ S(xe), yd.T @ S(ye), zd.T @ S(ze)))
    jacobian_task = block_diag(np.eye(3), AA)
    task = np.vstack((se - sd, 1 - xd.T @ xe, 1 - yd.T @ ye, 1 - zd.T @ ze))
    xi_d = Utils.dp_inv(jacobian_task) @ (-0.5 * task)
    return xi_d



m = 3.2  # mass (Kg)
M_body = 2 * 0.015 * np.eye(3)  # inertia (Kg*m^2)

w_min = 00  # [Hz]
w_max = 240  # [Hz]

# Design parameters
n = 8

k_f = 9.9016 * 10 ** (-4)  # N/RPM^2
k_d = 1.9 * 10 ** (-5)  # Nm/RPM^2
u_max = w_max**2 * 1e6
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
motor_tau = 0.01  # time constant of the motor dynamics

# Simulation parameters
dt = 1e-3
T = 160
imax = int(T / dt)

# Initial conditions
n_points = 5000
r, b, d = 1, 1, 0.2
curve = precomputed_hd(hd, n_points, r, b, d)
# curve_derivative = precomputed_hd(hd_derivative, n_points, r, b, d)
R0 = np.eye(3)
p0 = np.array([2, -2, 0]).reshape(-1, 1)
# p0 = curve[0, :3, 3].reshape(-1, 1)  # Start at beginning of curve
p = p0.copy()
# R0 = curve[0, :3, :3]  # Start with orientation of curve
R = R0.copy()
v = np.array([0, 0, 0]).reshape(-1, 1)
v_dot = np.array([0, 0, 0]).reshape(-1, 1)
omega = np.array([0, 0, 0]).reshape(-1, 1)
omega_dot = np.array([0, 0, 0]).reshape(-1, 1)
# integral_term = np.array([0, 0, 0, 0, 0, 0]).reshape(-1, 1)
integral_term = (
    np.array([1.7597, 2.5895, 8.6608, 0.1203, 0.1916, -0.4441]).reshape(-1, 1)
) * 0
# integral_term = np.array(
#         [1.7597 * 0, 2.5895 * 0, 8.6608, 0.1203 * 0, 0.1916 * 0, -0.4441 * 0]
#     ).reshape(-1, 1) * 0


u = np.zeros((n, 1))  # Initial propeller forces
inner_motor = np.zeros((n, 1))

# Gains
Kp, Ki = 20 * 3.0, 10 * 1.2  # 20 percent works for tau 0.01
Kp, Ki = 20 * 4, 10 * 1.2  # 20 percent works for tau 0.01
Kpv = 20 * 200.0 * 1.4 / 2
Kpw = 20 * 2.2
Kiv = 20 * 10.0 * 2.0
Kiw = 20 * 16.0
Kp = np.diag([Kpv, Kpv, Kpv, Kpw, Kpw , Kpw ])
Ki = np.diag([Kiv, Kiv, Kiv, Kiw, Kiw, Kiw])
# Kp, Ki = 20 * 0.2, 10 * 0.1  # 20 percent works for tau 0.01
kn1, kn2 = 1 * 0.4, 10
kt1, kt2, kt3 = kn1 * 1, 1, kn2
# Disturbances
rng = np.random.default_rng(42)
gravity_force = np.array([0, 0, m * 9.81]).reshape(-1, 1)
F_disturb = np.zeros((3, 1))
tau_disturb = np.zeros((3, 1))

# Logs
p_hist, v_hist = np.zeros((imax, 3)), np.zeros((imax, 3))
R_hist, omega_hist = np.zeros((imax, 3, 3)), np.zeros((imax, 3))
v_d_hist, omega_d_hist = np.zeros((imax, 3)), np.zeros((imax, 3))
F_hist, tau_hist = np.zeros((imax, 3)), np.zeros((imax, 3))
dist_hist = np.zeros((imax, 1))
nearest_htms = np.zeros((imax, 4, 4))

for i in range(imax):
    progress_bar(i, imax)
    sin_omega = (2 * np.pi / T) * 2
    # F_disturb = np.sin(sin_omega * i * dt) * np.array([1, 1, 1]).reshape(-1, 1) * 3
    # tau_disturb = np.sin(sin_omega * i * dt) * np.array([1, 1, 1]).reshape(-1, 1) * 3
    if imax * 0.4 < i < imax * 0.8:
        F_disturb = np.array([1, 1, 1]).reshape(-1, 1) * 12 * 0.5 * 0
        tau_disturb = np.array([1, 1, 1]).reshape(-1, 1) * 12 * 0.5 * 0
        # F_disturb = rng.normal(loc=0.0, scale=2.0, size=(3, 1))
        # tau_disturb = rng.normal(loc=0.0, scale=3.0, size=(3, 1))
    else:
        F_disturb = np.zeros((3, 1))
        tau_disturb = np.zeros((3, 1))

    htm = pose2htm(p, R)  # Current pose in HTM
    xi = np.vstack((v, omega))  # Current twist [v; omega]
    xi_d, dist, imin = Robot.vector_field_SE3(
        htm,
        curve,
        kt1=kt1,
        kt2=kt2,
        kt3=kt3,
        kn1=kn1,
        kn2=kn2,
        curve_derivative=[],
        delta=1e-3,
        ds=1e-3,
    )
    xi_d = np.array(xi_d).reshape(-1, 1)
    # xi_d = -0.5*(p - np.array([0, 0, 0]).reshape(-1, 1))
    if i < imax * 0.4 and i * dt < 10.0:
        xi_d = task_func(p, R, sd=p0 + 0.1 * np.array([1, 1, 1]).reshape(-1, 1),
                         Rd=R0)
    # print(f"\nXi_d = {xi_d.ravel()}")
    # print(f"Xi = {xi.ravel()}")
    # print(f"Integral term = {integral_term.ravel()}")
    # print(f"Xi error = {(xi_d - xi).ravel()}")
    # xi_d = np.vstack((xi_d, np.array([0, 0, 0]).reshape(-1, 1)))  # Desired twist
    Hd_star = curve[imin]

    M = R @ M_body @ R.T  # Inertia matrix in inertial frame
    wrench_d = (
        Kp @ (xi_d - xi) + Ki @ integral_term
    )  # integral_term = integral of (xi_d - xi) over time
    bias = np.vstack((-R.T @ gravity_force, np.zeros((3, 1))))
    R_dup = block_diag(R, R)
    wrench_d = R_dup.T @ wrench_d.reshape(
        -1, 1
    )  # Transform desired torque to body frame
    wrench_d = wrench_d + bias  # Add bias to compensate for weight and other factors

    # Get desired propeller forces and saturate
    # FOR SOME REASON QP RESULT IS AWFUL
    # u_d = Utils.solve_qp(
    #     2 * A.T @ A + 1e-5 * np.eye(n),
    #     -2 * A.T @ wrench_d,
    #     np.eye(n),
    #     np.zeros((n, 1)),
    #     # np.vstack((-np.eye(n), np.eye(n))),
    #     # np.vstack((-u_max * np.ones((n, 1)), np.zeros((n, 1)))),
    # )
    # u_d = np.array(u_d).reshape(-1, 1)
    u_d = np.clip(np.linalg.pinv(A) @ wrench_d, 0, u_max)
    inner_motor = inner_motor + (-inner_motor / motor_tau + u_d) * dt
    u = inner_motor / motor_tau
    # u = u_d
    wrench = A @ u
    wrench = R_dup @ wrench.reshape(-1, 1)  # Transform torque to inertial frame
    F, tau = wrench[:3], wrench[3:]
    tau = tau

    # Euler integration
    v_d, omega_d = xi_d[:3], xi_d[3:]
    R = exp_SO3(S(omega * dt)) @ R
    omega = omega + omega_dot * dt
    omega_dot = np.linalg.inv(M) @ (tau + tau_disturb)
    p = p + v * dt
    v = v + v_dot * dt
    v_dot = 1 / m * (F + F_disturb - gravity_force)
    integral_term = integral_term + (xi_d - xi) * dt

    # Save data
    p_hist[i, :] = p.flatten()
    v_hist[i, :] = v.flatten()
    R_hist[i, :, :] = R
    omega_hist[i, :] = omega.flatten()
    dist_hist[i] = dist
    F_hist[i, :] = F.flatten()
    tau_hist[i, :] = tau.flatten()
    nearest_htms[i, :, :] = Hd_star
    v_d_hist[i, :] = v_d.flatten()
    omega_d_hist[i, :] = omega_d.flatten()

# Print tau and F with maximum norm
print(f"Max torque: {tau_hist[np.argmax(np.linalg.norm(tau_hist, axis=1))]}")
print(f"Max force: {F_hist[np.argmax(np.linalg.norm(F_hist, axis=1))]}")
print(f"Mean linear velocity: {np.mean(np.linalg.norm(v_hist, axis=1))}")
print(f"Mean angular velocity: {np.mean(np.linalg.norm(omega_hist, axis=1))}")

print(integral_term.ravel())


# Plotting
def nvim_plot():

    fig = vector_field_plot(
        p_hist,
        v_hist,
        R_hist,
        curve[:, :3, 3],
        num_arrows=20,
        init_ball=0,
        final_ball=int(imax) - 1,
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

    # Plot Distance + position/angle error
    ori_errs = []
    pos_errs = []
    states = [pose2htm(p_hist[i, :], R_hist[i, :, :]) for i in range(imax)]
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

    # Plot velocity errors
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.02)
    fig.add_trace(
        go.Scatter(
            x=time_vec,
            y=np.linalg.norm(v_d_hist - v_hist, axis=1),
            showlegend=False,
            line=dict(width=3),
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=time_vec,
            y=np.linalg.norm(omega_d_hist - omega_hist, axis=1),
            showlegend=False,
            line=dict(width=3),
        ),
        row=2,
        col=1,
    )
    # Plot desired velocities
    for log in (v_d_hist, v_hist, omega_d_hist, omega_hist):
        for i in range(3):
            fig.add_trace(
                go.Scatter(
                    x=time_vec,
                    y=log[:, i],
                    line=dict(width=1, dash="dash" if i % 2 == 0 else "solid"),
                ),
                row=3,
                col=1,
            )

    fig.update_xaxes(
        title_text="Time (s)", gridcolor="gray", zerolinecolor="gray", row=2, col=1
    )
    fig.update_yaxes(
        title_text="||v_d - v|| (m/s)",
        gridcolor="gray",
        zerolinecolor="gray",
        row=1,
        col=1,
        title_standoff=30,
    )
    fig.update_yaxes(
        title_text="||ω_d - ω|| (rad/s)",
        gridcolor="gray",
        zerolinecolor="gray",
        row=2,
        col=1,
        title_standoff=30,
    )
    fig.show()

    # Regulation Plots test
    # fig = go.Figure()
    # for i in range(3):
    #     fig.add_trace(
    #             go.Scatter(
    #             x = time_vec,
    #             y = p_hist[:, i],
    #             )
    #     )
    # fig.show()
    # ori_errs = []
    # states = [pose2htm(p_hist[i, :], R_hist[i, :, :]) for i in range(imax)]
    # # Compute the distance, position error, and orientation error
    # for closest_point, state in zip(nearest_htms, states):
    #     p_near = closest_point[:3, 3]
    #     ori_near = R0
    #     p_curr = state[:3, 3]
    #     ori_curr = state[:3, :3]
    #     pos_errs.append(np.linalg.norm(p_near - p_curr) * 100)
    #     trace_ = np.trace(ori_curr)
    #     # trace_ = np.trace(ori_near @ np.linalg.inv(ori_curr))
    #     acos = np.arccos((trace_ - 1) / 2)
    #     # checks if acos is nan
    #     if np.isnan(acos):
    #         acos = 0
    #     ori_errs.append(acos * 180 / np.pi)
    #
    #
    # fig = go.Figure()
    # fig.add_trace(
    #         go.Scatter(
    #         x = time_vec,
    #         y = ori_errs,
    #         )
    # )
    # fig.show()
    #
    # fig = go.Figure()
    # fig.add_trace(
    #     go.Scatter3d(
    #         x=curve[:, 0, 3],
    #         y=curve[:, 1, 3],
    #         z=curve[:, 2, 3],
    #         mode="lines",
    #         line=dict(color="blue", width=2),
    #         name="Desired Path",
    #     )
    # )
    # fig.add_trace(
    #     go.Scatter3d(
    #         x=p_hist[:, 0],
    #         y=p_hist[:, 1],
    #         z=p_hist[:, 2],
    #         mode="lines",
    #         line=dict(color="red", width=2),
    #         name="Actual Path",
    #     )
    # )
    # fig.show()
    # fig = go.Figure()
    # fig.add_trace(
    #     go.Scatter(
    #         x=np.linspace(0, T, imax),
    #         y=dist_hist[:, 0],
    #         mode="lines",
    #         line=dict(width=1),
    #         name="Distance to path",
    #     )
    # )
    # fig.show()
    # fig = go.Figure()
    # fig.add_trace(go.Scatter(x=np.linspace(0, T, imax), y=v_hist[:, 0], mode='markers', line=dict(width=2), name='v_x'))
    # fig.add_trace(go.Scatter(x=np.linspace(0, T, imax), y=v_hist[:, 1], mode='markers', line=dict(width=2), name='v_y'))
    # fig.add_trace(go.Scatter(x=np.linspace(0, T, imax), y=v_hist[:, 2], mode='markers', line=dict(width=2), name='v_z'))
    # fig.show()


nvim_plot()
