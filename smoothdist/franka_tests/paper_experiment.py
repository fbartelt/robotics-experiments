import numpy as np
import uaibot as ub
from franka_tests.create_franka_emika_3_mod import create_franka_emika_3_mod
import plotly.graph_objects as go
import webbrowser
from pathlib import Path
######################################################
# PARAMETERS
######################################################

# Operation mode, mode = 0 (Euclidean) and mode = 1 (Our proposed distance)
mode = 1

# Choose the parameters h, epsilon (eps) and sigma in the paper.
# This is already pre-selected by the 'mode' variable, but you can change it if you want.
# As an heuristic, sigma is always set internally to sqrt(max(0,1-2*eps)).
# The value of k is set to 1 (so the distance is always one time differentiable).
# It also selects the safety margin delta for obstacles, delta_obs
# (in meters) and for auto collision delta_auto. They both should be different depending on the
# smoothing parameter.

is_conservative = False

if mode == 0:
    h = 1e-6
    eps = 0
    delta_obs = 0.03
    delta_auto = 0.01
else:
    h = 0.1
    # h = 1e-2
    gamma = 1
    eps = 0.01
    # delta_obs = 0.002
    delta_obs = -1
    delta_auto = 0.0001


# Obstacles
obstacles = []
obstacles.append(
    ub.Box(
        htm=ub.Utils.trn([0.53, 0.16, 0.45]),
        width=0.35,
        depth=0.05,
        height=0.90,
        color="magenta",
    )
)
obstacles.append(
    ub.Box(
        htm=ub.Utils.trn([0.53, -0.16, 0.45]),
        width=0.35,
        depth=0.05,
        height=0.90,
        color="magenta",
    )
)
obstacles.append(
    ub.Box(
        htm=ub.Utils.trn([0.53, 0.00, 0.925]),
        width=0.35,
        depth=0.35,
        height=0.05,
        color="magenta",
    )
)

# Initial configuration (rad)
q = np.matrix([[1.0582, -1.3811, 0.3629, -1.9647, -0.959, 1.4881, -0.1534]]).T

# Target pose
htm_tg = (
    ub.Utils.trn([0.64, 0, 0.75]) * ub.Utils.roty(np.pi / 2) * ub.Utils.rotz(np.pi / 2)
)

# Sampling time (seconds)
dt = 0.01

# Control matrix for the task function (1/second)
K = np.diag([0.4, 0.4, 0.4, 0.4, 0.4, 0.4])

# Regularization factor for the task function
reg = 0.01

# Gain for the CBF inequality (1/second)
eta = 0.5

# Maximum simulation time (seconds)
t_max = 35
# t_max = 1

# Maximum number of iterations for the generalized Von Neumman's algorithm
no_iter_max = 300

# Tolerance for convergence for the generalized Von Neumman's algorithm
tol = 2e-4


######################################################
# INITIALIZATIONS
######################################################

sim_time = 0
robot = create_franka_emika_3_mod()

# sim = ub.Simulation(background_color="lightblue")
sim = ub.Simulation()
sim.add(robot)
robot.add_ani_frame(0, q)

for obs in obstacles:
    sim.add(obs)

for i, link in enumerate(robot.links):
    for j, (col_obj, _) in enumerate(link.col_objects):
        sim.add(col_obj)

frame_tg = ub.Frame(htm=htm_tg)
sim.add(frame_tg)

# Auxiliary functions

def get_joint_config():
    # In a real application, this should be replaced by the
    # function that measures the real joint position in the robot
    global robot
    return robot.q


def compute_controller(_q):
    # Compute the control input
    global robot
    global htm_tg
    global eps
    global delta
    global K
    global eta
    global obstacles
    global reg
    global no_iter_max

    # Get the number of configurations
    n = np.shape(_q)[0]
    dist = 0.0

    # Initialize matrices A and b
    mat_A = np.matrix(np.zeros((0, n)))
    mat_b = np.matrix(np.zeros((0, 1)))

    # Implement obstacle avoidance constraints and stack into A and b
    for obs in obstacles:
        if mode == 0:
            dr = robot.compute_dist(
                q=_q, obj=obs, h=h, eps=eps, no_iter_max=no_iter_max, tol=tol
            )
        else:
            # dr = robot.signed_distance(obj=obs, q=_q, r=h)
            dr = robot.signed_distance(obj=obs, q=_q, gamma=gamma, is_conservative=is_conservative)
        mat_A = np.vstack((mat_A, dr.jac_dist_mat))
        mat_b = np.vstack((mat_b, -eta * (dr.dist_vect - delta_obs)))

    dist_vect_d = dr.dist_vect
    jac_mat_d = dr.jac_dist_mat
    dist = np.min(dr.dist_vect)
    # Implement auto-collision avoidance and stack into A and b
    # if mode == 0:
    # if mode != 2:
    dr = robot.compute_dist_auto(q=_q, h=h, eps=eps, no_iter_max=no_iter_max, tol=tol)
    # else:
    #     dr = robot.signed_distance_auto(q=_q, gamma=h, is_conservative=is_conservative)
    mat_A = np.vstack((mat_A, dr.jac_dist_mat))
    mat_b = np.vstack((mat_b, -eta * (dr.dist_vect - delta_auto)))

    # Implement constraints for joint limits avoidance and stack into A and b
    mat_A = np.vstack((mat_A, np.identity(n)))
    mat_b = np.vstack((mat_b, -eta * (_q - robot.joint_limit[:, 0])))

    mat_A = np.vstack((mat_A, -np.identity(n)))
    mat_b = np.vstack((mat_b, -eta * (robot.joint_limit[:, 1] - _q)))

    # Compute task function
    r, jac_r = robot.task_function(q=_q, htm_tg=htm_tg)

    # Assemble the H and f matrices of the optimization problem
    mat_H = jac_r.T * jac_r + reg * np.identity(n)
    mat_f = jac_r.T * (K * r)

    # Compute the control input
    try:
        u = ub.Utils.solve_qp(mat_H, mat_f, mat_A, mat_b)
    except:
        u = 0 * _q
        print("Unfeasible!")
        print(f"Distances: {dist_vect_d.ravel()}\nJacobian:\n{jac_mat_d}")
        raise ValueError("QP problem is unfeasible")

    return u, dist


def send_joint_velocity(_dotq):
    # In a real application, this should be replaced by the
    # function that sends the joint velocity _dotq to the robot
    global robot
    global dt
    global sim_time

    sim_time += dt
    robot.add_ani_frame(sim_time, robot.q + _dotq * dt)

def progress_bar(i, imax, bar_length=20, msg="Progress"):
    percent = i / imax
    filled_length = int(np.ceil(bar_length * percent))
    bar = '█' * filled_length + '-' * (bar_length - filled_length)
    print(f'\r{msg}: |{bar}| {percent:.1%}', end='\r')
    if i == imax:
        print()  # Move to the next line on completion

# Simulation
hist_u = []
hist_t = []
hist_dist = []
dist = 0.0
for i in range(round(t_max / dt)):
    progress_bar(i, round(t_max / dt), msg=f"Dist: {dist}")
    # print("\rPercent: " + str(round(100 * sim_time / t_max)))
    q = get_joint_config()
    u, dist = compute_controller(q)
    send_joint_velocity(u)

    hist_u.append(u)
    hist_dist.append(dist)
    hist_t.append(sim_time)

print("Done!")

# Plot the control input of the last joint
def nvim_plot(hist_t, hist_u, hist_dist=None):
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=hist_t,
            y=[u[-1, 0] for u in hist_u],
            mode="lines",
            name="Control input of the last joint",
        )
    )
    fig.update_layout(
        title="Control input of the last joint over time",
        xaxis_title="Time (s)",
        yaxis_title="Control input (rad/s)",
    )
    fig.show()
    if hist_dist is not None:
        fig_dist = go.Figure()
        fig_dist.add_trace(
            go.Scatter(
                x=hist_t,
                y=hist_dist,
                mode="lines",
                name="Minimum distance to obstacles over time",
            )
        )
        fig_dist.update_layout(
            title="Minimum distance to obstacles over time",
            xaxis_title="Time (s)",
            yaxis_title="Distance (m)",
        )
        fig_dist.show()
        return (fig, fig_dist)
    else:
        return fig

fig = nvim_plot(hist_t, hist_u, hist_dist)
# plt.plot(hist_t, [u[-1, 0] for u in hist_u])
# plt.show()


# Save  the simulation to see the results (open the html file control_sim.html that will
# be generated in the same folder the script was ran)
def open_in_browser(filename: str):
    """
    Opens an HTML file in the system's default web browser.
    Works cross-platform (Linux, macOS, Windows).
    """
    path = Path(filename).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    # Convert to file:// URL and open
    webbrowser.open_new_tab(path.as_uri())


file_name = f"control_sim_mode_{mode}"
sim.save(file_name=file_name)
open_in_browser(file_name + ".html")
