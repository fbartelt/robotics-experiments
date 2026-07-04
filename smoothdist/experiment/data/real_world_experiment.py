#!/usr/bin/env python
# coding: utf-8

# ## 1. Set up communication with the FR3
# 
# Works in the docker environment.

# ### 1.1 Set up communication with the arm

# In[1]:


get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
import sys 
from FR3Py.robot.interface import FR3Real
real_robot_interface = FR3Real(robot_id='fr3_rob1')


# In[2]:


# Print the arm state
real_robot_interface.getJointStates()


# # 2. Go to initial joint angle

# In[3]:


import numpy as np
import time


# In[4]:


## joint bounds
joint_lb = np.array([-2.3093, -1.5133, -2.4937, -2.7478, -2.48, 0.8521, -2.6895])
joint_ub = np.array([2.3093, 1.5133, 2.4937, -0.4461, 2.48, 4.2094, 2.6895])
joint_vel_lb = -np.array([2.175, 2.175, 2.175, 2.175, 2.61, 2.61, 2.61])
joint_vel_ub = np.array([2.175, 2.175, 2.175, 2.175, 2.61, 2.61, 2.61])


# In[5]:


T = 7
q_init = np.array([1.0582, -1.3811, 0.3629, -1.9647, -0.959, 1.4881, -0.1534])

Kp_joint = np.diag([1, 1, 1, 1, 4, 2, 1])*5
t_start = time.time()
while time.time() - t_start < T:
    robot_info = real_robot_interface.getJointStates()
    q = robot_info['q'] # shape (7,)

    W = np.diag(1.0/(joint_ub-joint_lb))
    eq = W @ (q - q_init)
    dq = - Kp_joint @ eq
    dq = np.clip(dq, joint_vel_lb, joint_vel_ub)
    real_robot_interface.setCommands(dq)

real_robot_interface.setCommands(np.zeros_like(dq))


# # 2. Define UAIBot environment

# In[6]:


import uaibot as ub
from create_franka_emika_3_mod import create_franka_emika_3_mod


# In[7]:


# Operation mode, mode = 0 (Euclidean) and mode = 1 (Our proposed distance)
mode = 1
is_conservative = False

if mode == 0:
    h = 1e-6
    eps = 0
    delta_obs = 0.03
    delta_auto = 1e-4
else:
    gamma = 2
    epsilon = 9e-4
    # This was the min distance to the expanded obstacles when
    delta_obs = -5.0e-4 * 1e1
    delta_auto = -1e-4


# In[8]:


# Obstacles
real_obstacles = []
real_obstacles.append(
    ub.Box(
        htm=ub.Utils.trn([0.53, 0.16, 0.45]),
        width=0.35,
        depth=0.05,
        height=0.90,
        color="magenta",
    )
)
real_obstacles.append(
    ub.Box(
        htm=ub.Utils.trn([0.53, -0.16, 0.45]),
        width=0.35,
        depth=0.05,
        height=0.90,
        color="magenta",
    )
)
real_obstacles.append(
    ub.Box(
        htm=ub.Utils.trn([0.53, 0.00, 0.925]),
        width=0.35,
        depth=0.35,
        height=0.05,
        color="magenta",
    )
)

# Agumented obstacles
obstacles = []
expand = 0.1
expand = 5e-2

obstacles.append(
    ub.Box(
        htm=ub.Utils.trn([0.53, 0.16, 0.45]),
        width=0.35 + expand,
        depth=0.05 + expand,
        height=0.90 + expand,
        color="cyan",
        opacity=0.7,
    )
)
obstacles.append(
    ub.Box(
        htm=ub.Utils.trn([0.53, -0.16, 0.45]),
        width=0.35 + expand,
        depth=0.05 + expand,
        height=0.90 + expand,
        color="cyan",
        opacity=0.7,
    )
)
obstacles.append(
    ub.Box(
        htm=ub.Utils.trn([0.53, 0.00, 0.925]),
        width=0.35 + expand,
        depth=0.35 + expand,
        height=0.05 + expand,
        color="cyan",
        opacity=0.7,
    )
)


# In[9]:


# Target pose
htm_tg = (
    ub.Utils.trn([0.64, 0, 0.75]) * ub.Utils.roty(np.pi / 2) * ub.Utils.rotz(np.pi / 2)
)


# In[10]:


# Control matrix for the task function (1/second)
K = np.diag([0.4, 0.4, 0.4, 0.4, 0.4, 0.4])

# Regularization factor for the task function
reg = 0.01

# Gain for the CBF inequality (1/second)
eta = 0.5

# Maximum experiment time (seconds)
t_max = 35

# Maximum number of iterations for the generalized Von Neumman's algorithm
no_iter_max = 300

# Tolerance for convergence for the generalized Von Neumman's algorithm
tol = 2e-4


# In[11]:


exp_time = 0
robot = create_franka_emika_3_mod()

if mode == 0:
    obstacles = real_obstacles


# In[12]:


# Auxiliary functions


def get_joint_config():
    robot_state = real_robot_interface.getJointStates()
    return robot_state['q']


def send_joint_velocity(_dotq):
    real_robot_interface.setCommands(_dotq)


err = False


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
    global err

    # Get the number of configurations
    n = np.shape(_q)[0]
    dist = 0.0

    # Initialize matrices A and b
    mat_A = np.matrix(np.zeros((0, n)))
    mat_b = np.matrix(np.zeros((0, 1)))

    # Implement obstacle avoidance constraints and stack into A and b
    dist = np.inf
    for obs in obstacles:
        if mode == 0:
            dr = robot.compute_dist(
                q=_q, obj=obs, h=h, eps=eps, no_iter_max=no_iter_max, tol=tol
            )
        else:
            dr = robot.signed_distance(
                obj=obs,
                q=_q,
                gamma=gamma,
                is_conservative=is_conservative,
                epsilon=epsilon,
                eps_edge=-1,
            )
        mat_A = np.vstack((mat_A, dr.jac_dist_mat))
        mat_b = np.vstack((mat_b, -eta * (dr.dist_vect - delta_obs)))
        dist = min(dist, np.min(dr.dist_vect))


    dist_vect_d = dr.dist_vect
    jac_mat_d = dr.jac_dist_mat
    # Implement auto-collision avoidance and stack into A and b
    if mode == 0:
        dr = robot.compute_dist_auto(
            q=_q, h=0.1, eps=0.01, no_iter_max=no_iter_max, tol=tol
        )
    else:
        dr = robot.signed_distance_auto(
            q=_q,
            gamma=gamma,
            is_conservative=is_conservative,
            epsilon=epsilon,
            eps_edge=-1,
        )
    auto_dist = np.min(dr.dist_vect)
    dist_vect_auto_d = dr.dist_vect
    jac_mat_auto_d = dr.jac_dist_mat

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
        # print(jac_mat_d)
    except:
        u = 0 * _q
        print("Unfeasible!")
        err = True
        raise ValueError("QP problem is unfeasible")

    real_dist = -1  # placeholder
    return u, (dist, auto_dist, real_dist)


# In[13]:


def progress_bar(percent, bar_length=20, msg="Progress"):
    filled_length = int(np.ceil(bar_length * percent))
    bar = "█" * filled_length + "-" * (bar_length - filled_length)
    print(f"\r{msg}: |{bar}| {percent:.1%}", end="\r")
    if percent > 0.999:
        print()  # Move to the next line on completion


# In[14]:


hist_u = []
timestamps = []
time_per_loop = []
hist_dist = []
hist_real_dist = []
hist_auto_dist = []
hist_q = []
dist = 0.0

start_time = time.time()
while (time.time() - start_time < t_max):
    loop_start = time.time()
    progress_bar((time.time() - start_time)/t_max, msg=f"Dist: {dist}")
    q = get_joint_config()[:,None]
    u, (dist, auto_dist, real_dist) = compute_controller(q)
    send_joint_velocity(np.asarray(u).flatten())
    loop_end = time.time()

    timestamps.append(loop_start)
    time_per_loop.append(loop_end-loop_start)
    hist_u.append(u)
    hist_dist.append(dist)
    hist_real_dist.append(real_dist)
    hist_auto_dist.append(auto_dist)
    hist_q.append(q)

print("Done!")


# # 3. Save the data

# In[15]:


import pickle
from datetime import datetime
import os

now = datetime.now()
formatted_date_time = now.strftime("%Y-%m-%d-%H-%M-%S")

directory = f"mode_{mode}/{formatted_date_time}"

# Create the directory if it doesn't exist
if not os.path.exists(directory):
    os.makedirs(directory)

# Define the file path
file_path = os.path.join(directory, "data.pickle")

# Data to be saved
data = {
    "q": hist_q,
    "timestamp": timestamps,
    "time_per_loop": time_per_loop,
    "hist_dq": hist_u,
    "hist_dist": hist_dist,
    "hist_real_list": hist_real_dist,
    "hist_auto_dist": hist_auto_dist
}

# Save the data using pickle
with open(file_path, "wb") as f:
    pickle.dump(data, f)


# # 4. Visualization the data

# In[16]:


import matplotlib.pyplot as plt


# In[17]:


with open(file_path, "rb") as f:
    data = pickle.load(f)


# In[18]:


time_per_loop = data['time_per_loop']
time_per_loop_np = np.array(time_per_loop)
plt.hist(time_per_loop_np, bins=100)
plt.show()


# In[19]:


hist_q = data['q']
plt.plot(np.cumsum(time_per_loop), [u[-1, 0] for u in hist_q])


# In[20]:


hist_u = data['hist_dq']
plt.plot(np.cumsum(time_per_loop), [u[-1, 0] for u in hist_u])

