# %%
import sys
import os
import pickle
import itertools
import numpy as np
import uaibot as ub
import plotly.graph_objects as go
import webbrowser
import tqdm
from time import time, perf_counter

sys.path.insert(0, "/home/fbartelt/Projects/robotics-experiments/smoothdist")
sys.path.insert(0, "/home/fbartelt/Documents/Projetos/robotics-experiments/smoothdist/")
from euclidean_sdf import esdf, compute_vertices_and_faces
from scipy.spatial import ConvexHull
from pathlib import Path
from plotly.subplots import make_subplots
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
from uaibot_cpp_bind import holder_distance
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.optimize import minimize_scalar


def signed_distance_2d(V1, V2, normals1, normals2):
    """
    Signed distance d(A,B) = max_{||n||=1} min_{a∈A, b∈B} n·(a−b).
    Positive if separated, negative if overlapping.
    normals1, normals2 are ignored (kept for interface compatibility).
    """
    # Use only the first two coordinates (z is always 0)
    A = V1[:, :2]  # shape (N1, 2)
    B = V2[:, :2]  # shape (N2, 2)

    # Pairwise difference tensor: (N1, N2, 2)
    diff = A[:, None, :] - B[None, :, :]

    def f(theta):
        """f(θ) = min_{a,b} n·(a-b) with n = [cosθ, sinθ]."""
        n = np.array([np.cos(theta), np.sin(theta)])
        dots = np.tensordot(diff, n, axes=([2], [0]))  # shape (N1, N2)
        return np.min(dots)

    # 1. Brute-force global search over a fine grid
    theta_grid = np.linspace(0, 2 * np.pi, 2000)
    f_vals = np.array([f(th) for th in theta_grid])
    best_idx = np.argmax(f_vals)
    theta0 = theta_grid[best_idx]

    # 2. Local refinement around the best candidate
    #    Bounds: a small window around θ0, wrapped for periodicity.
    margin = np.pi / 100
    low = theta0 - margin
    high = theta0 + margin
    # If window crosses 0 or 2π, shift the interval
    if low < 0:
        low += 2 * np.pi
        high += 2 * np.pi
        theta0 += 2 * np.pi
    elif high > 2 * np.pi:
        low -= 2 * np.pi
        high -= 2 * np.pi
        theta0 -= 2 * np.pi

    res = minimize_scalar(lambda th: -f(th), bounds=(low, high), method="bounded")
    theta_opt = res.x
    dist = f(theta_opt)

    # Dummy closest points (not needed)
    return dist, np.zeros(3), np.zeros(3)


# We will represent 2d objects as 3d objects with infinitesimal height.
def normals_from_vertices(vertices):
    """This function assumes that vertices are ordered and that all
    normals point outwards. Vertices are assumed 3D with z=0.
    """
    n = len(vertices)
    normals = np.zeros_like(vertices)
    for i in range(n):
        v1 = vertices[i]
        v2 = vertices[(i + 1) % n]
        edge = v2 - v1
        normal = np.array([-edge[1], edge[0], 0.0])
        normal_norm = np.linalg.norm(normal)
        if normal_norm > 1e-8:
            normal /= normal_norm
        normals[i] = normal
    return normals


def make_htm_2d(x, y, theta):
    """4x4 homogeneous transform for translation (x,y) and rotation theta (z-axis)."""
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s, 0, x], [s, c, 0, y], [0, 0, 1, 0], [0, 0, 0, 1]])


def transform_vertices(V, htm):
    """Apply a 4x4 HTM to an (N,3) array of vertices."""
    N = V.shape[0]
    Vh = np.column_stack([V, np.ones(N)])  # (N,4)
    V_trans = (htm @ Vh.T).T  # (N,4)
    return V_trans[:, :3]  # back to (N,3)


# Square centered at (0, 0) with lenght 1
V_square = np.array(
    [
        [-0.5, -0.5, 0.0],
        [-0.5, 0.5, 0.0],
        [0.5, 0.5, 0.0],
        [0.5, -0.5, 0.0],
    ]
)
# R = np.array(ub.Utils.rotz(np.pi / 4))[:3, :3]
# V_square = V_square @ R.T
normals_square = normals_from_vertices(V_square)
htm_square = np.array(ub.Utils.trn([-1.5*0, 0., 0.]))
htm_square = np.array(ub.Utils.trn([-1.5, 0., 0.]))
# Pentagon centered at (0, 0) with radius 1
s1 = np.sin(2 * np.pi / 5)
s2 = np.sin(4 * np.pi / 5)
c1 = np.cos(2 * np.pi / 5)
c2 = np.cos(np.pi / 5)
V_pentagon = (
    np.array(
        [
            [0.0, 1.0, 0.0],
            [s1, c1, 0.0],
            [s2, -c2, 0.0],
            [-s2, -c2, 0.0],
            [-s1, c1, 0.0],
        ]
    )
    * 0.5
)
# make 'last' edge parallel to square
R = np.array(ub.Utils.rotz(np.deg2rad(180 - 126)))[:3, :3]
V_pentagon = V_pentagon @ R.T

# Test two squares first
# V_pentagon = V_square.copy() * 0.8

normals_pentagon = normals_from_vertices(V_pentagon)
htm_pentagon = np.eye(4)

# check angles of normals wrt to +x axis
for n in normals_pentagon:
    angle = np.arctan2(n[1], n[0]) * 180 / np.pi
    print(f"Pentagon normal angle: {angle:.2f} degrees")

# %%
# ------------------------------------------------------------
# 3. Define the square’s motion over time (XY plane only)
# ------------------------------------------------------------
# t_start, t_end = 0.0, 5.0
# n_frames = 200
# t = np.linspace(t_start, t_end, n_frames)
#
# # Translation: start left of the pentagon, move to the right
# x_start, y_start = -2.5, 0.0
# x_end, y_end = 2.5, 0.0
#
# x_end = -1.5
# x_start = x_end
#
# # Rotation: continuous rotation (0 to 2π)
# theta_start, theta_end = 0.0, 6 * np.pi
#
# x = np.interp(t, [t_start, t_end], [x_start, x_end])
# y = np.interp(t, [t_start, t_end], [y_start, y_end])
# theta = np.interp(t, [t_start, t_end], [theta_start, theta_end])
# ------------------------------------------------------------
# 4. Compute signed distance at every time step
# ------------------------------------------------------------
distances = []
euclidean_distances = []
V_square_history = []  # store transformed vertices for plotting
t = []
dt = 1e-2
htm = np.array(htm_square)
x0, y0 = htm[0, -1].item(), htm[1, -1].item()
theta0 = np.acos((np.trace(htm[:3, :3]) - 1) / 2)
T = 5.
imax = int(T / dt)
omega = (np.pi / 2) / (T / 4) # 90° rotation mid-movement
omega = np.pi * 2 * 3 / 4
v = 3 / T

for i in tqdm(range(imax), total=imax):
    V_sq_t = transform_vertices(V_square, htm)
    N_sq_t = transform_vertices(normals_square, htm)
    V_square_history.append(V_sq_t)

    dist, *_ = holder_distance(
        V_sq_t, V_pentagon, N_sq_t, normals_pentagon, [], 2, True, 1e-6
    )
    esdf, *_ = signed_distance_2d(V_sq_t, V_pentagon, N_sq_t, normals_pentagon)
    distances.append(dist)
    euclidean_distances.append(esdf)
    t.append(i * dt)
    htm = make_htm_2d(v * dt * i + x0, 0.0 + y0, omega * dt * i + theta0)

distances = np.array(distances)
euclidean_distances = np.array(euclidean_distances)


# ------------------------------------------------------------
# 5. Plot 1: distance vs. time (static)
# ------------------------------------------------------------
def plot_dist(t, distances, euclidean_distances):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=t, y=distances, name="HD-SDF", mode="lines"))
    fig.add_trace(go.Scatter(x=t, y=euclidean_distances, name="SDF", mode="lines"))
    return fig


fig = plot_dist(t, distances, euclidean_distances)
fig.show()
# %%
# ------------------------------------------------------------
# 6. Plot 2: movement snapshots
# ------------------------------------------------------------
snapshot_times = np.array([0.0, T/4, T])
fig, axes = plt.subplots(1, 4, figsize=(16, 4))

for ax, snap_t in zip(axes, snapshot_times):
    idx = np.argmin(np.abs(t - snap_t))
    V_sq = V_square_history[idx]
    dist = distances[idx]

    # Draw pentagon
    pent = np.vstack([V_pentagon, V_pentagon[0]])  # close loop
    ax.plot(pent[:, 0], pent[:, 1], "g-", linewidth=2, label="Pentagon")
    # Draw square
    sq = np.vstack([V_sq, V_sq[0]])
    ax.plot(sq[:, 0], sq[:, 1], "r-", linewidth=2, label="Square")

    ax.set_aspect("equal")
    ax.set_title(f"t = {snap_t:.1f}, dist = {dist:.3f}")
    ax.grid(True)
    ax.legend(loc="upper right")
    ax.set_xlim(-3.5, 3.5)
    ax.set_ylim(-2, 2)

plt.suptitle("Square moving (and rotating) toward the pentagon", fontsize=14)
plt.tight_layout()
# plt.savefig('snapshots.png', dpi=150)
plt.show()

# %%
# ------------------------------------------------------------
# 7. (Optional) Animation – uncomment to see a live animation
# ------------------------------------------------------------
fig_anim, ax_anim = plt.subplots(figsize=(6, 6))
pent = np.vstack([V_pentagon, V_pentagon[0]])
ax_anim.plot(pent[:, 0], pent[:, 1], "g-", lw=2, label="Pentagon")
(sq_line,) = ax_anim.plot([], [], "r-", lw=2, label="Square")
time_text = ax_anim.text(0.02, 0.95, "", transform=ax_anim.transAxes)
ax_anim.set_xlim(-3.5, 3.5)
ax_anim.set_ylim(-2, 2)
ax_anim.set_aspect("equal")
ax_anim.grid(True)
ax_anim.legend()


def init():
    sq_line.set_data([], [])
    time_text.set_text("")
    return sq_line, time_text


def animate(i):
    V_sq = V_square_history[i]
    sq = np.vstack([V_sq, V_sq[0]])
    sq_line.set_data(sq[:, 0], sq[:, 1])
    time_text.set_text(f"t = {t[i]:.2f}, dist = {distances[i]:.3f}")
    return sq_line, time_text


ani = FuncAnimation(
    fig_anim,
    animate,
    frames=imax,
    init_func=init,
    blit=True,
    interval=dt,
    repeat=True,
)
plt.show()
