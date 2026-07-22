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
from uaibot_cpp_bind import holder_distance, smooth_min, smooth_max
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
        if normal_norm > 1e-1:
            normal /= normal_norm
        else:
            print("small norm")
        normals[i] = normal
    normals = np.vstack([normals, -normals])
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
V_square *= 0.3
normals_square = normals_from_vertices(V_square)
htm_square = np.array(ub.Utils.trn([-1.5 * 0, 0.0, 0.0]))
htm_square = np.array(ub.Utils.trn([-0.9, 0.0, 0.0]))
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
V_pentagon = V_square.copy() * 0.8

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
dt = 1e-3
T = 5.0
# Velocities
omega = (np.pi / 2) / (T / 4)  # 90° rotation mid-movement
omega = np.pi * 2 * 3 / 4
v = 3 / T

T = 2.0  # Change to plot only half, but keep velocities
imax = int(T / dt) + 1

# ------------------------------------------------------------
# 4. Compute signed distance at every time step
# ------------------------------------------------------------
distances = []
euclidean_distances = []
V_square_history = []  # store transformed vertices for plotting
t = []
htm = np.array(htm_square)
x0, y0 = htm[0, -1].item(), htm[1, -1].item()
theta0 = np.acos((np.trace(htm[:3, :3]) - 1) / 2)

# For manim
acquire_manim = False # saves or not epsilon data for manim
var_eps = np.linspace(1e-4, 1e-2, 50)
var_gammas = np.linspace(1, 10, 10)
dists_eps = { v : [] for v in var_eps}
dists_gammas = { v : [] for v in var_gammas}

for i in tqdm(range(imax), total=imax):
    V_sq_t = transform_vertices(V_square, htm)
    N_sq_t = normals_from_vertices(V_sq_t)
    V_square_history.append(V_sq_t)

    dist, *_ = holder_distance(
        V_sq_t, V_pentagon, N_sq_t, normals_pentagon, [], 2, True, epsilon=1e-3
    )
    esdf, *_ = signed_distance_2d(V_sq_t, V_pentagon, N_sq_t, normals_pentagon)
    distances.append(dist)
    euclidean_distances.append(esdf)
    t.append(i * dt)
    htm = make_htm_2d(v * dt * i + x0, 0.0 + y0, omega * dt * i + theta0)

    # For manim
    if acquire_manim:
        for eps_ in var_eps:
            d_, *_ = holder_distance(
                V_sq_t, V_pentagon, N_sq_t, normals_pentagon, [], 2, True, epsilon=eps_
            )
            dists_eps[eps_].append(d_)
        for gamma_ in var_gammas:
            d_, *_ = holder_distance(
                V_sq_t, V_pentagon, N_sq_t, normals_pentagon, [], gamma_, True, epsilon=1e-3
            )
            dists_gammas[gamma_].append(d_)


distances = np.array(distances)
euclidean_distances = np.array(euclidean_distances)

# For manim
if acquire_manim:
    eps_keys   = np.array(var_eps)                         # shape (n_eps,)
    eps_values = np.array([dists_eps[v] for v in var_eps]) # shape (n_eps, imax)
    gamma_keys   = np.array(var_gammas)
    gamma_values = np.array([dists_gammas[v] for v in var_gammas])

    np.savez(
        "./tests/anim_data.npz",
        t=np.array(t),
        V_square_history=np.array(V_square_history),  # shape (N_frames, 4, 3)
        distances=distances,  # hdsdf
        euclidean_distances=euclidean_distances,  # esdf
        V_pentagon=V_pentagon,
        # For manim
        eps_keys=eps_keys,
        eps_values=eps_values,
        gamma_keys=gamma_keys,
        gamma_values=gamma_values,
    )


# %%
data = np.load("./tests/anim_data.npz")
# t_array = data["t"]  # (N_frames,)
# V_square_history = data["V_square_history"]  # (N_frames, 4, 3)
# hdsdf = data["distances"]  # Holder distance
# esdf = data["euclidean_distances"]  # Euclidean signed distance
# V_pentagon = data["V_pentagon"]  # (M, 3)

# For varying curve
epsilons = data["eps_keys"][np.array([0, -1])]
hdsdf_vs_eps = data["eps_values"][np.array([0, -1])]
hd_sdf_eps = {str(e_): hdsdf_vs_eps[i_] for i_, e_ in enumerate(epsilons)}
# epsilons


def make_sdf_figure_callout(
    t,
    euclidean_dist,
    hd_sdf_dist,
    V1_hist,
    V2_hist,
    sep_interval=None,
    pen_interval=None,
    post_interval=None,
    interval_labels=None,
    num_snapshots=6,
    snapshot_scale=0.15,
    poly1_color="rgba(31, 119, 180, 0.5)",
    poly2_color="rgba(255, 127, 14, 0.5)",
    line1_color="blue",
    line2_color="orange",
    font_size=20,
    plot_width=1200,
    plot_height=480,
    hd_sdf_eps=None,
):
    """
    Distance plot with callout boxes for up to three intervals.
    Default labels: "Separation", "Penetration", "Post‑penetration".
    Use `interval_labels` (list of 3 strings) to override.
    """
    fig = go.Figure()

    # ----- 2. Data range -----
    t_min, t_max = np.min(t), np.max(t)
    t_range = t_max - t_min
    min_dist = min(np.nanmin(euclidean_dist), np.nanmin(hd_sdf_dist))
    max_dist = max(np.nanmax(euclidean_dist), np.nanmax(hd_sdf_dist))
    dist_range = max_dist - min_dist

    # ----- 0. Zero line-----
    fig.add_trace(
        go.Scatter(
            x=[t_min, t_max],
            y=[0, 0],
            mode="lines",
            line=dict(width=1.5, dash="dash", color="black"),
            showlegend=False,
        )
    )

    # ----- 1. Distance traces -----
    fig.add_trace(
        go.Scatter(
            x=t,
            y=euclidean_dist,
            mode="lines",
            name="Euclidean SDF",
            line=dict(color="red", width=3),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=t,
            y=hd_sdf_dist,
            mode="lines",
            name="HD‑SDF",
            line=dict(color="blue", width=3),
        )
    )

    if hd_sdf_dist is not None:
        other_colors = [ "lime", "magenta"]
        for idx_, key in enumerate(hd_sdf_eps.keys()):
            eps_name = key
            eps_hd_data = hd_sdf_eps[key]
            color = other_colors[idx_]
            fig.add_trace(
                go.Scatter(
                    x=t,
                    y=eps_hd_data,
                    mode="lines",
                    name=eps_name,
                    line=dict(color=color, width=3, dash='dashdot'),
                )
            )

    # ----- 3. Interval list with labels -----
    default_labels = ["Separation", "Penetration", "Post‑penetration"]
    intervals = []
    for iv, lab in zip([sep_interval, pen_interval, post_interval], default_labels):
        if iv is not None:
            intervals.append((iv, lab))
    if interval_labels is not None and len(interval_labels) >= len(intervals):
        for i, lab in enumerate(interval_labels[: len(intervals)]):
            intervals[i] = (intervals[i][0], lab)

    n_intervals = len(intervals)
    if n_intervals == 0:
        # nothing to highlight – just return the simple distance plot
        return fig

    # ----- 4. Layout helpers for callout placement -----
    margin_x = t_range * 0.02 * 0
    # Callout box dimensions (scaled to data range)
    # callout_height = dist_range * 0.5
    callout_height = dist_range * 0.4
    # Max width per box that still allows them to fit in the plot
    max_total_width = t_range - 2 * margin_x
    # If many intervals, reduce box width so that all fit with gaps
    if n_intervals == 1:
        callout_width = min(t_range * 0.35, max_total_width)
        box_centers = [t_min + margin_x + callout_width / 2]
    else:
        callout_width = min(t_range * 0.45, max_total_width / n_intervals * 0.95)
        gap = (max_total_width - n_intervals * callout_width) / (n_intervals + 1)
        box_centers = []
        x_start = t_min + margin_x + gap
        for i in range(n_intervals):
            box_centers.append(x_start + callout_width / 2 + i * (callout_width + gap))

    # Vertical position: above the distance curves
    top_padding = dist_range * 0.2
    # box_y0 = max_dist + top_padding
    box_y0 = max_dist + abs(max_dist) * 0.2
    box_y1 = box_y0 + callout_height
    y_axis_top = box_y1 + dist_range * 0.2
    # fig.update_yaxes(range=[min_dist - dist_range * 0.1, y_axis_top])

    # Add y_range here in order to plot squares deformed into a square-looking shape,
    # avoiding rectangle look
    y_range = [min_dist - dist_range * 0.1, box_y1 + abs(box_y1) * 0.01]
    x_range = [t[0], t[-1]]
    xy_ratio = (x_range[1] - x_range[0]) / (y_range[1] - y_range[0])
    y_span = y_range[1] - y_range[0]
    x_span = x_range[1] - x_range[0]
    y_correction = (plot_width * y_span) / (plot_height * x_span)
    print(f"xrange: {x_range}, y_range: {y_range}")
    print(f"xy_ratio: {xy_ratio}")

    # ----- 6. Create a callout for one interval -----
    def add_callout(interval, box_center, label):
        box_x0 = box_center - callout_width / 2
        box_x1 = box_center + callout_width / 2

        i0 = np.where(t == interval[0])[0].item()
        i1 = np.where(t == interval[1])[0].item()
        max_dist_interval = max(
            np.max(euclidean_distances[i0:i1]), np.max(distances[i0:i1])
        )
        y1 = max(0.01 * 0, max_dist_interval)
        y1 = y1 + abs(y1) * 0.1
        min_dist_interval = min(
            np.min(euclidean_distances[i0:i1]), np.min(distances[i0:i1])
        )
        y0 = min(-0.01 * 0, min_dist_interval)
        y0 = y0 - abs(y0) * 0.1

        # a) Highlight rectangle covering full data y‑range
        fig.add_shape(
            type="rect",
            x0=interval[0],
            y0=y0,
            x1=interval[1],
            y1=y1,
            line=dict(color="grey", width=1.5, dash="dot"),
            fillcolor="lightgrey",
            opacity=0.4,
        )

        # b) White background trace (sits behind the polygons)
        rect_x = [box_x0, box_x1, box_x1, box_x0, box_x0]
        rect_y = [box_y0, box_y0, box_y1, box_y1, box_y0]
        fig.add_trace(
            go.Scatter(
                x=rect_x,
                y=rect_y,
                fill="toself",
                fillcolor="white",
                mode="lines",
                line=dict(width=1.5, color="grey"),
                showlegend=False,
                hoverinfo="skip",
            )
        )

        # c) Dashed connecting lines
        fig.add_shape(
            type="line",
            x0=interval[0],
            y0=y1,
            x1=box_x0,
            y1=box_y0,
            line=dict(color="grey", width=1, dash="dot"),
        )
        fig.add_shape(
            type="line",
            x0=interval[1],
            y0=y1,
            x1=box_x1,
            y1=box_y0,
            line=dict(color="grey", width=1, dash="dot"),
        )

        # d) Snapshots
        i0 = np.argmin(np.abs(t - interval[0]))
        i1 = np.argmin(np.abs(t - interval[1]))
        if i1 <= i0:
            i1 = i0 + 1
        # snap_times = np.linspace(t[i0], t[i1], num_snapshots + 2)[1:-1]
        snap_times = np.linspace(t[i0], t[i1], num_snapshots)

        for k, ts in enumerate(snap_times):
            idx = np.argmin(np.abs(t - ts))
            V1 = np.asarray(V1_hist[idx])
            V2 = np.asarray(V2_hist[idx])
            combined = np.vstack([V1, V2])
            centroid = combined.mean(axis=0) if len(combined) > 0 else np.array([0, 0])

            # scale down
            def transform(V):
                return centroid + (V - centroid) * snapshot_scale

            V1s = transform(V1)
            V2s = transform(V2)
            # 2. Aspect‑ratio correction to force visual squareness
            #    stretch x‑coordinates around the centroid by xy_ratio
            V1s[:, 1] = centroid[1] + (V1s[:, 1] - centroid[1]) * y_correction
            V2s[:, 1] = centroid[1] + (V2s[:, 1] - centroid[1]) * y_correction

            # place in box
            target_x = box_x0 + (k + 0.5) * (box_x1 - box_x0) / num_snapshots
            target_y = (box_y0 + box_y1) / 2
            shift = np.array([target_x, target_y]) - centroid
            V1t = V1s + shift
            V2t = V2s + shift

            # polygons
            V1c = np.vstack([V1t, V1t[0]])
            V2c = np.vstack([V2t, V2t[0]])
            fig.add_trace(
                go.Scatter(
                    x=V1c[:, 0],
                    y=V1c[:, 1],
                    mode="lines",
                    fill="toself",
                    fillcolor=poly1_color,
                    line=dict(color=line1_color, width=1),
                    showlegend=False,
                    hoverinfo="skip",
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=V2c[:, 0],
                    y=V2c[:, 1],
                    mode="lines",
                    fill="toself",
                    fillcolor=poly2_color,
                    line=dict(color=line2_color, width=1),
                    showlegend=False,
                    hoverinfo="skip",
                )
            )

            # Time label below snapshot
            fig.add_annotation(
                x=target_x,
                y=box_y0 + callout_height * 0.25,
                text=f"t = {ts:.2f}",
                showarrow=False,
                font=dict(size=font_size, color="black"),
                xanchor="center",
                yanchor="top",
            )

    # ----- 7. Place all callouts -----
    for (interval, label), center in zip(intervals, box_centers):
        add_callout(interval, center, label)

    # ----- 8. Axes styling (unchanged) -----
    fig.update_xaxes(
        title_text="Time (seconds)",
        showline=True,
        linewidth=1,
        linecolor="black",
        mirror=True,
        automargin=False,
        title_standoff=15,
    )
    fig.update_yaxes(
        title_text="Distance",
        showline=True,
        linewidth=1,
        linecolor="black",
        mirror=True,
        automargin=False,
        title_standoff=15,
        range=y_range,
        # Workaround for dashed zeroline
        zeroline=True,
        zerolinecolor="white",
        zerolinewidth=1.5,
    )
    fig.add_shape(
        type="rect",
        xref="paper",
        yref="paper",
        x0=0,
        y0=0,
        x1=1,
        y1=1,
        line=dict(color="black", width=2),
    )
    fig.update_layout(
        template="plotly_white",
        hovermode="x unified",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=0.01,
            xanchor="left",
            x=0.01,
            bgcolor="rgba(255,255,255,0.7)",
        ),
        font=dict(size=font_size),
        margin=dict(l=90, r=10, t=10, b=90, pad=0),
        width=plot_width,
        height=plot_height,
    )
    return fig


V_pentagon_hist_2d = [V_pentagon[:, :2] for _ in range(len(V_square_history))]
V_square_hist_2d = [np.array([v[:2] for v in verts]) for verts in V_square_history]

fig = make_sdf_figure_callout(
    np.array(t),
    euclidean_distances,
    distances,  # your HD‑SDF
    V_square_hist_2d,
    V_pentagon_hist_2d,
    # sep_interval=(0, 0.1),  # debugging
    # pen_interval=(0.5, 1.0), # debugging
    sep_interval=(0.3, 0.668),  # e.g., seconds where squares are parallel
    pen_interval=(1.332, 1.668),  # e.g., penetration with multiple nearest edges
    # post_interval=(4.335, 4.83),
    # pen_interval=(2.25, 2.75),  # pentagon
    # post_interval=(4.25, 4.75), # pentagon
    num_snapshots=3,
    snapshot_scale=0.22,
    hd_sdf_eps=hd_sdf_eps,
)

fig.show()
fig.write_image("distance_comparison_example_square.pdf", width=1200, height=480)
# %%
# ------------------------------------------------------------
# 6. Plot 2: movement snapshots
# ------------------------------------------------------------
# snapshot_times = np.array([0.0, T/4, T])
snapshot_times = np.array([0.0, 0.34, 4.68])
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
