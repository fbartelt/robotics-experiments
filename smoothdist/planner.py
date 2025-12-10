# %%
import numpy as np
import plotly.graph_objects as go
import plotly.colors as pc

from distances import signed_dist2convex, phi, smooth_min
from polygon import (
    Polytope,
    add_polygon,
    create_level_sets,
    NonConvexPolygon,
    generate_random_polyhedron,
    generate_random_polyhedron_set,
)
from polyhedron import add_polyhedron


def add_path(
    fig, path_hist, num_paths, base_color="#000000", q0_symbol="square", qd_symbol="x"
):
    base_color = "#000000"
    base_color_rgb = pc.convert_colors_to_same_type(base_color, colortype="rgb")[0][0]
    base_color_rgba = (
        base_color_rgb.replace(" ", "").replace(")", "").replace("rgb", "rgba")
    )

    # Adds num_paths paths from path_hist to the figure
    # each path is colored with a gradient from base_color 0.1 opacity to base_color 1.0 opacity
    # q0 and qd are marked with different symbols
    # paths2add = path_hist[:: max(1, len(path_hist) // num_paths)]
    idxs2add = np.round(
        np.linspace(0, len(path_hist) - 1, num=min(num_paths, len(path_hist)))
    ).astype(int)
    paths2add = np.array(path_hist)[idxs2add]
    print(f"Adding {len(paths2add)} paths to the figure.")
    for i, path in enumerate(paths2add):
        alpha = 0.1 + (0.5 * i) / (len(paths2add) - 1) if len(paths2add) > 1 else 1.0
        alpha = alpha if i < len(paths2add) - 1 else 1.0
        color = base_color_rgba + f", {alpha})"
        fig.add_trace(
            go.Scatter(
                x=path[0, :],
                y=path[1, :],
                mode="lines+markers",
                line=dict(color=color, width=2),
                marker=dict(size=6, color=color),
                name=f"Path {i+1}",
                showlegend=False,
            )
        )
        # Mark q0
        fig.add_trace(
            go.Scatter(
                x=[path[0, 0]],
                y=[path[1, 0]],
                mode="markers",
                marker=dict(symbol=q0_symbol, size=10, color=color),
                name="Start",
                showlegend=(i == len(paths2add) - 1),
            )
        )
        # Mark qd
        fig.add_trace(
            go.Scatter(
                x=[path[0, -1]],
                y=[path[1, -1]],
                mode="markers",
                marker=dict(symbol=qd_symbol, size=10, color=color),
                name="Goal",
                showlegend=(i == len(paths2add) - 1),
            )
        )

def add_path3d(
    fig, path_hist, num_paths, base_color="#000000", q0_symbol="square", qd_symbol="x"
):
    base_color = "#000000"
    base_color_rgb = pc.convert_colors_to_same_type(base_color, colortype="rgb")[0][0]
    base_color_rgba = (
        base_color_rgb.replace(" ", "").replace(")", "").replace("rgb", "rgba")
    )

    # Adds num_paths paths from path_hist to the figure
    # each path is colored with a gradient from base_color 0.1 opacity to base_color 1.0 opacity
    # q0 and qd are marked with different symbols
    # paths2add = path_hist[:: max(1, len(path_hist) // num_paths)]
    idxs2add = np.round(
        np.linspace(0, len(path_hist) - 1, num=min(num_paths, len(path_hist)))
    ).astype(int)
    paths2add = np.array(path_hist)[idxs2add]
    print(f"Adding {len(paths2add)} paths to the figure.")
    for i, path in enumerate(paths2add):
        alpha = 0.1 + (0.5 * i) / (len(paths2add) - 1) if len(paths2add) > 1 else 1.0
        alpha = alpha if i < len(paths2add) - 1 else 1.0
        color = base_color_rgba + f", {alpha})"
        marker_size = 4
        fig.add_trace(
            go.Scatter3d(
                x=path[0, :],
                y=path[1, :],
                z=path[2, :],
                mode="lines+markers",
                line=dict(color=color, width=4),
                marker=dict(size=2, color=color),
                name=f"Path {i+1}",
                showlegend=False,
            )
        )
        # Mark q0
        fig.add_trace(
            go.Scatter3d(
                x=[path[0, 0]],
                y=[path[1, 0]],
                z=[path[2, 0]],
                mode="markers",
                marker=dict(symbol=q0_symbol, size=marker_size, color=color),
                name="Start",
                showlegend=(i == len(paths2add) - 1),
            )
        )
        # Mark qd
        fig.add_trace(
            go.Scatter3d(
                x=[path[0, -1]],
                y=[path[1, -1]],
                z=[path[2, -1]],
                mode="markers",
                marker=dict(symbol=qd_symbol, size=marker_size, color=color),
                name="Goal",
                showlegend=(i == len(paths2add) - 1),
            )
        )


def deform_path(
    init_path,
    obstacles,
    kind="in",
    h=0.01,
    r=0.1,
    min_path=False,
    zeta=1.0,
    alpha=1.0,
):
    if zeta == 0:
        min_path = False
    path = init_path.copy()
    N = path.shape[1]  # number of points in the path
    m = path.shape[0]  # dimension of the space
    n_obstacles = len(obstacles)
    dists, grads = np.zeros((N,)), np.zeros((N, m))
    for j, p_ in enumerate(init_path.T):
        p = p_.copy().reshape(-1, 1)
        dists_, grads_ = np.zeros((n_obstacles,)), np.zeros((n_obstacles, m))
        for i, obstacle in enumerate(obstacles):
            A, b = obstacle.A, obstacle.b
            # Compute each d_S and gradient (1 x m)
            dist_, grad_ = signed_dist2convex(
                phi, p, A, b, r=r, h=h, test=kind, compute_gradient=True
            )
            dists_[i] = np.round(dist_, 6)
            if np.round(dist_, 4) >= 0:
                grads_[i, :] = grad_.ravel()
            else:
                grads_[i, :] = grad_.ravel()
        # Compute D_O and gradient (1 x n_obstacles)
        dist, grad = smooth_min(dists_, r=r, compute_gradient=True)
        grad_sat = 1 / (1 + np.exp(alpha * dist))
        dist = (-1 / alpha) * np.log(
            0.5 * (1 + np.exp(-alpha * dist))
        )  # Smooth saturation
        grad_full = grad_sat * grad @ grads_  # (1 x m)
        # idx_min = np.argmin(dists_)
        # grad_full = grads_[idx_min, :].reshape(1, -1)
        dists[j] = dist
        grads[j, :] = grad_full.ravel()

    for j in range(path.shape[1] - 2):
        k = j + 1  # Do not change initial and final point
        # const_obs = [b - A @ path[:, j].reshape(-1, 1) for A, b in obstacles]
        # err = np.max(const_obs)
        # coeff = np.abs(dists[j])  # np.sign(dists[j]) * np.sqrt(np.abs(dists[j]))
        coeff = 1.0
        if dists[k] > 0:
            coeff = 1.0
            # coeff = np.sqrt(np.abs(dists[k]))
        else:
            coeff = np.sqrt(np.abs(dists[k]))
        path[:, k] += grads[k] * coeff

    if min_path:
        for j, point in enumerate(path.T[1:-1]):
            k = j + 1
            prev_grad = path[:, k - 1].ravel() - point.ravel()
            next_grad = path[:, k + 1].ravel() - point.ravel()
            path[:, k] = point + zeta * (prev_grad + next_grad)

    return path, dists, grads


# %%
max_polygons = 8  # 8 (1337), 5 (1001)
max_vertices = 15
bounding_box = (-20.0, -20, 20, 20)
# Distance between vertices will be at least 2*first element, and at most
# 2*second element of radius_limits:
radius_limits = (2, 6)
q0 = np.array([-3.0, -12]).reshape(-1, 1)
q0 = np.array([-8.5, -16]).reshape(-1, 1)  # 1001
q0 = np.array([-17.0, -17]).reshape(-1, 1)  # 1337
qd = np.array([18, 15]).reshape(-1, 1)
# qd = np.array([11.0, 15]).reshape(-1, 1)
n_points = 100
h = 0.01
r = 0.1
zeta = 0.5
alpha = np.log(2) / 0.2
min_path = True
max_attempts = 500
seed = 1337  # 1001, 69 cool, 42 NICE post mods, 100 is cool
min_area = None
radius = None
num_vertices = None

obstacles = Polytope.random_set(
    n_polytopes=max_polygons,
    intersect_polytopes=False,
    q0=q0,
    qd=qd,
    max_vertices=max_vertices,
    radius_lim=radius_limits,
    bbox=bounding_box,
    seed=seed,
    min_area=min_area,
    max_attempts=max_attempts,
    radius=radius,
    num_vertices=num_vertices,
)

lambda_ = np.linspace(0, 1, n_points)
init_path = (1 - lambda_) * q0 + lambda_ * qd
path = init_path.copy()
dists = [-100]
iter_ = 0
path_hist = [init_path.copy()]
max_iters = 1500
kind = "in"
kind = "out"
kind = None

# bounding_box = (-20.0, -20, 20, 20.11) # 1337 plotting related
# fig = go.Figure()
fig = create_level_sets(
    obstacles,
    r=r,
    h=h,
    kind="both",
    bbox=bounding_box,
    n_points=200,
    n_contours=40,
    add_reference=False,
    test=None,
    rescale=True,
)
# for obstacle in obstacles:
#     add_polygon(fig, obstacle.A, obstacle.b, add_reference=False)

# obstacles = [obstacles[0]]
while np.any(np.array(dists) < 0.0):
    if iter_ >= max_iters:
        print(f"reached max iterations: {max_iters}")
        break

    path, dist, grad = deform_path(
        path, obstacles, kind=kind, h=h, r=r, min_path=min_path, zeta=zeta, alpha=alpha
    )
    path_hist.append(path.copy())
    dists = dist
    if iter_ % 10 == 0:
        print(f"iteration {iter_}: min dist = {np.min(dists)}")

    iter_ += 1

print(f"deformation completed in {iter_} iterations with min dist = {np.min(dists)}")
add_path(fig, path_hist, num_paths=6, base_color="black")
fig.update_layout(width=1200, height=800)
fig.show()

# fig.write_image(f"path_seed_{seed}_maxpoly_{max_polygons}.pdf")


# [jump]

# %%
n_polyhedra = 12
max_vertices = 15
bounding_box = (-20.0, -20, -20, 20, 20, 20)
# Distance between vertices will be at least 2*first element, and at most
# 2*second element of radius_limits:
radius_limits = (2, 10)
q0 = np.array([-1., -15, 10.]).reshape(-1, 1)
qd = np.array([18., 18, -18.]).reshape(-1, 1)
# qd = np.array([11.0, 15]).reshape(-1, 1)
n_points = 100
h = 0.01
r = 0.1
zeta = 0.5
alpha = np.log(2) / 0.2
min_path = True
max_attempts = 500
seed = 1001  # 1001, 69 cool, 42 NICE post mods, 100 is cool
min_volume = 4/3 * np.pi * (3 ** 3)  # at least radius 2
radius = None
num_vertices = None

obstacles = Polytope.random_set_polyhedra(
    n_polyhedra=n_polyhedra,
    intersect_polyhedra=False,
    q0=q0,
    qd=qd,
    max_vertices=max_vertices,
    radius_lim=radius_limits,
    bbox=bounding_box,
    seed=seed,
    dim=3,
    min_volume=min_volume,
    max_attempts=max_attempts,
    radius=radius,
    num_vertices=num_vertices,
)

# Test with simple cube 
# obstacles = [
#     Polytope(
#         A=np.array([
#             [1, 0, 0],
#             [-1, 0, 0],
#             [0, 1, 0],
#             [0, -1, 0],
#             [0, 0, 1],
#             [0, 0, -1],
#         ]),
#         b=np.array([5, 5, 5, 5, 5, 5]).reshape(-1, 1),
#     )
# ]

lambda_ = np.linspace(0, 1, n_points)
init_path = (1 - lambda_) * q0 + lambda_ * qd
path = init_path.copy()
dists = [-100]
iter_ = 0
path_hist = [init_path.copy()]
max_iters = 200
kind = "in"
kind = "out"
kind = None

# bounding_box = (-20.0, -20, 20, 20.11) # 1337 plotting related
fig = go.Figure()
for obstacle in obstacles:
    add_polyhedron(fig, obstacle.A, obstacle.b, add_reference=False)

# obstacles = [obstacles[0]]
while np.any(np.array(dists) < 0.0):
    if iter_ >= max_iters:
        print(f"reached max iterations: {max_iters}")
        break

    path, dist, grad = deform_path(
        path, obstacles, kind=kind, h=h, r=r, min_path=min_path, zeta=zeta, alpha=alpha
    )
    path_hist.append(path.copy())
    dists = dist
    if iter_ % 10 == 0:
        print(f"iteration {iter_}: min dist = {np.min(dists)}")

    iter_ += 1

print(f"deformation completed in {iter_} iterations with min dist = {np.min(dists)}")
add_path3d(fig, path_hist, num_paths=6, base_color="black")
fig.update_layout(width=1200, height=800)
fig.show()


