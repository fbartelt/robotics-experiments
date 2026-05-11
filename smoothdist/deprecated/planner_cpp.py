import numpy as np
import plotly.graph_objects as go
import plotly.colors as pc
import cyipopt

# from distances import signed_dist2convex, phi, smooth_min
from smoothfunctions import (
    signedDist2Convex,
    smoothMinListWithGradient,
    smoothMinList,
    phi,
)
from typing import List, Tuple, Optional, Callable
from polygon import (
    Polytope,
    add_polygon,
    create_level_sets,
    NonConvexPolygon,
    generate_random_polyhedron,
    generate_random_polyhedron_set,
)
from polyhedron import add_polyhedron
from path_planning import (
    deform_path_ipopt,
    add_path,
    add_path3d,
)

# %%
# Ipopt optimization approach
max_polygons = 10  # 7 (1111), 10 (1337), 5 (1001)
max_vertices = 20
bounding_box = (-20.0, -20, 20, 20)
# Distance between vertices will be at least 2*first element, and at most
# 2*second element of radius_limits:
radius_limits = (2, 6)

seed = 1111  # 1001, 1337, 1111
q0 = np.array([-3.0, -12]).reshape(-1, 1)
# SEED 1001
if seed == 1001:
    q0 = np.array([-17.0, -7]).reshape(-1, 1)  # 1001
    qd = np.array([19, -7]).reshape(-1, 1)
    max_polygons = 10
elif seed == 1111:
    q0 = np.array([-17.0, -17]).reshape(-1, 1)  # 1111
    qd = np.array([6.8, 19.0]).reshape(-1, 1)
    max_polygons = 15
elif seed == 1337:
    q0 = np.array([-6.8, -17.9]).reshape(-1, 1)  # 1337
    # q0 = np.array([1., -17.9]).reshape(-1, 1)  # 1337
    qd = np.array([13, 17.9]).reshape(-1, 1)
    max_polygons = 20
# SEED 1337
# q0 = np.array([-17.0, -17]).reshape(-1, 1)  # 1337, 1111
# qd = np.array([18, 15]).reshape(-1, 1)
# qd = np.array([11.0, 15]).reshape(-1, 1)
n_points = 100
h = 0.01
r = 0.1
zeta = 0.5 * 1
alpha = np.log(2) / 5e-2
min_path = True
max_attempts = 500
lambda_ = np.linspace(0, 1, n_points)
init_path = (1 - lambda_) * q0 + lambda_ * qd  # (2 x n_points)
init_path = init_path.T  # (n_points x 2)
path = init_path.copy()
delta = 3.0 * np.linalg.norm(path[:, 1] - path[:, 0]) ** 2
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

dists = [-100]
iter_ = 0
path_hist = [init_path.copy()]
opt_max_iters = 200
kind = "in"
# kind = "out"
kind = None

# bounding_box = (-20.0, -20, 20, 20.11) # 1337 plotting related
# fig = go.Figure()
print(f"Creating level sets for {len(obstacles)} obstacles.")
fig = create_level_sets(
    obstacles,
    r=r,
    h=h,
    kind="both",
    bbox=bounding_box,
    n_points=20,
    n_contours=40,
    add_reference=False,
    test=None,
    rescale=True,
)
print("Level sets created.")
# for obstacle in obstacles:
#     add_polygon(fig, obstacle.A, obstacle.b, add_reference=False)

# obstacles = [obstacles[0]]
ipopt_options = {
    "mu_strategy": "adaptive",
    "tol": 1e-3,  # Relax tolerance (default 1e-8)
    "max_iter": opt_max_iters,  # Increase iteration limit
    "acceptable_iter": 10,  # Stop after 10 "good enough" iters
    # Output control
    "print_level": 5,
    "print_frequency_iter": 10,
    "print_timing_statistics": "yes",
    # "print_frequency_time": 10.,
    # "max_wall_time": 120.0,
}

path_opt, path_hist, info = deform_path_ipopt(
    init_path,
    obstacles,
    method="esdf",
    verbose=True,
    max_iter=opt_max_iters,
    ipopt_options=ipopt_options,
    kind=kind,
    h=h,
    r=r,
    alpha=alpha,
    zeta=zeta,
    min_path=min_path,
    delta=delta,
)

print(f"deformation completed in {iter_} iterations with min dist = {np.min(dists)}")
add_path(fig, path_hist, num_paths=6, base_color="black")
fig.update_layout(width=1200, height=800)
fig.show()

# fig.write_image(f"path_seed_{seed}_maxpoly_{max_polygons}.pdf")


# [jump]
# %%
"""3D case"""
n_polyhedra = 29
max_vertices = 15
bounding_box = (-20.0, -20, -20, 20, 20, 20)
# Distance between vertices will be at least 2*first element, and at most
# 2*second element of radius_limits:
radius_limits = (2, 10)
q0 = np.array([-1.0, -15, 10.0]).reshape(-1, 1)
qd = np.array([18.0, 18, -18.0]).reshape(-1, 1)
# qd = np.array([11.0, 15]).reshape(-1, 1)
n_points = 100
h = 0.01
r = 0.1
zeta = 0.5
alpha = np.log(2) / 5e-2
min_path = True
max_attempts = 500
opt_max_iters = 200
seed = 1337  # 1001, 69 cool, 42 NICE post mods, 100 is cool
min_volume = 4 / 3 * np.pi * (3**3)  # at least radius 2
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

print(f"Generated {len(obstacles)} obstacles.")

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
init_path = init_path.T  # Shape (N x n)
path = init_path.copy()
print(f"Path shape: {path.shape}")
delta = 3.0 * np.linalg.norm(path[1] - path[0]) ** 2
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

path_opt, path_hist, info = deform_path_ipopt(
    init_path,
    obstacles,
    method="esdf",
    verbose=True,
    max_iter=opt_max_iters,
    kind=kind,
    h=h,
    r=r,
    alpha=alpha,
    zeta=zeta,
    min_path=min_path,
    delta=delta,
)

print(f"deformation completed in {iter_} iterations with min dist = {np.min(dists)}")
add_path3d(fig, path_hist, num_paths=6, base_color="black")
fig.update_layout(width=1200, height=800)
fig.show()
