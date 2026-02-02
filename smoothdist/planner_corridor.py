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


bounding_box = (-12., -10., 7., 10.)
# bounding_box = (-20., -20., 20., 20.)
q0 = np.array([-10.0, -17]).reshape(-1, 1)  # 1111
q0 = np.array([-8.0, -9]).reshape(-1, 1)  # 1111
# qd = -q0
qd = np.array([0., 17]).reshape(-1, 1)
qd = np.array([6.0, 9.0]).reshape(-1, 1)  # 1111

n_points = 100
h = 0.01
r = 0.1
zeta = 0.5 / 2
zeta = np.sqrt(0.5 / 2) * 1e1
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

# Obstacles will be rectangles forming a corridor
A1 = np.array([
    [1.0, 0.0],
    [-1.0, 0.0],
    [0.0, 1.0],
    [0.0, -1.0],
])
# b1 = np.array([-0.5, 5.0*3, 15.0, 5.0])
# o1 = Polytope(A1, b1)
# b2 = np.array([5.0*3, -0.5, 15.0, 5.0])
# o2 = Polytope(A1, b2)
# horizontal_left = np.array([-0.5, 20.0, -5., 15.0])
# horizontal_right = np.array([20., -0.5, -5., 15.0])
# o3 = Polytope(A1, horizontal_left)
# o4 = Polytope(A1, horizontal_right)
# horizontal_top_right = np.array([20., 2.5, 30.0, -16.0])
# o5 = Polytope(A1, horizontal_top_right)
left_wall = np.array([-5, 7.0, 5.0, 5.0])
o1 = Polytope(A1, left_wall)
top_wall = np.array([3.0, 7.0, 7.0, -5.0])
o2 = Polytope(A1, top_wall)
bottom_wall = np.array([-5.0, 12.0, -5.0, 8.0])
o3 = Polytope(A1, bottom_wall)
bottom_wall2 = np.array([7.0, 4.0, -5.0, 8.0])
o4 = Polytope(A1, bottom_wall2)
right_wall = np.array([7., -5.0, 7.0, 5.0])
o5 = Polytope(A1, right_wall)
table = np.array([4.0, 4.0, 3.0, 3.0])
o6 = Polytope(A1, table)
obstacles = [o1, o2, o3, o4, o5, o6]

dists = [-100]
iter_ = 0
path_hist = [init_path.copy()]
opt_max_iters = 200 * 1
kind = None
method = "ours"
# method = "esdf"

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

path_opt, path_hist, info = deform_path_ipopt(
    init_path,
    obstacles,
    method=method,
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
add_path(fig, path_hist, num_paths=6, base_color="black")
fig.update_layout(width=1200, height=800).show()
# aux_fun(path_opt)
# fig.show()
print(info['status'])




