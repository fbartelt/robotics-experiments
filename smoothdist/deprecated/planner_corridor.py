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
    ESDF_CGAL,
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
    animate_deformation_matplotlib,
    show_animation,
)

A1 = np.array(
    [
        [1.0, 0.0],
        [-1.0, 0.0],
        [0.0, 1.0],
        [0.0, -1.0],
    ]
)

# Room-like environment
bounding_box = (-12.0, -10.0, 7.0, 10.0)
q0 = np.array([-8.0, -9]).reshape(-1, 1)  # 1111
qd = np.array([6.0, 9.0]).reshape(-1, 1)  # 1111
left_wall = np.array([-5, 7.0, 5.0, 5.0])
o1 = Polytope(A1, left_wall)
top_wall = np.array([3.0, 7.0, 7.0, -5.0])
o2 = Polytope(A1, top_wall)
bottom_wall = np.array([-5.0, 12.0, -5.0, 8.0])
o3 = Polytope(A1, bottom_wall)
bottom_wall2 = np.array([7.0, 4.0, -5.0, 8.0])
o4 = Polytope(A1, bottom_wall2)
right_wall = np.array([7.0, -5.0, 7.0, 5.0])
o5 = Polytope(A1, right_wall)
obstacles = [
    o1,
    o2,
    o3,
    o4,
    o5,
]
table = np.array([4.0, 4.0, 3.0, 3.0])
o6 = Polytope(A1, table)
# obstacles.append(o6)
desk_base = np.array([-2.0, 4.0, 3.0, -1.5])
for i in range(3):
    for j in range(4):
        displacement = np.array([i * 2.5, -j * 2.0])
        desk_i = desk_base + A1 @ displacement
        o_desk = Polytope(A1, desk_i)
        obstacles.append(o_desk)


n_points = 100
h = 0.01
r = 0.1
zeta = 0.5 / 2
zeta = np.sqrt(0.5 / 2) * 1e1 * 88
# alpha = np.log(2) / 7e-1
alpha = 10.0 # Removing saturation and adding more weight to negative
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
fig, cmap_data = create_level_sets(
    obstacles,
    r=r,
    h=h,
    kind="both",
    method=method,
    bbox=bounding_box,
    n_points=20,
    n_contours=40,
    add_reference=False,
    test=None,
    rescale=True,
    return_cmap_data=True,
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
print(info["status"])
# %%
animation = animate_deformation_matplotlib(
    path_hist,
    init_path,
    obstacles,
    q0=q0,
    qd=qd,
    p1=cmap_data[0],
    p2=cmap_data[1],
    distances=cmap_data[2],
    frame_delay=(5 / len(path_hist)) * 1000,
)
show_animation(animation, filename=f"animations/deform_{method}.html")
