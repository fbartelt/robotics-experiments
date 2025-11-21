# %%
import sys, os
import numpy as np
import plotly.graph_objects as go
import plotly.colors as pc

from polygon import (
    Polytope,
    NonConvexPolygon,
    add_polygon,
    get_polytope_constraints,
    create_level_sets,
)
from distances import signed_dist2convex, id_phi, phi, smooth_min, signed_dist2nonconvex
from polyhedron import add_polyhedron, create_isosurfaces

# %%
n_points = 100
max_iters = 100
h = 0.01
r = 0.1
eps = 5e-2
bulge = True
min_path = True
k = 5e-1
eta = 10.0
n_contours = 70
max_iters = 200
# bounding_box = (-6, -6, 6, 6)
bounding_box = (-1, -1, 5, 5)

seed = 42  # 100 is cool

polygon = Polytope.random(
    num_vertices=7, radius_lim=(1e-1, 1.0), bbox=bounding_box, seed=seed
)
polygon2 = Polytope.random(
    num_vertices=5, radius_lim=(1e-1, 1.0), bbox=bounding_box, seed=seed + 1
)
polygon3 = Polytope.random(
    num_vertices=6, radius_lim=(1e-1, 1.0), bbox=bounding_box, seed=seed + 2
)

polygons = [polygon, polygon2, polygon3]
# bounding_box = (-0.5, 0.8, 2, 3)
fig = create_level_sets(
    polygons,
    eps=eps,
    r=r,
    h=h,
    eta=eta,
    kind="both",
    bbox=bounding_box,
    n_points=n_points,
    n_countours=n_contours,
    ignore=[],
    test=False,
)
fig.show()

# %%
""" Nonconvex test"""
n_points = 100
max_iters = 100
h = 0.01
r = 0.1
eps = 5e-2
bulge = True
min_path = True
k = 5e-1
eta = 10.0
n_contours = 70
max_iters = 200
# bounding_box = (-6, -6, 6, 6)
bounding_box = (-1, -1, 5, 5)

seed = 42  # 100 is cool


A1 = np.array(
    [
        [1.0, 0],  # x <= 2
        [-1, 0],  # x >= 0
        [0, -1],  # y >= 0
        [0, 1],  # y <= 1
    ]
)
b1 = np.array([2.0, 0, 0, 1])
# b1 = np.array([2.0, 2.0, 0, 1])
# Vertical
A2 = np.array(
    [
        [1.0, 0],  # x <= 1
        [-1, 0],  # x >= 0
        [0, 1],  # y <= 3
        [0, -1],  # y >= 1
    ]
)
b2 = np.array([1.0, 0, 3, -1])
# Top horizontal
A3 = np.array(
    [
        [1.0, 0],  # x <= 2
        [-1, 0],  # x >= 0
        [0, -1],  # y >= 3
        [0, 1],  # y <= 4
    ]
)
b3 = np.array([2.0, 0, -3, 4])

A_list = [
    A1,
    A2,
    A3,
]
b_list = [
    b1,
    b2,
    b3,
]

shared_boundaries = [[None for _ in range(len(b_list))] for _ in range(len(b_list))]
# 3d array S[i, j, k] means that the k-th constraint of the i-th polytope
# is a common boundary with the j-th polytope.
# S[i, j] is a list that contains the indices of each constraint shared
# S[j, i] can be different from S[i, j] given that they are defined from
# each matrix A_i and A_j.
shared_boundaries[0][1] = [3]
shared_boundaries[1][0] = [3]
shared_boundaries[1][2] = [2]
shared_boundaries[2][1] = [2]

polygon = NonConvexPolygon(A_list, b_list, shared_boundaries)

fig = create_level_sets(
    polygon,
    eps=eps,
    r=r,
    h=h,
    eta=eta,
    kind="both",
    bbox=bounding_box,
    n_points=n_points,
    n_countours=n_contours,
    ignore=[],
    test=None,
)
fig.show()


# %%
""" 3D Convex test"""
n_points = 30
max_iters = 100
h = 0.01
r = 0.1
eps = 5e-2
bulge = True
min_path = True
k = 5e-1
eta = 10.0
n_contours = 25

bounding_box = (-4, -2, -4, 4, 0, 4)
seed = 42  # 100 is cool

# Define a simple cube polyhedron
A = np.array(
    [
        [1, 0, 0],  # x <= 1
        [-1, 0, 0],  # x >= -1
        [0, 1, 0],  # y <= 1
        [0, -1, 0],  # y >= -1
        [0, 0, 1],  # z <= 1
        [0, 0, -1],  # z >= -1
    ]
)
b = (
    np.array([1, 1, 1, 1, 1, 1]) / 2
    - (A @ np.array([2, 0, 2]).reshape(-1, 1)).flatten()
)
b2 = (
    np.array([1, 1, 1, 1, 1, 1]) / 2
    + (A @ np.array([0.5, 0, 0.5]).reshape(-1, 1)).flatten()
)
polyhedron = Polytope(A, b)
poly2hedron = Polytope(A, b2)
polyhedra = [polyhedron, poly2hedron]
fig = create_isosurfaces(
    polyhedra,
    r=r,
    h=h,
    eta=eta,
    kind="both",
    bbox=bounding_box,
    n_points=n_points,
    n_countours=n_contours,
    ignore=[],
    test=None,
    # test='in',
    caps=dict(x_show=False, y_show=False, z_show=True),
    opacity=1.0,
    surface_count=n_contours,
    # surface=dict(count=n_contours, fill=0.2, pattern="odd"),
    colorscale="Portland",
    name="Isosurfaces",
)
fig.show()

# %%
""" 3D NonConvex test"""
n_points = 30
max_iters = 100
h = 0.01
r = 0.1
eps = 5e-2
bulge = True
min_path = True
k = 5e-1
eta = 10.0
n_contours = 25

bounding_box = (-4, -2, -4, 4, 0, 4)
seed = 42  # 100 is cool

# Define an L-shaped polyhedron
A1 = np.array(
    [
        [1, 0, 0],  # x <= 1
        [-1, 0, 0],  # x >= -1
        [0, 1, 0],  # y <= 1
        [0, -1, 0],  # y >= -1
        [0, 0, 1],  # z <= 0
        [0, 0, -1],  # z >= -1
    ]
)
b1 = np.array([3 * 1.0, 3 * 1, 1.0, 1, 3 * 0, 3 * 1])
A2 = np.array(
    [
        [1, 0, 0],  # x <= 1
        [-1, 0, 0],  # x >= 0
        [0, 1, 0],  # y <= 1
        [0, -1, 0],  # y >= -1
        [0, 0, 1],  # z <= 1
        [0, 0, -1],  # z >= 0
    ]
)
b2 = np.array([3 * 1.0, 3 * 0, 1.0, 1, 3 * 1, 3 * 0])

# Other case: little cube on top/center of big cube
A3 = np.array(
    [
        [1, 0, 0],  # x <= 1
        [-1, 0, 0],  # x >= -1
        [0, 1, 0],  # y <= 1
        [0, -1, 0],  # y >= -1
        [0, 0, 1],  # z <= 1
        [0, 0, -1],  # z >= -1
    ]
)
b3 = np.array([1.0, 1, 1.0, 1, 1.0, 1.0])
A4 = np.array(
    [
        [1, 0, 0],  # x <= 0.5
        [-1, 0, 0],  # x >= -0.5
        [0, 1, 0],  # y <= 0.5
        [0, -1, 0],  # y >= -0.5
        [0, 0, 1],  # z <= 1.5
        [0, 0, -1],  # z >= 1
    ]
)
b4 = np.array([0.5, 0.5, 0.5, 0.5, 2.0, -1.0])

A_list = [
    # A1,
    # A2,
    A3,
    A4,
]
b_list = [
    # b1,
    # b2,
    b3,
    b4,
]

shared_boundaries = [[None for _ in range(len(b_list))] for _ in range(len(b_list))]
# 3d array S[i, j, k] means that the k-th constraint of the i-th polytope
# is a common boundary with the j-th polytope.
# S[i, j] is a list that contains the indices of each constraint shared
# S[j, i] can be different from S[i, j] given that they are
# defined from each matrix A_i and A_j.
### TODO: Should not deactivate x <= 1 constraint (same constraint for both polytopes)
### Same for constraints on y
shared_boundaries[0][1] = [4]
shared_boundaries[1][0] = [5]
## Cube case
shared_boundaries[0][1] = [4]
shared_boundaries[1][0] = [5]
polyhedron = NonConvexPolygon(A_list, b_list, shared_boundaries)

polyhedra = [polyhedron]
# polyhedra = [Polytope(A1, b1), Polytope(A2, b2)]
fig = create_isosurfaces(
    polyhedra,
    r=r,
    h=h,
    eta=eta,
    kind="both",
    bbox=bounding_box,
    n_points=n_points,
    n_countours=n_contours,
    ignore=[],
    test=None,
    # test='in',
    normalize_for_visualization=True,
    caps=dict(x_show=False, y_show=False, z_show=True),
    opacity=1.0,
    surface_count=n_contours,
    # surface=dict(count=n_contours, fill=0.2, pattern="odd"),
    colorscale="RdBu",
    name="Isosurfaces",
)
fig.show()


# %%
""" Plotlyu example"""

X, Y, Z = np.mgrid[-5:5:40j, -5:5:40j, -5:5:40j]

# ellipsoid
values = X * X * 0.5 + Y * Y + Z * Z * 2
print(X.shape, values.shape)
print(X.flatten().shape, values.flatten().shape)

fig = go.Figure(
    data=go.Isosurface(
        x=X.flatten(),
        y=Y.flatten(),
        z=Z.flatten(),
        value=values.flatten(),
        # isomin=10,
        # isomax=40,
        surface_count=10,
        caps=dict(x_show=False, y_show=False),
    )
)
fig.show()
