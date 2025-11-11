# %%
import sys, os
import numpy as np
import plotly.graph_objects as go
import plotly.colors as pc

from polygon import Polytope, NonConvexPolygon, add_polygon, get_polytope_constraints
from distances import signed_dist2convex, id_phi, phi, smooth_min, signed_dist2nonconvex
from polyhedron import add_polyhedron


def create_level_sets_convex(
    polygons,
    eps=1e-3,
    r=1e-1,
    h=1e-1,
    eta=1.0,
    kind="both",
    bbox=(-5, -5, 5, 5),
    n_points=100,
    n_countours=50,
    ignore=[],
    test_=False,
):
    fig = go.Figure()

    if not isinstance(polygons, (list, tuple)):
        polygons = [polygons]

    for polygon in polygons:
        add_polygon(fig, polygon.A, polygon.b, add_reference=False)

    p1 = np.linspace(bbox[0], bbox[2], n_points)
    p2 = np.linspace(bbox[1], bbox[3], n_points)

    # Compute distances to single polygon:
    distances = []
    for x in p1:
        for y in p2:
            p = np.array([x, y]).reshape(-1, 1)
            dists = []
            for polygon in polygons:
                dist_i = signed_dist2convex(
                    id_phi,
                    # phi,
                    p,
                    polygon.A,
                    polygon.b,
                    r=0.01,
                    h=0.5,
                    # test="out"
                    # test="in"
                    test=None,
                )
                dists.append(dist_i)
            distances.append(smooth_min(dists, r=r))
            # distances.append(dist)

    distances = np.array(distances).reshape(n_points, n_points).T
    contour = go.Contour(
        x=p1,
        y=p2,
        z=distances,
        colorscale="RdBu",
        ncontours=n_countours,
        name="Level Sets",
    )
    fig.add_trace(contour)
    # Update layout
    fig.update_layout(
        xaxis_title="x",
        yaxis_title="y",
        showlegend=False,
        width=800,
        height=800,
        xaxis_range=[bbox[0], bbox[2]],
        yaxis_range=[bbox[1], bbox[3]],
        margin=dict(t=0, l=10, r=10, b=10),
    )

    return fig


def create_level_sets_nonconvex(
    polygons,
    eps=1e-3,
    r=1e-1,
    h=1e-1,
    eta=1.0,
    kind="both",
    bbox=(-5, -5, 5, 5),
    n_points=100,
    n_countours=50,
    ignore=[],
    test=False,
):
    fig = go.Figure()

    if not isinstance(polygons, (list, tuple)):
        polygons = [polygons]

    for polygon in polygons:
        for iii, poly in enumerate(polygon.polytopes):
            A = poly.A.copy()
            b = poly.b.copy()
            add_polygon(fig, A, b, aux=iii, add_reference=False)

    p1 = np.linspace(bbox[0], bbox[2], n_points)
    p2 = np.linspace(bbox[1], bbox[3], n_points)

    # Compute distances to single polygon:
    distances = []
    for x in p1:
        for y in p2:
            p = np.array([x, y]).reshape(-1, 1)
            dists = []
            for polygon in polygons:
                dist_i = signed_dist2nonconvex(
                    id_phi,
                    # phi,
                    p,
                    polygon.A_list,
                    polygon.b_list,
                    polygon.shared_boundaries,
                    r=0.01,
                    h=0.5,
                    # test="out"
                    test=test,
                    # test=None
                )
                dists.append(dist_i)
            distances.append(smooth_min(dists, r=r))
            # distances.append(dist)

    distances = np.array(distances).reshape(n_points, n_points).T
    contour = go.Contour(
        x=p1,
        y=p2,
        z=distances,
        colorscale="RdBu",
        ncontours=n_countours,
        name="Level Sets",
    )
    fig.add_trace(contour)
    # Update layout
    fig.update_layout(
        xaxis_title="x",
        yaxis_title="y",
        showlegend=False,
        width=800,
        height=800,
        xaxis_range=[bbox[0], bbox[2]],
        yaxis_range=[bbox[1], bbox[3]],
        margin=dict(t=0, l=10, r=10, b=10),
    )

    return fig


def create_isosurfaces_convex(
    polyhedra,
    r=1e-1,
    h=1e-1,
    eta=1.0,
    kind="both",
    bbox=(-5, -5, -5, 5, 5, 5),
    n_points=100,
    n_countours=50,
    ignore=[],
    test=False,
    *args,
    **kwargs
):
    fig = go.Figure()

    if not isinstance(polyhedra, (list, tuple)):
        polyhedra = [polyhedra]

    for polyhedron in polyhedra:
        pass
        # add_polyhedron(fig, polyhedron.A, polyhedron.b, add_reference=False)

    p1, p2, p3 = np.mgrid[
        bbox[0] : bbox[3] : n_points * 1j,
        bbox[1] : bbox[4] : n_points * 1j,
        bbox[2] : bbox[5] : n_points * 1j,
    ]
    # Compute distances to single polygon:
    distances = np.zeros(p1.shape)
    for i in range(n_points):
        for j in range(n_points):
            for k in range(n_points):
                p = np.array([p1[i, j, k], p2[i, j, k], p3[i, j, k]]).reshape(-1, 1)
                dists = []
                for polyhedron in polyhedra:
                    dist_i = signed_dist2convex(
                        id_phi,
                        # phi,
                        p,
                        polyhedron.A,
                        polyhedron.b,
                        r=0.01,
                        h=0.5,
                        test=test,
                    )
                    dists.append(dist_i)
                distances[i, j, k] = smooth_min(dists, r=r)

    # Create isosurface
    isosurface = go.Isosurface(
        x=p1.flatten(),
        y=p2.flatten(),
        z=p3.flatten(),
        value=distances.flatten(),
        isomin=np.min(distances),
        isomax=np.max(distances),
        # isomin=-0.1,
        # isomax=0.1,
        *args,
        **kwargs,
    )
    fig.add_trace(isosurface)
    # Update layout

    # Compute distances to single polyhedron:
    # distances = np.zeros((n_points, n_points, n_points))
    # for i, x in enumerate(p1):
    #     for j, y in enumerate(p2):
    #         for k, z in enumerate(p3):
    #             p = np.array([x, y, z]).reshape(-1, 1)
    #             dists = []
    #             for polyhedron in polyhedra:
    #                 dist_i = signed_dist2convex(
    #                         id_phi,
    #                         # phi,
    #                         p,
    #                         polyhedron.A,
    #                         polyhedron.b,
    #                         r=0.01,
    #                         h=0.5,
    #                         test=test
    #                 )
    #                 dists.append(dist_i)
    #             distances[i, j, k] = smooth_min(dists, r=r)
    #
    # # Create isosurface
    # isosurface = go.Isosurface(
    #     x=np.tile(p1, n_points * n_points),
    #     y=np.tile(np.repeat(p2, n_points), n_points),
    #     z=np.repeat(p3, n_points * n_points),
    #     value=distances.flatten(),
    #     # isomin=-0.1,
    #     # isomax=0.1,
    #     caps=dict(x_show=False, y_show=False),
    #     opacity=0.8,
    #     # surface=dict(count=n_countours, fill=0.9, pattern='odd'),
    #     # surface_fill=0.7,
    #     surface_count=n_countours,
    #     colorscale="RdBu",
    #     name="Isosurfaces",
    # )
    # fig.add_trace(isosurface)
    # Update layout

    return fig


# %%
n_points = 100
max_iters = 100
h = 0.1
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
fig = create_level_sets_convex(
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
    test_=False,
)
fig.show()


# %%
""" Nonconvex test"""
n_points = 100
max_iters = 100
h = 0.1
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

fig = create_level_sets_nonconvex(
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
h = 0.1
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
b = np.array([1, 1, 1, 1, 1, 1]) / 2 - (A @ np.array([2, 0, 2]).reshape(-1, 1)).flatten()
b2 = np.array([1, 1, 1, 1, 1, 1]) / 2 + (A @ np.array([0.5, 0, 0.5]).reshape(-1, 1)).flatten()
polyhedron = Polytope(A, b)
poly2hedron = Polytope(A, b2)
polyhedra = [polyhedron, poly2hedron]
fig = create_isosurfaces_convex(
    polyhedra,
    r=r,
    h=h,
    eta=eta,
    kind="both",
    bbox=bounding_box,
    n_points=n_points,
    n_countours=n_contours,
    ignore=[],
    # test=None,
    test='out',
    caps=dict(x_show=False, y_show=False, z_show=True),
    opacity=1.0,
    surface_count=n_contours,
    # surface=dict(count=n_contours, fill=0.2, pattern="odd"),
    colorscale="Portland",
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
