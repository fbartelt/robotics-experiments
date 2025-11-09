# %%
import sys, os
import numpy as np
import plotly.graph_objects as go
import plotly.colors as pc

from polygon import Polytope, NonConvexPolygon, add_polygon, get_polytope_constraints
from distances import signed_dist2convex, id_phi, phi, smooth_min

def create_level_sets(
        polygons,
        eps=1e-3,
        r=1e-1,
        h=1e-1,
        eta=1.0,
        kind='both',
        bbox=(-5, -5, 5, 5),
        n_points=100,
        n_countours=50,
        ignore=[],
        test_=False):
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
                        r=0.01, h=0.5,
                        # test="out"
                        # test="in"
                        test=None
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
        xaxis_title="x", yaxis_title="y", showlegend=False, width=800, height=800,
        xaxis_range=[bbox[0], bbox[2]], yaxis_range=[bbox[1], bbox[3]],
        margin=dict(t=0, l=10, r=10, b=10)
    )

    return fig



n_points = 100
max_iters = 100
h = 0.1
r = 0.1
eps = 5e-2
bulge = True
min_path = True
k = 5e-1
eta=10.0
n_contours = 70
max_iters = 200
# bounding_box = (-6, -6, 6, 6)
bounding_box = (-1, -1, 5, 5)

seed = 42 # 100 is cool

polygon = Polytope.random(num_vertices=7, radius_lim=(1e-1, 1.0), bbox=bounding_box, seed=seed)
polygon2 = Polytope.random(num_vertices=5, radius_lim=(1e-1, 1.0), bbox=bounding_box, seed=seed+1)
polygon3 = Polytope.random(num_vertices=6, radius_lim=(1e-1, 1.0), bbox=bounding_box, seed=seed+2)

polygons = [polygon, polygon2, polygon3]
# bounding_box = (-0.5, 0.8, 2, 3)
fig = create_level_sets(
    polygons,
    eps=eps,
    r=r,
    h=h,
    eta=eta,
    kind='both',
    bbox=bounding_box,
    n_points=n_points,
    n_countours=n_contours,
    ignore=[],
    test_=False
)
fig.show()
