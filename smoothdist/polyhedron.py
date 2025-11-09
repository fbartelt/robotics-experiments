import numpy as np
import plotly.graph_objects as go
from polygon import find_strictly_feasible_point, get_polytope_constraints
from scipy.optimize import linprog
from scipy.spatial import HalfspaceIntersection, ConvexHull
from scipy.special import factorial


def add_polyhedron(fig, A, b, aux=0, add_reference=False):
    interior_point = find_strictly_feasible_point(A, b)
    halfspaces = np.hstack((A, -b[:, None]))
    hs = HalfspaceIntersection(halfspaces, interior_point)
    reconstructed_vertices = hs.intersections
    # Use ConvexHull to order them
    hull = ConvexHull(reconstructed_vertices)
    ordered_vertices = reconstructed_vertices[hull.vertices]
    x = ordered_vertices[:, 0]
    y = ordered_vertices[:, 1]
    z = ordered_vertices[:, 2]

    fig.add_trace(
        go.Mesh3d(
            x=x,
            y=y,
            z=z,
            color='lightblue',
            opacity=0.5,
            name='Polyhedron'
        )
    )

    if add_reference:
        print("Not implemented yet for 3D polyhedra.")

