import numpy as np
import plotly.graph_objects as go
from polygon import find_strictly_feasible_point, get_polytope_constraints
from scipy.optimize import linprog
from scipy.spatial import HalfspaceIntersection, ConvexHull
from scipy.special import factorial
from distances import signed_dist2convex, id_phi, phi, smooth_min, signed_dist2nonconvex
from polygon import Polytope, NonConvexPolygon


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
            alphahull=0,
            color="rgba(163, 159, 158, 0.4)",
            opacity=0.5,
            name='Polyhedron'
        )
    )

    if add_reference:
        print("Not implemented yet for 3D polyhedra.")


def create_isosurfaces(
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
    add_reference=False,
    normalize_for_visualization=False,
    *args,
    **kwargs
):
    fig = go.Figure()

    if not isinstance(polyhedra, (list, tuple)):
        polyhedra = [polyhedra]

    for polyhedron in polyhedra:
        if isinstance(polyhedron, NonConvexPolygon):
            for iii, poly in enumerate(polyhedron.polytopes):
                A = poly.A.copy()
                b = poly.b.copy()
                add_polyhedron(fig, A, b, aux=iii, add_reference=add_reference)
        elif isinstance(polyhedron, Polytope):
            add_polyhedron(fig, polyhedron.A, polyhedron.b, add_reference=add_reference)
        else:
            raise ValueError("Unknown polyhedron type")

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
                    if isinstance(polyhedron, NonConvexPolygon):
                        dist_i = signed_dist2nonconvex(
                            # id_phi,
                            phi,
                            p,
                            polyhedron.A_list,
                            polyhedron.b_list,
                            polyhedron.shared_boundaries,
                            r=r,
                            h=h,
                            test=test,
                        )
                    elif isinstance(polyhedron, Polytope):
                        dist_i = signed_dist2convex(
                            # id_phi,
                            phi,
                            p,
                            polyhedron.A,
                            polyhedron.b,
                            r=r,
                            h=h,
                            test=test,
                        )
                    else:
                        raise ValueError("Unknown polyhedron type")
                    dists.append(dist_i)
                distances[i, j, k] = smooth_min(dists, r=r)

    print(f"Min distance: {np.min(distances)}, Max distance: {np.max(distances)}")
    if normalize_for_visualization:
        if np.max(distances) > 0:
            # Center values aoround zero
            # Multiply every negative value by a factor to enhance visualization
            max_positive = np.max(distances[distances >= 0])
            distances[distances < 0] *= max_positive
        
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


