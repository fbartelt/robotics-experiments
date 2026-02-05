import numpy as np
import plotly.graph_objects as go
from scipy.optimize import linprog
from scipy.spatial import HalfspaceIntersection, ConvexHull
from scipy.special import factorial

# from typing import List, Tuple
from distances import signed_dist2nonconvex
from scipy.special import gamma
from smoothfunctions import (
    signedDist2Convex,
    ESDF_CGAL,
    smoothMinList,
    phi,
)


def get_polytope_constraints(vertices):
    """
    Given a set of vertices (points) of a convex polytope,
    return the matrix A and vector b such that Ax <= b describes the polytope.

    Parameters:
        vertices (ndarray): An (N, d) array of N points in d dimensions.

    Returns:
        A (ndarray): An (m, d) matrix.
        b (ndarray): An (m,) vector.
        The inequality Ax <= b defines the convex hull of the input vertices.
    """
    hull = ConvexHull(vertices)
    A = hull.equations[:, :-1]
    b = -hull.equations[:, -1]
    return A, b


def find_strictly_feasible_point(A, b):
    """
    Solve an LP to find a strictly feasible point x such that A x < b
    """
    m, n = A.shape
    # Objective: maximize δ (slack)
    c = np.zeros(n + 1)
    c[-1] = -1  # Maximize δ ⇒ minimize -δ

    # Constraints: A x + δ ||A_i|| ≤ b_i
    norms = np.linalg.norm(A, axis=1)
    A_lp = np.hstack((A, norms[:, None]))
    bounds = [(None, None)] * n + [(0, None)]  # δ ≥ 0

    res = linprog(c, A_ub=A_lp, b_ub=b, bounds=bounds, method="highs")
    if res.success:
        return res.x[:-1]  # Return x (ignore δ)
    else:
        raise ValueError("Could not find a strictly feasible point.")


def add_polygon(fig, A, b, aux=0, add_reference=True):
    interior_point = find_strictly_feasible_point(A, b)
    halfspaces = np.hstack((A, -b[:, None]))
    hs = HalfspaceIntersection(halfspaces, interior_point)
    reconstructed_vertices = hs.intersections

    # Use ConvexHull to order them
    hull = ConvexHull(reconstructed_vertices)
    ordered_vertices = reconstructed_vertices[hull.vertices]

    x = np.append(ordered_vertices[:, 0], ordered_vertices[0, 0])
    y = np.append(ordered_vertices[:, 1], ordered_vertices[0, 1])

    fig.add_trace(
        go.Scatter(
            x=x,
            y=y,
            mode="lines",
            fill="toself",
            fillcolor="rgba(163, 159, 158, 0.2)",
            line=dict(color="rgba(163, 159, 158, 1)"),
        )
    )
    # Draw each polygon edge with its own color
    COLOR_CYCLE = [
        "rgba(255, 0, 0, 0.8)",  # Red
        "rgba(0, 255, 0, 0.8)",  # Green
        "rgba(0, 0, 255, 0.8)",  # Blue
        "rgba(255, 165, 0, 0.8)",  # Orange
        "rgba(0, 255, 255, 0.8)",  # Cyan
        "rgba(75, 0, 130, 0.8)",  # Indigo
        "rgba(238, 130, 238, 0.8)",  # Violet
        "rgba(255, 20, 147, 0.8)",  # Deep Pink
        "rgba(255, 105, 180, 0.8)",  # Hot Pink
    ]
    MARKER_CYCLE = [
        "circle",
        "square",
        "circle-open",
        "square-open",
        "triangle-up",
        "triangle-down",
        "triangle-left",
        "triangle-right",
    ]
    if add_reference:
        x_range = fig.layout.xaxis.range or [x.min() - 1, x.max() + 1]
        y_range = fig.layout.yaxis.range or [y.min() - 1, y.max() + 1]
        xmin, xmax = x_range
        ymin, ymax = y_range
        for i, (a, bi) in enumerate(zip(A, b)):
            a = np.asarray(a)
            color = COLOR_CYCLE[aux % len(COLOR_CYCLE)]
            # Plot constraint line: a[0]*x + a[1]*y = b
            if abs(a[1]) > 1e-8:
                # Solve for y
                x_vals = np.linspace(xmin, xmax, 10)
                y_vals = (bi - a[0] * x_vals) / a[1]
            elif abs(a[0]) > 1e-8:
                # Vertical line
                x_vals = np.full(500, bi / a[0])
                y_vals = np.linspace(ymin, ymax, 10)
            else:
                continue  # skip invalid constraints

            fig.add_trace(
                go.Scatter(
                    x=x_vals,
                    y=y_vals,
                    mode="markers",
                    line=dict(color=color, width=2),
                    marker=dict(symbol=MARKER_CYCLE[i % len(MARKER_CYCLE)], size=8),
                    name=f"Constraint {i}",
                )
            )


class Polytope:
    def __init__(self, A, b):
        self.A = np.array(A, dtype=float)
        self.b = np.array(b, dtype=float).ravel()
        self.vertices = self.get_vertices()
        self.centroid = self.get_centroid()
        self.dist_to_centroid = None

    @staticmethod
    def get_polytope_vertices(A, b):
        m, n = A.shape
        # Objective: maximize δ (slack)
        c = np.zeros(n + 1)
        c[-1] = -1  # Maximize δ ⇒ minimize -δ

        # Constraints: A x + δ ||A_i|| ≤ b_i
        norms = np.linalg.norm(A, axis=1)
        A_lp = np.hstack((A, norms[:, None]))
        bounds = [(None, None)] * n + [(0, None)]  # δ ≥ 0

        res = linprog(c, A_ub=A_lp, b_ub=b, bounds=bounds, method="highs")
        interior_point = res.x[:-1]  # Exclude δ
        halfspaces = np.hstack((A, -b[:, None]))
        hs = HalfspaceIntersection(halfspaces, interior_point)
        reconstructed_vertices = hs.intersections

        # Use ConvexHull to order them
        hull = ConvexHull(reconstructed_vertices)
        ordered_vertices = reconstructed_vertices[hull.vertices]
        return ordered_vertices

    @staticmethod
    def get_polytope_centroid(vertices):
        hull = ConvexHull(vertices)
        dim = vertices.shape[1]
        total_volume = 0.0
        centroid = np.zeros(dim)

        # Decompose the polytope into simplices from a fixed vertex (e.g., first one)
        ref_point = np.mean(
            vertices[hull.vertices], axis=0
        )  # Could also use hull.points[0]

        for simplex in hull.simplices:
            pts = vertices[simplex]
            simplex_points = np.vstack([ref_point, pts])
            volume = abs(
                np.linalg.det(simplex_points[1:] - simplex_points[0])
            ) / factorial(dim)
            simplex_centroid = np.mean(simplex_points, axis=0)
            centroid += volume * simplex_centroid
            total_volume += volume

        if total_volume == 0:
            raise ValueError("Degenerate polytope with zero volume.")

        return centroid / total_volume

    @staticmethod
    def random(
        max_vertices=20,
        radius_lim=(1e-1, 1.0),
        bbox=(-5, -5, 5, 5),
        seed=None,
        min_area=None,
        max_attempts=100,
        radius=None,
        num_vertices=None,
    ):
        A, b, vertices, area = generate_random_polygon(
            max_vertices=max_vertices,
            radius_lim=radius_lim,
            bbox=bbox,
            seed=seed,
            min_area=min_area,
            max_attempts=max_attempts,
            radius=radius,
            num_vertices=num_vertices,
        )

        return Polytope(A, b)

    @staticmethod
    def random_set(
        n_polytopes=4,
        intersect_polytopes=False,
        q0=None,
        qd=None,
        max_vertices=20,
        radius_lim=(1e-1, 1.0),
        bbox=(-5, -5, 5, 5),
        seed=None,
        min_area=None,
        max_attempts=1000,
        radius=None,
        num_vertices=None,
    ):
        polygons = generate_random_polygon_set(
            n_polygons=n_polytopes,
            intersect_polygons=intersect_polytopes,
            q0=q0,
            qd=qd,
            max_vertices=max_vertices,
            radius_lim=radius_lim,
            bbox=bbox,
            seed=seed,
            min_area=min_area,
            max_attempts=max_attempts,
            radius=radius,
            num_vertices=num_vertices,
        )
        polytopes = [Polytope(A, b) for (A, b, _, _, _) in polygons]
        return polytopes

    @staticmethod
    def random_set_polyhedra(
        n_polyhedra=4,
        intersect_polyhedra=False,
        q0=None,
        qd=None,
        max_vertices=20,
        radius_lim=(1e-1, 1.0),
        dim=3,
        bbox=(-5, 5),
        seed=None,
        min_volume=None,
        max_attempts=1000,
        radius=None,
        num_vertices=None,
    ):
        polyhedra = generate_random_polyhedron_set(
            n_polyhedra=n_polyhedra,
            intersect_polyhedra=intersect_polyhedra,
            q0=q0,
            qd=qd,
            max_vertices=max_vertices,
            radius_lim=radius_lim,
            dim=dim,
            bbox=bbox,
            seed=seed,
            min_volume=min_volume,
            max_attempts=max_attempts,
            radius=radius,
            num_vertices=num_vertices,
        )
        polytopes = [Polytope(A, b) for (A, b, _, _, _) in polyhedra]
        return polytopes

    def get_vertices(self):
        return self.get_polytope_vertices(self.A, self.b)

    def get_centroid(self):
        return self.get_polytope_centroid(self.vertices)


class NonConvexPolygon:
    """A class to represent a non-convex polygon defined by the union of
    multiple convex polytopes, whose intersection is one-dimensional.
    """

    def __init__(self, A_list, b_list, shared_boundaries):
        if len(A_list) != len(b_list):
            raise ValueError("As and bs must have the same length")
        self.polytopes = [Polytope(A, b) for A, b in zip(A_list, b_list)]
        self.shared_boundaries = shared_boundaries
        self.A_list = A_list.copy()
        self.b_list = b_list.copy()


def generate_random_polygon(
    max_vertices=20,
    radius_lim=(1e-1, 1.0),
    bbox=(-5, -5, 5, 5),
    seed=None,
    min_area=None,
    max_attempts=100,
    radius=None,
    num_vertices=None,
):
    def gen1(seed, num_vertices=num_vertices, radius=radius):
        rng = np.random.default_rng(seed)
        if num_vertices is None:
            num_vertices = rng.integers(3, max_vertices + 1).item()
        if radius is None:
            radius = rng.uniform(radius_lim[0], radius_lim[1])
        angles = np.sort(rng.uniform(0, 2 * np.pi, num_vertices))
        vertices = np.array([radius * np.cos(angles), radius * np.sin(angles)]).T
        # Calculate safe translation boundaries
        xmin, ymin, xmax, ymax = bbox
        offset = rng.uniform(
            low=[xmin + radius, ymin + radius], high=[xmax - radius, ymax - radius]
        )
        vertices += offset
        hull = ConvexHull(vertices)
        A = hull.equations[:, :-1]
        b = -hull.equations[:, -1]
        area = hull.volume
        return A, b, vertices, area

    if min_area is None:
        return gen1(seed)
    else:
        attempts = 0
        area = 0.0
        while area < min_area and attempts < max_attempts:
            A, b, vertices, area = gen1(
                seed + attempts if seed is not None else attempts
            )
            attempts += 1
        return A, b, vertices, area


def generate_random_polyhedron(
    max_vertices=20,
    radius_lim=(1e-1, 1.0),
    bbox=(-5, 5),
    dim=3,
    seed=None,
    min_volume=None,
    max_attempts=100,
    radius=None,
    num_vertices=None,
):
    if len(bbox) != 2 * dim:
        if len(bbox) == 2:
            bbox = (bbox[0],) * dim + (bbox[1],) * dim
        else:
            raise ValueError("Bounding box must have length 2 * dim")

    def nsphere_coords(dim, r, angles):
        coords = np.zeros(dim)
        for i in range(dim):
            coords[i] = (
                r
                * np.prod(np.sin(angles[:i]))
                * (np.cos(angles[i]) if i < dim - 1 else 1)
            )
        return coords

    def gen1(seed, num_vertices=num_vertices, radius=radius):
        rng = np.random.default_rng(seed)
        if num_vertices is None:
            num_vertices = rng.integers(dim + 1, max_vertices + 1).item()
        if radius is None:
            radius = rng.uniform(radius_lim[0], radius_lim[1])
        phi_angles = np.sort(rng.uniform(0, np.pi, (num_vertices, dim - 1)))
        phi_angles[:, -1] *= 2  # Last angle in [0, 2pi]

        vertices = np.array(
            [nsphere_coords(dim, radius, angles) for angles in phi_angles]
        )
        # Calculate safe translation boundaries
        try:
            offset = rng.uniform(
                low=[bbox[i] + radius for i in range(dim)],
                high=[bbox[i + dim] - radius for i in range(dim)],
            )
        except ValueError:
            print(bbox, dim, radius)
            raise ValueError("Bounding box too small for the given radius limits")
        vertices += offset
        hull = ConvexHull(vertices)
        A = hull.equations[:, :-1]
        b = -hull.equations[:, -1]
        volume = hull.volume
        return A, b, vertices, volume

    if min_volume is None:
        return gen1(seed)
    else:
        attempts = 0
        volume = 0.0
        while volume < min_volume and attempts < max_attempts:
            A, b, vertices, volume = gen1(
                seed + attempts if seed is not None else attempts
            )
            attempts += 1
        return A, b, vertices, volume


def is_point_inside_polygon(point, A, b, tol=1e-6):
    res = linprog(
        c=[
            0.0,
        ]
        * len(point),
        # c=[0.0, 0.0],  # dummy objective
        A_ub=A,
        b_ub=b,
        # bounds=[(point[0], point[0]), (point[1], point[1])],
        bounds=[(p, p) for p in point],
        method="highs",
    )
    return res.success and res.status == 0


def polygons_intersect(A1, b1, A2, b2):
    A_combined = np.vstack([A1, A2])
    b_combined = np.vstack([b1.reshape(-1, 1), b2.reshape(-1, 1)])

    res = linprog(
        # c=[0.0, 0.0],
        c=[
            0.0,
        ]
        * A1.shape[1],
        A_ub=A_combined,
        b_ub=b_combined,
        # bounds=(None, None),
        bounds=[(None, None)] * A1.shape[1],
        method="highs",
    )
    return res.success and res.status == 0


def generate_random_polygon_set(
    n_polygons=4,
    intersect_polygons=False,
    q0=None,
    qd=None,
    max_vertices=20,
    radius_lim=(1e-1, 1.0),
    bbox=(-5, -5, 5, 5),
    seed=None,
    min_area=None,
    max_attempts=1000,
    radius=None,
    num_vertices=None,
):
    rng = np.random.default_rng(seed)
    polygons = []
    attempts = 0

    if min_area is None:
        min_area = np.pi / 2 * radius_lim[0] ** 2

    while len(polygons) < n_polygons and attempts < max_attempts:
        A, b, vertices, area = generate_random_polygon(
            max_vertices=max_vertices,
            radius_lim=radius_lim,
            bbox=bbox,
            seed=rng.integers(0, 1e9).item(),
            min_area=min_area,
            max_attempts=max_attempts,
            radius=radius,
            num_vertices=num_vertices,
        )
        attempts += 1

        if area < min_area:
            continue

        # Check q0, qd are outside
        if q0 is not None and is_point_inside_polygon(q0.ravel(), A, b):
            continue
        if qd is not None and is_point_inside_polygon(qd.ravel(), A, b):
            continue

        # Check intersections with previous polygons
        if not intersect_polygons:
            if any(polygons_intersect(A, b, Ap, bp) for (Ap, bp, _, _, _) in polygons):
                continue
        # Get center of polygon and radius of circumscribed circle
        center = np.mean(vertices, axis=0)
        radius_ = 1.001 * np.max(np.linalg.norm(vertices - center, axis=1)).item()

        # Passed all checks
        polygons.append((A, b, vertices, center, radius_))

    if attempts == max_attempts:
        raise RuntimeError("Too many attempts to generate non-overlapping polygons")

    return polygons


def generate_random_polyhedron_set(
    n_polyhedra=4,
    intersect_polyhedra=False,
    q0=None,
    qd=None,
    max_vertices=20,
    radius_lim=(1e-1, 1.0),
    dim=3,
    bbox=(-5, 5),
    seed=None,
    min_volume=None,
    max_attempts=1000,
    radius=None,
    num_vertices=None,
):
    rng = np.random.default_rng(seed)
    polyhedra = []
    attempts = 0

    if min_volume is None:
        # Half the volume of a n-sphere with radius equal to the minimum radius limit
        min_volume = (
            1 / 2 * (np.pi ** (dim / 2) / gamma(dim / 2 + 1)) * radius_lim[0] ** dim
        )

    while len(polyhedra) < n_polyhedra and attempts < max_attempts:
        A, b, vertices, volume = generate_random_polyhedron(
            max_vertices=max_vertices,
            radius_lim=radius_lim,
            bbox=bbox,
            seed=rng.integers(0, 1e9).item(),
            dim=dim,
            min_volume=min_volume,
            max_attempts=max_attempts,
            radius=radius,
            num_vertices=num_vertices,
        )
        attempts += 1

        if volume < min_volume:
            continue

        # Check q0, qd are outside
        if q0 is not None and is_point_inside_polygon(q0.ravel(), A, b):
            continue
        if qd is not None and is_point_inside_polygon(qd.ravel(), A, b):
            continue

        # Check intersections with previous polygons
        if not intersect_polyhedra:
            if any(polygons_intersect(A, b, Ap, bp) for (Ap, bp, _, _, _) in polyhedra):
                continue
        # Get center of polygon and radius of circumscribed circle
        center = np.mean(vertices, axis=0)
        radius_ = 1.001 * np.max(np.linalg.norm(vertices - center, axis=1)).item()

        # Passed all checks
        polyhedra.append((A, b, vertices, center, radius_))

    if attempts == max_attempts:
        raise RuntimeError("Too many attempts to generate non-overlapping polygons")

    return polyhedra


def create_level_sets(
    polygons,
    r=1e-1,
    h=1e-1,
    kind="both",
    bbox=(-5, -5, 5, 5),
    n_points=100,
    n_contours=50,
    test=None,
    method="ours",
    add_reference=False,
    rescale=False,
    return_cmap_data=False,
):
    fig = go.Figure()

    if not isinstance(polygons, (list, tuple)):
        polygons = [polygons]

    for polygon in polygons:
        if isinstance(polygon, NonConvexPolygon):
            for iii, poly in enumerate(polygon.polytopes):
                A = poly.A.copy()
                b = poly.b.copy()
                add_polygon(fig, A, b, aux=iii, add_reference=add_reference)
        elif isinstance(polygon, Polytope):
            add_polygon(fig, polygon.A, polygon.b, add_reference=add_reference)
        else:
            raise ValueError("Unknown polygon type")

    p1 = np.linspace(bbox[0], bbox[2], n_points)
    p2 = np.linspace(bbox[1], bbox[3], n_points)

    # Compute distances to single polygon:
    distances = []
    for x in p1:
        for y in p2:
            p = np.array([x, y]).reshape(-1, 1)
            dists = []
            for polygon in polygons:
                if isinstance(polygon, NonConvexPolygon):
                    dist_i = signed_dist2nonconvex(
                        # id_phi,
                        phi,
                        p,
                        polygon.A_list,
                        polygon.b_list,
                        polygon.shared_boundaries,
                        r=r,
                        h=h,
                        # test="out"
                        test=test,
                        # test=None
                    )
                elif isinstance(polygon, Polytope):
                    if method.lower() == "esdf":
                        dist_i, *_ = ESDF_CGAL(
                            p=p,
                            A=polygon.A,
                            b=polygon.b.reshape(-1, 1),
                        )
                    else:
                        dist_i, _ = signedDist2Convex(
                            p=p,
                            A=polygon.A,
                            b=polygon.b.reshape(-1, 1),
                            r=r,
                            eps=h,
                            test="" if test is None else test,
                        )
                        # dist_i = signed_dist2convex(
                        #     # id_phi,
                        #     phi,
                        #     p,
                        #     polygon.A,
                        #     polygon.b,
                        #     r=r,
                        #     h=h,
                        #     # test="out"
                        #     test=test,
                        #     # test=None
                        # )
                else:
                    raise ValueError("Unknown polygon type")
                dists.append(dist_i)
            distances.append(smoothMinList(dists, r=r))
            # distances.append(dist)

    distances = np.array(distances).reshape(n_points, n_points).T
    if rescale:
        smallest = np.min(distances)
        largest = np.max(distances)
        factor = largest / np.abs(smallest)
        distances[distances < 0] *= factor

    contour = go.Contour(
        x=p1,
        y=p2,
        z=distances,
        colorscale="RdBu",
        ncontours=n_contours,
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

    if not return_cmap_data:
        return fig
    else:
        return fig, (p1, p2, distances)
