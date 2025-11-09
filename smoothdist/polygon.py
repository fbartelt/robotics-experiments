import numpy as np
import plotly.graph_objects as go
from scipy.optimize import linprog
from scipy.spatial import HalfspaceIntersection, ConvexHull
from scipy.special import factorial


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
