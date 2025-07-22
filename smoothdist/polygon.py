import numpy as np
import plotly.graph_objects as go
from scipy.optimize import linprog
from scipy.spatial import HalfspaceIntersection, ConvexHull
from scipy.special import factorial


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

    res = linprog(c, A_ub=A_lp, b_ub=b, bounds=bounds, method='highs')
    if res.success:
        return res.x[:-1]  # Return x (ignore δ)
    else:
        raise ValueError("Could not find a strictly feasible point.")


def add_polygon(fig, A, b):
    interior_point = find_strictly_feasible_point(A, b)
    halfspaces = np.hstack((A, -b[:, None]))
    hs = HalfspaceIntersection(halfspaces, interior_point)
    reconstructed_vertices = hs.intersections

    # Use ConvexHull to order them
    hull = ConvexHull(reconstructed_vertices)
    ordered_vertices = reconstructed_vertices[hull.vertices]

    x = np.append(ordered_vertices[:, 0], ordered_vertices[0, 0])
    y = np.append(ordered_vertices[:, 1], ordered_vertices[0, 1])

    fig.add_trace(go.Scatter(
        x=x, y=y, fill="toself",
        fillcolor="rgba(163, 159, 158, 0.2)",
        line=dict(color="rgba(163, 159, 158, 1)"),
    ))

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
