import numpy as np
import plotly.graph_objects as go
from scipy.optimize import linprog
from scipy.spatial import HalfspaceIntersection, ConvexHull
from scipy.special import factorial
from typing import List, Tuple, Optional
from distances import signed_dist2convex, id_phi, phi, smooth_min, signed_dist2nonconvex


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

def generate_random_polygon_set(
    n_polygons=4,
    intersect_polygons=False,
    q0=None,
    qd=None,
    max_vertices=20,
    radius_lim=(1e-1, 1.0),
    bbox=(-5, -5, 5, 5),
    min_area=None,
    max_attempts=1000,
    seed=None
):
    rng = np.random.default_rng(seed)
    polygons = []
    attempts = 0

    if min_area is None:
        min_area = np.pi/2 * radius_lim[0] ** 2

    while len(polygons) < n_polygons and attempts < max_attempts:
        A, b, vertices, area = generate_random_polygon(
            max_vertices=max_vertices,
            radius_lim=radius_lim,
            bbox=bbox,
            seed=rng.integers(1e9)
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
        radius = 1.001*np.max(np.linalg.norm(vertices - center, axis=1)).item()

        # Passed all checks
        polygons.append((A, b, vertices, center, radius))

    if attempts == max_attempts:
        raise RuntimeError("Too many attempts to generate non-overlapping polygons")

    return polygons


# --------------------------
# Polygon clipping helpers
# --------------------------
def intersect_segment_with_vertical(p: Tuple[float,float], q: Tuple[float,float], x0: float) -> Optional[Tuple[float,float]]:
    (x1,y1), (x2,y2) = p, q
    dx = x2 - x1
    if abs(dx) < 1e-15:
        return None
    t = (x0 - x1) / dx
    if -1e-12 <= t <= 1 + 1e-12:
        y = y1 + t*(y2-y1)
        return (x0, y)
    return None

def intersect_segment_with_horizontal(p: Tuple[float,float], q: Tuple[float,float], y0: float) -> Optional[Tuple[float,float]]:
    (x1,y1), (x2,y2) = p, q
    dy = y2 - y1
    if abs(dy) < 1e-15:
        return None
    t = (y0 - y1) / dy
    if -1e-12 <= t <= 1 + 1e-12:
        x = x1 + t*(x2-x1)
        return (x, y0)
    return None

def remove_duplicate_and_collinear(pts: List[Tuple[float,float]], eps: float = 1e-9) -> List[Tuple[float,float]]:
    """
    Remove consecutive duplicates and near-collinear points from polygon vertex list.
    pts assumed to be in order (closedness not required). Keeps at least 3 points if possible.
    """
    if not pts:
        return []
    # first remove consecutive duplicates (within eps)
    cleaned = [pts[0]]
    for p in pts[1:]:
        if np.hypot(p[0]-cleaned[-1][0], p[1]-cleaned[-1][1]) > eps:
            cleaned.append(p)
    # maybe last equals first -> make simple open list (we'll handle closure logic later)
    if len(cleaned) > 1 and np.hypot(cleaned[0][0]-cleaned[-1][0], cleaned[0][1]-cleaned[-1][1]) < eps:
        cleaned.pop()

    if len(cleaned) < 3:
        return cleaned

    # remove near-collinear triples
    def is_collinear(a,b,c, eps_col=1e-9):
        # area of triangle abc
        return abs((b[0]-a[0])*(c[1]-a[1]) - (b[1]-a[1])*(c[0]-a[0])) <= eps_col

    res = []
    n = len(cleaned)
    for i in range(n):
        prev = cleaned[(i-1) % n]
        cur  = cleaned[i]
        nxt  = cleaned[(i+1) % n]
        if is_collinear(prev, cur, nxt, eps):
            # skip cur (it is collinear) unless skipping would make polygon degenerate
            # we will skip now and keep minimal geometry; the modulo ensures correctness
            continue
        res.append(cur)

    # It's possible we removed too many; ensure at least 3 points (if original had >=3)
    if len(res) < 3 and len(cleaned) >= 3:
        # fallback: try less strict collinearity tolerance
        res = []
        for i in range(n):
            prev = cleaned[(i-1) % n]
            cur  = cleaned[i]
            nxt  = cleaned[(i+1) % n]
            if not is_collinear(prev, cur, nxt, eps*10):
                res.append(cur)
        if len(res) < 3:
            # last resort: return first 3 distinct points
            unique = []
            for p in cleaned:
                if p not in unique:
                    unique.append(p)
                if len(unique) == 3:
                    break
            return unique
    return res

def sutherland_hodgman_clip(poly: List[Tuple[float,float]], axis: str, lo: float, hi: float) -> List[Tuple[float,float]]:
    """
    Clip polygon 'poly' (list of (x,y) clockwise) against slab:
      if axis == 'y': keep vertices with lo <= x <= hi   (vertical slab)
      if axis == 'x': keep vertices with lo <= y <= hi   (horizontal slab)
    Uses Sutherland–Hodgman polygon clipping by applying two half-plane clips.
    Returns clipped polygon (may be empty).
    """
    def clip_halfplane(poly_pts, axis, side, value):
        # side: 'le' -> keep coord <= value ; 'ge' -> keep coord >= value
        out = []
        n = len(poly_pts)
        if n == 0:
            return out
        for i in range(n):
            cur = poly_pts[i]
            prev = poly_pts[i-1]
            if axis == 'y':
                cur_coord = cur[0]
                prev_coord = prev[0]
            else:
                cur_coord = cur[1]
                prev_coord = prev[1]

            def inside(coord):
                if side == 'le':
                    return coord <= value + 1e-12
                else:
                    return coord >= value - 1e-12

            cur_in = inside(cur_coord)
            prev_in = inside(prev_coord)

            if cur_in:
                if not prev_in:
                    # entering - compute intersection
                    if axis == 'y':
                        ipt = intersect_segment_with_vertical(prev, cur, value)
                    else:
                        ipt = intersect_segment_with_horizontal(prev, cur, value)
                    if ipt is not None:
                        out.append(ipt)
                out.append(cur)
            elif prev_in:
                # leaving - compute intersection, append it
                if axis == 'y':
                    ipt = intersect_segment_with_vertical(prev, cur, value)
                else:
                    ipt = intersect_segment_with_horizontal(prev, cur, value)
                if ipt is not None:
                    out.append(ipt)
        return out

    clipped1 = clip_halfplane(poly, axis, 'ge', lo)
    clipped2 = clip_halfplane(clipped1, axis, 'le', hi)
    # Remove duplicates and collinear points
    cleaned = remove_duplicate_and_collinear(clipped2, eps=1e-9)
    return cleaned

# --------------------------
# Half-space (A,b) from polygon vertices (clockwise)
# --------------------------
def polygon_to_A_b(poly: List[Tuple[float,float]]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Given polygon vertices in clockwise order, return A, b such that A @ x <= b.
    Each edge (p -> q) produces one inequality. We compute inward-pointing normal n_in = (e_y, -e_x)
    where e = q - p. For clockwise polygon, n_in points inside.
    """
    if len(poly) < 3:
        return np.zeros((0,2)), np.zeros((0,))
    A_rows = []
    b_rows = []
    for i in range(len(poly)):
        p = np.array(poly[i], dtype=float)
        q = np.array(poly[(i+1) % len(poly)], dtype=float)
        e = q - p
        n_in = np.array([e[1], -e[0]])
        norm = np.linalg.norm(n_in)
        if norm < 1e-12:
            continue
        n_in = n_in / norm
        A_row = -n_in  # so that A @ x <= b
        b_row = -n_in.dot(p)
        A_rows.append(A_row)
        b_rows.append(b_row)
    if not A_rows:
        return np.zeros((0,2)), np.zeros((0,))
    A = np.vstack(A_rows)
    b = np.array(b_rows)
    return A, b

# --------------------------
# Main decomposition function
# --------------------------
def trapezoidal_decompose_to_halfspaces(polygon: List[Tuple[float,float]],
                                        axis: str = 'y') -> List[dict]:
    """
    polygon: list of (x,y) in CLOCKWISE order (simple polygon).
    axis: 'y' -> vertical slices between unique x coords (default)
          'x' -> horizontal slices between unique y coords

    Returns a list of dicts, each with keys:
      - 'poly': list of polygon vertices (clockwise) of the cell
      - 'A': numpy array (m x 2)
      - 'b': numpy array (m,)
      - 'slab': (lo, hi) interval on the chosen axis
    """
    assert axis in ('x', 'y')
    coords = np.array(polygon)
    if axis == 'y':
        vals = np.unique(np.round(coords[:,0], 12))
    else:
        vals = np.unique(np.round(coords[:,1], 12))

    vals_sorted = np.sort(vals)
    cells = []

    for i in range(len(vals_sorted)-1):
        lo = float(vals_sorted[i])
        hi = float(vals_sorted[i+1])
        if abs(hi - lo) < 1e-12:
            continue
        clipped = sutherland_hodgman_clip(polygon, axis, lo, hi)
        if len(clipped) >= 3:
            pts = np.array(clipped)
            # ensure clockwise ordering (shoelace)
            area2 = 0.0
            for j in range(len(pts)):
                x1,y1 = pts[j]
                x2,y2 = pts[(j+1)%len(pts)]
                area2 += (x1*y2 - x2*y1)
            if area2 > 0:
                clipped = clipped[::-1]
            # final removal of collinear/duplicate
            clipped = remove_duplicate_and_collinear(clipped, eps=1e-9)
            if len(clipped) >= 3:
                A,b = polygon_to_A_b(clipped)
                cells.append({'poly': clipped, 'A': A, 'b': b, 'slab': (lo,hi)})
    return cells



# # --------------------------
# # Small helper structure
# # --------------------------
# class Node:
#     """
#     Optional: simple node graph representation.
#     You can build a polygon as a circular doubly-linked graph of Node,
#     with clockwise ordering: node.left and node.right are neighbors.
#     """
#     def __init__(self, x: float, y: float, left: 'Node' = None, right: 'Node' = None, idx: int = None):
#         self.x = float(x)
#         self.y = float(y)
#         self.left = left
#         self.right = right
#         self.idx = idx
#
#     def pos(self):
#         return (self.x, self.y)
#
# def graph_to_vertex_list(start_node: Node) -> List[Tuple[float, float]]:
#     """
#     Given a Node in a clockwise circular polygon graph, return list of (x,y)
#     in clockwise order starting from start_node.
#     """
#     verts = []
#     n = start_node
#     while True:
#         verts.append((n.x, n.y))
#         n = n.right
#         if n is start_node:
#             break
#     return verts
#
# # --------------------------
# # Polygon clipping helpers
# # --------------------------
# def intersect_segment_with_vertical(p: Tuple[float,float], q: Tuple[float,float], x0: float) -> Optional[Tuple[float,float]]:
#     """
#     Intersect segment pq with vertical line x = x0.
#     Returns intersection point or None if parallel/no intersection within segment.
#     """
#     (x1,y1), (x2,y2) = p, q
#     if (x2 - x1) == 0:
#         return None  # parallel or collinear vertical
#     t = (x0 - x1) / (x2 - x1)
#     if 0 <= t <= 1:
#         y = y1 + t*(y2-y1)
#         return (x0, y)
#     return None
#
# def intersect_segment_with_horizontal(p: Tuple[float,float], q: Tuple[float,float], y0: float) -> Optional[Tuple[float,float]]:
#     (x1,y1), (x2,y2) = p, q
#     if (y2 - y1) == 0:
#         return None
#     t = (y0 - y1) / (y2 - y1)
#     if 0 <= t <= 1:
#         x = x1 + t*(x2-x1)
#         return (x, y0)
#     return None
#
# def sutherland_hodgman_clip(poly: List[Tuple[float,float]], axis: str, lo: float, hi: float) -> List[Tuple[float,float]]:
#     """
#     Clip polygon 'poly' (list of (x,y) clockwise) against slab:
#       if axis == 'y': keep vertices with lo <= x <= hi   (vertical slab)
#       if axis == 'x': keep vertices with lo <= y <= hi   (horizontal slab)
#     Uses Sutherland–Hodgman polygon clipping by applying two half-plane clips.
#     Returns clipped polygon (may be empty).
#     """
#     def clip_halfplane(poly_pts, axis, side, value):
#         # side: 'le' -> keep coord <= value ; 'ge' -> keep coord >= value
#         out = []
#         n = len(poly_pts)
#         if n == 0:
#             return out
#         for i in range(n):
#             cur = poly_pts[i]
#             prev = poly_pts[i-1]
#             if axis == 'y':
#                 cur_coord = cur[0]
#                 prev_coord = prev[0]
#             else:
#                 cur_coord = cur[1]
#                 prev_coord = prev[1]
#
#             def inside(coord):
#                 if side == 'le':
#                     return coord <= value + 1e-12
#                 else:
#                     return coord >= value - 1e-12
#
#             cur_in = inside(cur_coord)
#             prev_in = inside(prev_coord)
#
#             if cur_in:
#                 if not prev_in:
#                     # entering - compute intersection
#                     if axis == 'y':
#                         ipt = intersect_segment_with_vertical(prev, cur, value)
#                     else:
#                         ipt = intersect_segment_with_horizontal(prev, cur, value)
#                     if ipt is not None:
#                         out.append(ipt)
#                 out.append(cur)
#             elif prev_in:
#                 # leaving - compute intersection, append it
#                 if axis == 'y':
#                     ipt = intersect_segment_with_vertical(prev, cur, value)
#                 else:
#                     ipt = intersect_segment_with_horizontal(prev, cur, value)
#                 if ipt is not None:
#                     out.append(ipt)
#         return out
#
#     # First clip with lo (>= lo)
#     clipped1 = clip_halfplane(poly, axis, 'ge', lo)
#     # Then clip with hi (<= hi)
#     clipped2 = clip_halfplane(clipped1, axis, 'le', hi)
#     # Remove near-duplicate consecutive points
#     def dedup(pts):
#         if not pts:
#             return pts
#         res = [pts[0]]
#         for p in pts[1:]:
#             if np.hypot(p[0]-res[-1][0], p[1]-res[-1][1]) > 1e-9:
#                 res.append(p)
#         # also check last vs first
#         if len(res) > 1 and np.hypot(res[0][0]-res[-1][0], res[0][1]-res[-1][1]) < 1e-9:
#             res.pop()
#         return res
#     return dedup(clipped2)
#
# # --------------------------
# # Half-space (A,b) from polygon vertices (clockwise)
# # --------------------------
# def polygon_to_A_b(poly: List[Tuple[float,float]]) -> Tuple[np.ndarray, np.ndarray]:
#     """
#     Given polygon vertices in clockwise order, return A, b such that A @ x <= b.
#     Each edge (p -> q) produces one inequality. We compute inward-pointing normal n_in = (e_y, -e_x)
#     where e = q - p. For clockwise polygon, n_in points inside. To produce A @ x <= b,
#     we set A_row = -n_in, b_row = -n_in.dot(p). That guarantees interior satisfies inequality.
#     """
#     if len(poly) < 3:
#         return np.zeros((0,2)), np.zeros((0,))
#     A_rows = []
#     b_rows = []
#     for i in range(len(poly)):
#         p = np.array(poly[i])
#         q = np.array(poly[(i+1) % len(poly)])
#         e = q - p
#         n_in = np.array([e[1], -e[0]])  # inward-pointing normal for clockwise polygon
#         # Normalize to avoid huge numbers (optional)
#         norm = np.linalg.norm(n_in)
#         if norm < 1e-12:
#             continue
#         n_in = n_in / norm
#         A_row = -n_in
#         b_row = -n_in.dot(p)
#         A_rows.append(A_row)
#         b_rows.append(b_row)
#     if not A_rows:
#         return np.zeros((0,2)), np.zeros((0,))
#     A = np.vstack(A_rows)
#     b = np.array(b_rows)
#     return A, b
#
# # --------------------------
# # Main decomposition function
# # --------------------------
# def trapezoidal_decompose_to_halfspaces(polygon: List[Tuple[float,float]],
#                                         axis: str = 'y') -> List[dict]:
#     """
#     polygon: list of (x,y) in CLOCKWISE order (simple polygon).
#     axis: 'y' -> vertical slices between unique x coords (default)
#           'x' -> horizontal slices between unique y coords
#
#     Returns a list of dicts, each with keys:
#       - 'poly': list of polygon vertices (clockwise) of the cell
#       - 'A': numpy array (m x 2)
#       - 'b': numpy array (m,)
#     """
#     assert axis in ('x', 'y')
#     coords = np.array(polygon)
#     if axis == 'y':
#         vals = np.unique(np.round(coords[:,0], 12))
#         coord_axis_index = 0
#     else:
#         vals = np.unique(np.round(coords[:,1], 12))
#         coord_axis_index = 1
#
#     # sort unique coordinates
#     vals_sorted = np.sort(vals)
#     cells = []
#
#     # We'll build slabs between consecutive unique coords.
#     for i in range(len(vals_sorted)-1):
#         lo = float(vals_sorted[i])
#         hi = float(vals_sorted[i+1])
#         if abs(hi - lo) < 1e-12:
#             continue
#         clipped = sutherland_hodgman_clip(polygon, axis, lo, hi)
#         if len(clipped) >= 3:
#             # Ensure clockwise ordering (clip can maintain ordering, but let's be safe)
#             # compute signed area
#             pts = np.array(clipped)
#             area2 = 0.0
#             for j in range(len(pts)):
#                 x1,y1 = pts[j]
#                 x2,y2 = pts[(j+1)%len(pts)]
#                 area2 += (x1*y2 - x2*y1)
#             # area2 < 0 → clockwise (since standard shoelace positive for CCW)
#             if area2 > 0:
#                 clipped = clipped[::-1]
#             A,b = polygon_to_A_b(clipped)
#             cells.append({'poly': clipped, 'A': A, 'b': b, 'slab': (lo,hi)})
#     # Optionally, there might be cells exactly on the boundary at the extreme x (rare). We omit zero-width slabs.
#     return cells


def create_level_sets(
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
    add_reference=False,
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
                    dist_i = signed_dist2convex(
                        # id_phi,
                        phi,
                        p,
                        polygon.A,
                        polygon.b,
                        r=r,
                        h=h,
                        # test="out"
                        test=test,
                        # test=None
                    )
                else:
                    raise ValueError("Unknown polygon type")
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


