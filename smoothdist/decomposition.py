import numpy as np
from typing import List, Tuple, Optional


# --------------------------
# Polygon clipping helpers
# --------------------------
def intersect_segment_with_vertical(
    p: Tuple[float, float], q: Tuple[float, float], x0: float
) -> Optional[Tuple[float, float]]:
    (x1, y1), (x2, y2) = p, q
    dx = x2 - x1
    if abs(dx) < 1e-15:
        return None
    t = (x0 - x1) / dx
    if -1e-12 <= t <= 1 + 1e-12:
        y = y1 + t * (y2 - y1)
        return (x0, y)
    return None


def intersect_segment_with_horizontal(
    p: Tuple[float, float], q: Tuple[float, float], y0: float
) -> Optional[Tuple[float, float]]:
    (x1, y1), (x2, y2) = p, q
    dy = y2 - y1
    if abs(dy) < 1e-15:
        return None
    t = (y0 - y1) / dy
    if -1e-12 <= t <= 1 + 1e-12:
        x = x1 + t * (x2 - x1)
        return (x, y0)
    return None


def remove_duplicate_and_collinear(
    pts: List[Tuple[float, float]], eps: float = 1e-9
) -> List[Tuple[float, float]]:
    """
    Remove consecutive duplicates and near-collinear points from polygon vertex list.
    pts assumed to be in order (closedness not required). Keeps at least 3 points if possible.
    """
    if not pts:
        return []
    # first remove consecutive duplicates (within eps)
    cleaned = [pts[0]]
    for p in pts[1:]:
        if np.hypot(p[0] - cleaned[-1][0], p[1] - cleaned[-1][1]) > eps:
            cleaned.append(p)
    # maybe last equals first -> make simple open list (we'll handle closure logic later)
    if (
        len(cleaned) > 1
        and np.hypot(cleaned[0][0] - cleaned[-1][0], cleaned[0][1] - cleaned[-1][1])
        < eps
    ):
        cleaned.pop()

    if len(cleaned) < 3:
        return cleaned

    # remove near-collinear triples
    def is_collinear(a, b, c, eps_col=1e-9):
        # area of triangle abc
        return (
            abs((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]))
            <= eps_col
        )

    res = []
    n = len(cleaned)
    for i in range(n):
        prev = cleaned[(i - 1) % n]
        cur = cleaned[i]
        nxt = cleaned[(i + 1) % n]
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
            prev = cleaned[(i - 1) % n]
            cur = cleaned[i]
            nxt = cleaned[(i + 1) % n]
            if not is_collinear(prev, cur, nxt, eps * 10):
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


def sutherland_hodgman_clip(
    poly: List[Tuple[float, float]], axis: str, lo: float, hi: float
) -> List[Tuple[float, float]]:
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
            prev = poly_pts[i - 1]
            if axis == "y":
                cur_coord = cur[0]
                prev_coord = prev[0]
            else:
                cur_coord = cur[1]
                prev_coord = prev[1]

            def inside(coord):
                if side == "le":
                    return coord <= value + 1e-12
                else:
                    return coord >= value - 1e-12

            cur_in = inside(cur_coord)
            prev_in = inside(prev_coord)

            if cur_in:
                if not prev_in:
                    # entering - compute intersection
                    if axis == "y":
                        ipt = intersect_segment_with_vertical(prev, cur, value)
                    else:
                        ipt = intersect_segment_with_horizontal(prev, cur, value)
                    if ipt is not None:
                        out.append(ipt)
                out.append(cur)
            elif prev_in:
                # leaving - compute intersection, append it
                if axis == "y":
                    ipt = intersect_segment_with_vertical(prev, cur, value)
                else:
                    ipt = intersect_segment_with_horizontal(prev, cur, value)
                if ipt is not None:
                    out.append(ipt)
        return out

    clipped1 = clip_halfplane(poly, axis, "ge", lo)
    clipped2 = clip_halfplane(clipped1, axis, "le", hi)
    # Remove duplicates and collinear points
    cleaned = remove_duplicate_and_collinear(clipped2, eps=1e-9)
    return cleaned


# --------------------------
# Half-space (A,b) from polygon vertices (clockwise)
# --------------------------
def polygon_to_A_b(poly: List[Tuple[float, float]]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Given polygon vertices in clockwise order, return A, b such that A @ x <= b.
    Each edge (p -> q) produces one inequality. We compute inward-pointing normal n_in = (e_y, -e_x)
    where e = q - p. For clockwise polygon, n_in points inside.
    """
    if len(poly) < 3:
        return np.zeros((0, 2)), np.zeros((0,))
    A_rows = []
    b_rows = []
    for i in range(len(poly)):
        p = np.array(poly[i], dtype=float)
        q = np.array(poly[(i + 1) % len(poly)], dtype=float)
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
        return np.zeros((0, 2)), np.zeros((0,))
    A = np.vstack(A_rows)
    b = np.array(b_rows)
    return A, b


# --------------------------
# Main decomposition function
# --------------------------
def trapezoidal_decompose_to_halfspaces(
    polygon: List[Tuple[float, float]], axis: str = "y"
) -> List[dict]:
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
    assert axis in ("x", "y")
    coords = np.array(polygon)
    if axis == "y":
        vals = np.unique(np.round(coords[:, 0], 12))
    else:
        vals = np.unique(np.round(coords[:, 1], 12))

    vals_sorted = np.sort(vals)
    cells = []

    for i in range(len(vals_sorted) - 1):
        lo = float(vals_sorted[i])
        hi = float(vals_sorted[i + 1])
        if abs(hi - lo) < 1e-12:
            continue
        clipped = sutherland_hodgman_clip(polygon, axis, lo, hi)
        if len(clipped) >= 3:
            pts = np.array(clipped)
            # ensure clockwise ordering (shoelace)
            area2 = 0.0
            for j in range(len(pts)):
                x1, y1 = pts[j]
                x2, y2 = pts[(j + 1) % len(pts)]
                area2 += x1 * y2 - x2 * y1
            if area2 > 0:
                clipped = clipped[::-1]
            # final removal of collinear/duplicate
            clipped = remove_duplicate_and_collinear(clipped, eps=1e-9)
            if len(clipped) >= 3:
                A, b = polygon_to_A_b(clipped)
                cells.append({"poly": clipped, "A": A, "b": b, "slab": (lo, hi)})
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
