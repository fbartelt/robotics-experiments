import os, sys
# sys.path.insert(0, "/home/fbartelt/Documents/Projetos/robotics-experiments/smoothdist/third_party/fcl/build/lib")
import uaibot as ub
import numpy as np
from sdf import compute_distance
from scipy.optimize import linprog
from scipy.spatial import ConvexHull, HalfspaceIntersection

# ---------- Halfspace → Mesh ----------
def compute_vertices_and_faces(A_, b_):
    """
    Convert convex polytope Ax <= b (m×3, m) to a triangular mesh.
    Returns (vertices (N,3), faces (list of ints, length 3*F)).
    """
    # 1. Find an interior point (Chebyshev centre)
    A, b = np.array(A_), np.array(b_)
    m, n = A.shape
    norm_A = np.linalg.norm(A, axis=1)
    c = np.zeros(n + 1)
    c[-1] = -1.0
    A_ub = np.hstack([A, norm_A.reshape(-1, 1)])
    res = linprog(c, A_ub=A_ub, b_ub=b, bounds=(None, None))
    if not res.success:
        raise RuntimeError("Cannot find interior point for polytope.")
    interior = res.x[:n].ravel()

    halfspaces = np.hstack([A, -b.reshape(-1, 1)])   # shape (m, 4)
    hs = HalfspaceIntersection(halfspaces, interior)
    verts = hs.intersections

    # 3. Convex hull of those vertices → faces
    hull = ConvexHull(verts)
    faces = hull.simplices.flatten().astype(np.int32).tolist()
    # FCL expects each triangular face to end with -1
    # fcl_faces = []
    # for i in range(0, len(faces), 3):
    #     fcl_faces += faces[i:i+3] + [-1]
    # faces = fcl_faces   # pass this to C++
    return verts.astype(np.float32), faces

def esdf(ub_obj1, ub_obj2, vertices1, vertices2, faces1, faces2):
    # vertices1, faces1 = np.zeros((0, 3)), np.zeros((0, 3))
    # vertices2, faces2 = np.zeros((0, 3)), np.zeros((0, 3))
    #
    # if isinstance(ub_obj1, ub.ConvexPolytope):
    #     vertices1, faces1 = compute_vertices_and_faces(ub_obj1.A, ub_obj1.b)
    # if isinstance(ub_obj2, ub.ConvexPolytope):
    #     vertices2, faces2 = compute_vertices_and_faces(ub_obj2.A, ub_obj2.b)

    return compute_distance(ub_obj1.cpp_obj, ub_obj2.cpp_obj, vertices1, faces1, vertices2, faces2)
