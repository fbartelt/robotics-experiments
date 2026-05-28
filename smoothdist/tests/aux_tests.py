import sys
import os
import pickle
import itertools
import numpy as np
import uaibot as ub
import uaibot_cpp_bind as ub_cpp
import plotly.graph_objects as go
import webbrowser
from time import time, perf_counter

# sys.path.insert(0, "/home/fbartelt/Projects/robotics-experiments/smoothdist")
# sys.path.insert(0, "/home/fbartelt/Documents/Projetos/robotics-experiments/smoothdist/")
from euclidean_sdf import esdf, compute_vertices_and_faces
from scipy.spatial import ConvexHull
from pathlib import Path
from plotly.subplots import make_subplots
from multiprocessing import Pool, cpu_count

import hashlib
from collections import Counter
import numpy as np


def _array_fingerprint(arr):
    """Return a hashable tuple that uniquely represents the array's data."""
    # Ensure C-contiguous layout for consistent bytes
    data = np.ascontiguousarray(arr)
    # Hash the raw bytes; combine with shape and dtype string
    digest = hashlib.sha256(data.tobytes()).digest()
    return (arr.shape, arr.dtype.str, digest)


def lists_of_arrays_equal(list1, list2):
    """Check if two lists of arrays contain exactly the same multiset of arrays."""
    if len(list1) != len(list2):
        return False
    c1 = Counter(_array_fingerprint(a) for a in list1)
    c2 = Counter(_array_fingerprint(a) for a in list2)
    return c1 == c2


def open_in_browser(filename: str):
    """
    Opens an HTML file in the system's default web browser.
    Works cross-platform (Linux, macOS, Windows).
    """
    path = Path(filename).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    # Convert to file:// URL and open
    webbrowser.open_new_tab(path.as_uri())


def create_platonic_solid(n_faces, radius=1.0, *args, **kwargs):
    """
    Create a platonic solid with a given number of faces.

    Parameters
    ----------
    n_faces : int
        Number of faces. Supported: 4 (tetrahedron), 6 (cube), 8
        (octahedron), 12 (dodecahedron), 20 (icosahedron).
    radius : float, optional
        Circumscribed sphere radius. Default is 1.0.
    *args, **kwargs
        Additional arguments passed to the ConvexPolytope constructor,
        such as 'htm' for the homogeneous transformation matrix.

    Returns
    -------
    ub.ConvexPolytope
    """
    phi = (1 + np.sqrt(5)) / 2  # golden ratio

    # Canonical vertices with circumradius = 1 (local frame, center at origin)
    match n_faces:
        case 4:  # tetrahedron
            v = np.array(
                [[1.0, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]]
            ) / np.sqrt(3)

        case 6:  # cube
            v = np.array(
                [
                    [-1.0, -1, -1],
                    [-1, -1, 1],
                    [-1, 1, -1],
                    [-1, 1, 1],
                    [1, -1, -1],
                    [1, -1, 1],
                    [1, 1, -1],
                    [1, 1, 1],
                ]
            ) / np.sqrt(3)

        case 8:  # octahedron
            v = np.array(
                [[1.0, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]]
            )  # already radius 1

        case 12:  # dodecahedron
            # all 20 vertices have norm √3 – scale to 1
            verts = []
            # (±1, ±1, ±1)
            for x in (-1, 1):
                for y in (-1, 1):
                    for z in (-1, 1):
                        verts.append([x, y, z])
            # (0, ±φ, ±1/φ) and cyclic permutations
            for a in (-phi, phi):
                for b in (-1 / phi, 1 / phi):
                    verts.append([0, a, b])
                    verts.append([b, 0, a])
                    verts.append([a, b, 0])
            v = np.array(verts) / np.sqrt(3)

        case 20:  # icosahedron
            verts = []
            norm_factor = np.sqrt(1 + phi**2)
            # cyclic permutations of (0, ±1, ±φ)
            for a in (-1, 1):
                for b in (-phi, phi):
                    verts.append([0, a, b])
                    verts.append([b, 0, a])
                    verts.append([a, b, 0])
            v = np.array(verts) / norm_factor

        case _:
            raise ValueError(
                f"Unsupported number of faces: {n_faces}. "
                "Must be one of {4, 6, 8, 12, 20}."
            )

    # Apply radius scaling
    v *= radius
    # Uaibot already converts A, b to world frame using the htm, so we
    # can just compute in local frame and let it handle the transformation.

    # Compute half‑space representation (Ax ≤ b) from convex hull
    hull = ConvexHull(v)
    A = hull.equations[:, :3]  # outward normals
    b = -hull.equations[:, 3]  # right‑hand side
    eqs = hull.equations
    # Normalize so that normal is unit (already unit if from Qhull) and offset sign consistent
    # For outward normals, all offset signs should be positive? Not necessarily, but we can
    # group by the plane parameters. Better: round to some tolerance.
    unique_planes = {}
    tolerance = 1e-10
    for eq in eqs:
        # round to tolerance to group identical planes
        key = tuple(np.round(eq / np.linalg.norm(eq[:3]), decimals=10))
        if key not in unique_planes:
            unique_planes[key] = eq
    A = np.array([v[:3] for v in unique_planes.values()])
    b = -np.array([v[3] for v in unique_planes.values()])
    # Now A has one row per distinct face
    # print(f"Created {n_faces}-face solid with {len(A)} unique planes (originally {len(hull.equations)})")

    # The default htm must be the identity. Otherwise, uaibot will compute
    # one aligned with a vertex.
    htm = kwargs.pop("htm", np.eye(4))
    poly = ub.ConvexPolytope(A=A, b=b, htm=np.eye(4), *args, **kwargs)
    poly.set_ani_frame(htm=htm)
    print(f"htm: {htm}")
    return poly


# %%
cube = create_platonic_solid(
    6, np.sqrt(3) / 2, htm=ub.Utils.trn([0, 0, 0.5]), color="blue"
)
A = np.array(
    [
        [1.0, 0, 0],
        [-1, 0, 0],
        [0, 1, 0],
        [0, -1, 0],
        [0, 0, 1],
        [0, 0, -1],
    ]
)
b1 = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5])
poly = ub.ConvexPolytope(A=A, b=b1, opacity=0.5, htm=np.eye(4))
# poly.set_ani_frame(htm=ub.Utils.trn([0, 0, 0.0]))
sim = ub.Simulation.create_sim_grid([cube])
sim.save("/tmp", "test")
open_in_browser("/tmp/test.html")
poly.A
cube.A
cube.htm
# %%
"""INVESTIGATE: HD-SDF returns 0 (x) and CHD-SDF a negative (v)"""
htm1 = ub.Utils.trn([0, 0, 0.0])
trn2 = np.array([0.0, 0, 0.5])
htm2 = ub.Utils.trn(trn2)
lx, ly, lz = 1.0, 1.0, 1.0
ubobj1 = ub.Box(htm=htm1, width=lx, depth=ly, height=lz, color="blue")
a = 0.2
lx, ly, lz = a, a, a
ubobj2 = ub.Box(htm=htm2, width=lx, depth=ly, height=lz)
# Create the same cube with Ax <= b
A = np.array(
    [
        [1.0, 0, 0],
        [-1, 0, 0],
        [0, 1, 0],
        [0, -1, 0],
        [0, 0, 1],
        [0, 0, -1],
    ]
)
b1 = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5])
b2 = b1 * a
# b2 = b1 - (A @ trn2.reshape(-1, 1)).ravel()

ubpoly1 = ub.ConvexPolytope(A=A, b=b1, htm=htm1)
ubpoly2 = ub.ConvexPolytope(A=A, b=b2, htm=htm2)
# ubpoly2.add_ani_frame(time=0.0, htm=htm2)

epsilon = 1e-3
gamma = 2
chdsdf, *graidents = ubobj1.signed_distance(
    ubobj2, gamma=gamma, epsilon=epsilon, is_conservative=True
)
hdsdf, *graidents = ubobj1.signed_distance(
    ubobj2, gamma=gamma, epsilon=epsilon, is_conservative=False
)
print()
print(f"Using ub.Box\nHD-SDF: {hdsdf}\nCHD-SDF: {chdsdf}")
cd2, *graidents = ubpoly1.signed_distance(
    ubpoly2, gamma=gamma, epsilon=epsilon, is_conservative=True
)
d2, *graidents = ubpoly1.signed_distance(
    ubpoly2, gamma=gamma, epsilon=epsilon, is_conservative=False
)
print()
print(f"Using ub.ConvexPolytope\nHD-SDF: {d2}\nCHD-SDF: {cd2}")

sim = ub.Simulation.create_sim_grid([ubobj1, ubobj2])
sim.save("/tmp", "test")
open_in_browser("/tmp/test.html")

# %%
""" CHECK MOVMENTS"""
htm1 = ub.Utils.trn([0, 0, 0.0])
trn2 = np.array([-2.0, 0, 0.0])
htm2 = ub.Utils.trn(trn2)
lx, ly, lz = 1.0, 1.0, 1.0
ubobj1 = ub.Box(htm=htm1, width=lx, depth=ly, height=lz, color="blue")
a = 0.2
ubobj2 = ub.Box(htm=htm2, width=a, depth=a, height=a)
# Create the same cube with Ax <= b
A = np.array(
    [
        [1.0, 0, 0],
        [-1, 0, 0],
        [0, 1, 0],
        [0, -1, 0],
        [0, 0, 1],
        [0, 0, -1],
    ]
)
b1 = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5])
b2 = b1 * a
# b2 = b1 - (A @ trn2.reshape(-1, 1)).ravel()

# ubpoly1 = ub.ConvexPolytope(A=A, b=b1, htm=htm1)
# ubpoly2 = ub.ConvexPolytope(A=A, b=b2, htm=htm2)
ubpoly1 = create_platonic_solid(6, radius=lx * np.sqrt(3) / 2, color="blue")
ubpoly2 = create_platonic_solid(6, radius=a * np.sqrt(3) / 2, htm=htm2)

epsilon = 1e-3
eps_edge = 1e-6
gamma = 2

dists = {"hd-sdf": [], "chd-sdf": [], "poly-hd": [], "poly-chd": []}
T = 5.0
dt = 1e-3
imax = int(T / dt)
# ubpoly1.add_ani_frame(time=0.0, htm=ub.Utils.trn([2.0, 0.0, 0.0]))
dists["hd-sdf"]

ubpoly1_cpp = ub.Utils.obj_to_cpp(ubpoly1)
vertices1 = np.array(ubpoly1_cpp.vertices_local)

for i in range(imax):
    displacement = np.array([0.001, 0.0, 0.0])
    # displacement = np.zeros((3,))
    trn = ub.Utils.trn(displacement)
    htm_curr = ubobj2.htm
    htm_poly = ubpoly2.htm
    # rot = ub.Utils.rotx(2 * np.pi * i * dt / T)
    rot = np.eye(4)

    chdsdf, *graidents = ubobj1.signed_distance(
        ubobj2, gamma=gamma, epsilon=epsilon, is_conservative=True
    )
    hdsdf, *graidents = ubobj1.signed_distance(
        ubobj2, gamma=gamma, epsilon=epsilon, is_conservative=False
    )
    dists["hd-sdf"].append(hdsdf)
    dists["chd-sdf"].append(chdsdf)

    # verts1 = (htm @ np.hstack([vertices1, np.ones((vertices1.shape[0], 1))]).T).T[:, :-1]
    # ubpoly2_cpp = ub.Utils.obj_to_cpp(ubpoly2)
    ubpoly2_cpp = ub_cpp.CPP_GeometricPrimitives.create_convexpolytope(
        ubpoly2.htm, A, b2
    )
    ubbox1_cpp = ub.Utils.obj_to_cpp(ubobj1)
    ubbox2_cpp = ub.Utils.obj_to_cpp(ubobj2)

    vertices2 = np.array(ubpoly2_cpp.vertices_local)
    verts2 = np.array(
        htm_curr @ np.hstack([vertices2, np.ones((vertices2.shape[0], 1))]).T
    ).T[:, :-1]
    boxverts = np.array(ubobj2.get_vertices())
    if np.any(np.isnan(verts2)):
        print("NAN")
        print(vertices2)
        print(htm_curr)
        break

    stp = False
    verts2_l = [np.round(np.array(v).astype(np.float32), 1) for v in verts2.tolist()]
    boxverts_l = [
        np.round(np.array(v).astype(np.float32), 1) for v in boxverts.tolist()
    ]
    # if not lists_of_arrays_equal(verts2_l, boxverts_l):
    #     print(verts2)
    #     print(boxverts)
    #     print("Different vertices")
    #     print(ubpoly2.A)
    #     print(ubpoly2.b)
    #     print(ubpoly2_cpp.A)
    #     print(ubpoly2_cpp.b)
    #     print(ubpoly2_cpp.A_local)
    #     print(ubpoly2_cpp.b_local)
    # stp = True
    normals = ub_cpp.get_candidate_normals(ubpoly1_cpp, ubpoly2_cpp, False, 1e-6)
    normals2 = ub_cpp.get_candidate_normals(ubbox1_cpp, ubbox2_cpp, False, 1e-6)
    # if not lists_of_arrays_equal(normals[0], normals2[0]):
    #     print("FACE NORMALS A do not match!!!")
    #     print(normals[0], normals2[0])
    #     stp = True
    # if not lists_of_arrays_equal(normals[1], normals2[1]):
    #     print("FACE NORMALS B do not match!!!")
    #     print(normals[1], normals2[1])
    #     stp = True
    # if not lists_of_arrays_equal(normals[2], normals2[2]):
    #     print("EDGE NORMALS do not match!!!")
    #     print(np.array(normals[2]), np.array(normals2[2]))
    #     stp = True
    # polyhdsdf, *_ = ub_cpp.holder_distance(
    #     vertices1, verts2, normals[0], normals[1], normals[2], gamma, True, epsilon
    # )
    #
    # normals_c = ub_cpp.get_candidate_normals(ubpoly1_cpp, ubpoly2_cpp, True, 1e-6)
    # polychdsdf, *_ = ub_cpp.holder_distance(
    #     vertices1, verts2, normals_c[0], normals_c[1], normals_c[2], gamma, True, epsilon
    # )
    polychdsdf, *graidents = ubpoly1.signed_distance(
        ubpoly2,
        gamma=gamma,
        epsilon=epsilon,
        is_conservative=True,
        eps_edge=1e-6,
    )
    polyhdsdf, *graidents = ubpoly1.signed_distance(
        ubpoly2, gamma=gamma, epsilon=epsilon, is_conservative=False, eps_edge=1e-6
    )
    dists["poly-hd"].append(polyhdsdf)
    dists["poly-chd"].append(polychdsdf)

    htm_new = htm_curr @ trn @ rot
    htm_new_poly = htm_poly @ trn @ rot

    ubobj2.add_ani_frame(time=i * dt, htm=htm_new)
    ubpoly2.add_ani_frame(time=i * dt, htm=htm_new_poly)
    if stp:
        break


def dist_plot(hdsdf, chdsdf, polyhdsdf, polychdsdf):
    # Create plot
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=np.arange(imax) * dt,
            y=hdsdf,
            mode="lines",
            name="HD-SDF",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=np.arange(imax) * dt,
            y=chdsdf,
            mode="lines",
            name="CHD-SDF",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=np.arange(imax) * dt,
            y=polyhdsdf,
            mode="lines",
            name="(Polytope) HD-SDF",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=np.arange(imax) * dt,
            y=polychdsdf,
            mode="lines",
            name="(Polytope) CHD-SDF",
        )
    )
    fig.update_layout(
        title="Distance Comparison",
        xaxis_title="Time (s)",
        yaxis_title="Distance (m)",
    )
    fig.show()


dist_plot(dists["hd-sdf"], dists["chd-sdf"], dists["poly-hd"], dists["poly-chd"])

sim = ub.Simulation.create_sim_grid([ubobj1, ubobj2, ubpoly1, ubpoly2])
sim.save("/tmp", "test")
open_in_browser("/tmp/test.html")
