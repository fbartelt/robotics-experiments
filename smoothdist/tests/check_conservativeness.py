# %%
import sys
import os
import pickle
import itertools
import numpy as np
import uaibot as ub
import plotly.graph_objects as go
import webbrowser
from time import time, perf_counter

sys.path.insert(0, "/home/fbartelt/Projects/robotics-experiments/smoothdist")
sys.path.insert(0, "/home/fbartelt/Documents/Projetos/robotics-experiments/smoothdist/")
from euclidean_sdf import esdf, compute_vertices_and_faces
from scipy.spatial import ConvexHull
from pathlib import Path
from plotly.subplots import make_subplots
from multiprocessing import Pool, cpu_count
from tqdm import tqdm


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


def random_htm(
    translation_max=[1.0, 1.0, 1.0],
    translation_min=[-1.0, -1.0, -1.0],
    rotation_max=[2 * np.pi, 2 * np.pi, 2 * np.pi],
    rotation_min=[0.0, 0.0, 0.0],
    rng=None,
    seed=None,
):
    if rng is None:
        if seed is not None:
            rng = np.random.default_rng(seed)
        else:
            rng = np.random.default_rng()
    translation = rng.uniform(translation_min, translation_max)
    rotation = rng.uniform(rotation_min, rotation_max)
    R = (
        ub.Utils.rotx(rotation[0])
        @ ub.Utils.roty(rotation[1])
        @ ub.Utils.rotz(rotation[2])
    )
    htm = np.array(ub.Utils.trn(translation) @ R)
    return htm


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
    htm = kwargs.pop("htm", np.eye(4))
    ubobj = ub.ConvexPolytope(A=A, b=b, *args, **kwargs)
    ubobj.set_ani_frame(htm=htm)
    return ubobj

def get_box_verticesNplanes(box):
    htm = np.array(box.htm)
    R, p = htm[:3, :3], htm[:3, -1].reshape(-1, 1)
    lx, ly, lz = box.width / 2, box.depth / 2, box.height / 2
    local_verts = [
        np.array([-lx, -ly, -lz]),
        np.array([lx, -ly, -lz]),
        np.array([lx, ly, -lz]),
        np.array([-lx, ly, -lz]),
        np.array([-lx, -ly, lz]),
        np.array([lx, -ly, lz]),
        np.array([lx, ly, lz]),
        np.array([-lx, ly, lz]),
    ]
    verts = np.array([
        R @ v.reshape(-1, 1) + p for v in local_verts
    ]).reshape(-1, 3)

    hull = ConvexHull(verts)
    faces = hull.simplices.flatten().astype(np.int32).tolist()
    # FCL expects each triangular face to end with -1
    # fcl_faces = []
    # for i in range(0, len(faces), 3):
    #     fcl_faces += faces[i:i+3] + [-1]
    # faces = fcl_faces   # pass this to C++
    return verts.astype(np.float32), faces

# ----------------------------------------------------------------------
# 1.  Core distance function – returns (dist_sdf, dist_hdsdf)
# ----------------------------------------------------------------------
def compute_both_distances(seed, params):
    """
    Generate a random configuration from `seed` and compute both
    the exact SDF and the HD‑SDF distances.
    Returns (dist_sdf, dist_hdsdf).
    """
    rng = np.random.default_rng(seed)

    # ---- unpack parameters ----
    t_max = params["translation_max"]
    t_min = params["translation_min"]
    r_max = params["rotation_max"]
    r_min = params["rotation_min"]
    radius = params["max_radius"]
    n_f1 = params["n_faces1"]
    n_f2 = params["n_faces2"]
    gamma = params["gamma"]
    skip_grad = params["skip_gradients"]
    eps_edge = params["eps_edge"]

    # ---- create the two polyhedra ----
    # htm1 = random_htm(t_max, t_min, r_max, r_min, rng)
    htm1 = np.eye(4)
    # obj1 = create_platonic_solid(n_faces=n_f1, radius=radius, htm=htm1)
    obj1 = ub.Box(width=radius, height=radius, depth=radius, htm=htm1)
    htm2 = random_htm(t_max, t_min, r_max, r_min, rng)
    obj2 = ub.Box(width=radius, height=radius, depth=radius, htm=htm2)
    # obj2 = create_platonic_solid(n_faces=n_f2, radius=radius, htm=htm2)

    # ---- SDF (exact) ----
    # vertices1, faces1 = compute_vertices_and_faces(obj1.A, obj1.b)
    # vertices2, faces2 = compute_vertices_and_faces(obj2.A, obj2.b)
    vertices1, faces1 = get_box_verticesNplanes(obj1)
    vertices2, faces2 = get_box_verticesNplanes(obj2)
    dist_sdf, _ = esdf(obj1, obj2, vertices1, vertices2, faces1, faces2)
    # esdf returns (distance, time_us); we only need distance

    # ---- HD‑SDF (differentiable, signed) ----
    dist_hd, *_ = obj1.signed_distance(
        obj2,
        gamma=gamma,
        is_conservative=False,
        skip_gradient=skip_grad,
        epsilon=1e-6,
        eps_edge=eps_edge,
    )

    return dist_sdf, dist_hd


# ----------------------------------------------------------------------
# 2.  Worker for parallel mapping – unpacks arguments and calls the above
# ----------------------------------------------------------------------
def worker(task):
    """
    task: (solidA, solidB, seed, base_params, faces)
    Returns: (solidA, solidB, seed, dist_sdf, dist_hd)
    """
    solidA, solidB, seed, base_params, faces, eps = task

    # Build pair‑specific parameters
    params = base_params.copy()
    params["n_faces1"] = faces[solidA]
    params["n_faces2"] = faces[solidB]
    params["eps_edge"] = eps

    dist_sdf, dist_hd = compute_both_distances(seed, params)
    return solidA, solidB, seed, dist_sdf, dist_hd, eps


# ----------------------------------------------------------------------
# 3.  False‑positive rate vs tolerance
# ----------------------------------------------------------------------
def false_positive_rates(sdf_all, hd_all, tolerances):
    """
    sdf_all, hd_all : 1-D arrays of distances (same length)
    tolerances       : array of thresholds to test
    Returns array of FP percentages.
    """
    sdf = np.asarray(sdf_all)
    hd = np.asarray(hd_all)
    rates = []
    for tol in tolerances:
        mask = (sdf >= tol) & (sdf <= 1e3) # Avoids inf values
        if mask.sum() == 0:
            rates.append(0.0)
        else:
            fp = (hd <= 0.0) & mask
            rates.append(fp.sum() / mask.sum() * 100.0)
    return np.array(rates)


def plot_fp_rate(tolerances, fp_rates, legend_labels, tol_zero_rate=None, title=None):
    """Plotly figure for the false‑positive curve."""
    fig = go.Figure()
    for fp, lbl, in zip(fp_curves, legend_labels):
        fig.add_trace(
            go.Scatter(
                x=tolerances,
                y=fp,
                mode="lines+markers",
                name=lbl,
                line=dict(width=2.5),
                marker=dict(size=6),
            )
        )
    if tol_zero_rate is not None:
        fig.add_annotation(
            x=tolerances[0],
            y=tol_zero_rate,
            text=f"tol = 0: {tol_zero_rate:.2f}%",
            showarrow=True,
            arrowhead=2,
            ax=0,
            ay=-20,
        )
    fig.update_xaxes(type="log", title_text="Tolerance (SDF ≥ tol)")
    fig.update_yaxes(
        title_text="False‑positive rate (%)"#, range=[0, max(fp_rates) * 1.1]
    )
    fig.update_layout(
        title=title if title else "HD‑SDF false positives vs. tolerance",
        template="plotly_white",
        hovermode="x unified",
    )
    return fig


# ----------------------------------------------------------------------
# 4.  Main – parameter setup, parallel execution, plotting
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # ---------- Configuration ----------
    platonic_solids = ["tetra", "cube", "octa", "dodeca", "icosa"]
    faces = {"tetra": 4, "cube": 6, "octa": 8, "dodeca": 12, "icosa": 20}

    base_params = {
        "translation_max": [2.0, 2.0, 2.0],
        "translation_min": [-2.0, -2.0, -2.0],
        "rotation_max": [2 * np.pi, 2 * np.pi, 2 * np.pi],
        "rotation_min": [0.0, 0.0, 0.0],
        "max_radius": 1.0,
        "gamma": 2,
        "skip_gradients": True,
    }

    N = 500000  # configurations per pair (adjust as needed)
    eps_edges = np.logspace(-8, 2, 10)
    tolerances = np.logspace(-4, 1, 50)  # 0.0001 … 10.0
    num_workers = cpu_count()

    # ---------- Build task list ----------
    tasks = []
    # for sA, sB in itertools.combinations_with_replacement(platonic_solids, 2):
    #     for seed in range(N):
    #         tasks.append((sA, sB, seed, base_params, faces))
    for seed in range(N):
        for eps in eps_edges:
            tasks.append(("cube", "cube", seed, base_params, faces, eps))

    print(f"Using {num_workers} workers for {len(tasks)} tasks...")

    # ---------- Parallel run ----------
    with Pool(processes=num_workers) as pool:
        results = list(
            tqdm(
                pool.imap_unordered(worker, tasks),
                total=len(tasks),
                desc="Computing distances",
            )
        )

    # # ---------- Collect all distances ----------
    # sdf_distances = []
    # hd_distances = []
    # for _, _, _, d_sdf, d_hd in results:
    #     sdf_distances.append(d_sdf)
    #     hd_distances.append(d_hd)
    #
    # # ---------- False‑positive sweep ----------
    # fp_rates = false_positive_rates(sdf_distances, hd_distances, tolerances)
    #
    # # tol = 0 (all configurations, since SDF >= 0 always)
    # tol_zero_mask = np.array(sdf_distances) >= 0.0
    # tol_zero_rate = (
    #     ((np.array(hd_distances) <= 0) & tol_zero_mask).sum()
    #     / tol_zero_mask.sum()
    #     * 100.0
    # )
    # # ---------- Plot and save ----------
    # fig = plot_fp_rate(
    #     tolerances,
    #     fp_rates,
    #     tol_zero_rate=tol_zero_rate,
    #     title="HD‑SDF conservatism (false positive rate)",
    # )
    # fig.write_image("false_positive_rate.pdf")
 
    # ---- group by eps ----
    from collections import defaultdict
    eps_distances = defaultdict(lambda: {"sdf": [], "hd": []})
    # for seed, eps, d_sdf, d_hd in results:
    for solidA, solidB, seed, dist_sdf, dist_hd, eps in results:
        eps_distances[eps]["sdf"].append(dist_sdf)
        eps_distances[eps]["hd"].append(dist_hd)

    # ---- compute FP curves for each eps ----
    fp_curves = []
    legend_labels = []
    for eps in eps_edges:
        sdf_arr = np.array(eps_distances[eps]["sdf"])
        hd_arr  = np.array(eps_distances[eps]["hd"])
        fp_rates = false_positive_rates(sdf_arr, hd_arr, tolerances)
        fp_curves.append(fp_rates)
        legend_labels.append(f"ε={eps:.1e}")

    # ---- plot all curves together ----
    fig = plot_fp_rate(
        tolerances, fp_curves, legend_labels,
        title=f"False‑positive rate vs. tolerance (cube–cube, varying ε, over {N} seeds)"
    )
    fig.write_image("eps_comparison.pdf")
    # fig.show()

    # # Save raw data for later use
    # data_out = {
    #     "tolerances": tolerances,
    #     "fp_rates": fp_rates,
    #     "tol_zero_rate": tol_zero_rate,
    #     "sdf_distances": sdf_distances,
    #     "hd_distances": hd_distances,
    # }
    # with open("fp_analysis.pkl", "wb") as f:
    #     pickle.dump(data_out, f)
    # ---- save data ----
    data_out = {
        "eps_edges": eps_edges,
        "tolerances": tolerances,
        "fp_curves": fp_curves,
        "legend_labels": legend_labels,
    }

    with open("fp_eps_sweep.pkl", "wb") as f:
        pickle.dump(data_out, f)

    print("Done. Figure saved as 'false_positive_rate.pdf', data as 'fp_analysis.pkl'.")
