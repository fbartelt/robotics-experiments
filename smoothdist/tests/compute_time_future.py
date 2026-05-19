# %%
import sys
import os
import itertools
import numpy as np
import uaibot as ub
import plotly.graph_objects as go
import webbrowser
from time import time, perf_counter
sys.path.insert(0, "/home/fbartelt/Documents/Projetos/robotics-experiments/smoothdist/")
from euclidean_sdf import esdf, compute_vertices_and_faces
from scipy.spatial import ConvexHull
from pathlib import Path
from plotly.subplots import make_subplots


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

    return ub.ConvexPolytope(A=A, b=b, *args, **kwargs)


def run_single(seed, params):
    """Create two random boxes and compute the time spent calculating
    the distance.

    Parameters
    ----------
    seed : int
        The random seed to use for reproducibility.
    params : dict
        A dictionary containing the parameters for the distance calculation, including:

    """
    rng = np.random.default_rng(seed)
    translation_max = params.get("translation_max", [4.0, 4.0, 4.0])
    translation_min = params.get("translation_min", [1.0, 1.0, 1.0])
    rotation_max = params.get("rotation_max", [2 * np.pi, 2 * np.pi, 2 * np.pi])
    rotation_min = params.get("rotation_min", [0.0, 0.0, 0.0])
    max_radius = params.get("max_radius", 1.0)
    n_faces1 = params.get("n_faces1", 6)  # default to cube
    n_faces2 = params.get("n_faces2", 6)  # default to cube
    mode = params.get("mode", "HDSDF")  # Default to 'HSDF' if not provided
    gamma = params.get("gamma", 0.1)
    verbose = params.get("verbose", False)
    skip_gradients = params.get("skip_gradients", True)

    htm1 = random_htm(translation_max, translation_min, rotation_max, rotation_min, rng)
    obj1 = create_platonic_solid(n_faces=n_faces1, radius=max_radius, htm=htm1)
    htm2 = random_htm(translation_max, translation_min, rotation_max, rotation_min, rng)
    obj2 = create_platonic_solid(n_faces=n_faces2, radius=max_radius, htm=htm2)

    match mode.upper().replace("-", ""):
        case x if "HDSDF" in x:
            is_conservative = "C" in params["mode"]
            start_time = time()
            dist, *gradients = obj1.signed_distance(
                obj2,
                gamma,
                is_conservative=is_conservative,
                skip_gradient=skip_gradients,
            )
            end_time = time()
        case "SDF":
            vertices1, faces1 = compute_vertices_and_faces(obj1.A, obj1.b)
            vertices2, faces2 = compute_vertices_and_faces(obj2.A, obj2.b)
            # start_time = time()
            # Compute time in c++ to avoid including FCL conversion
            start_time = 0.0
            dist, end_time = esdf(obj1, obj2, vertices1, vertices2, faces1, faces2)
            end_time /= 1e6  # converts microseconds to seconds
            # end_time = time()
        case "GDF":
            start_time = time()
            dist = obj1.compute_dist(obj2, h=0.1, eps=0.01)
            end_time = time()
        case _:
            raise ValueError(f"Unknown mode: {mode}")

    elapsed_time = end_time - start_time
    if verbose:
        print(
            f"Seed: {seed}, Mode: {mode}, Time: {elapsed_time:.6e} seconds, Distance: {dist:.6f}"
        )
    return (elapsed_time, dist)

# %%
if __name__ == "__main__":
    # ------------------------------
    # Configuration
    # ------------------------------
    platonic_solids = ["tetra", "cube", "octa", "dodeca", "icosa"]
    # platonic_solids = ["octa"]
    vertices = {"tetra": 4, "cube": 8, "octa": 6, "dodeca": 20, "icosa": 12}
    faces = {"tetra": 4, "cube": 6, "octa": 8, "dodeca": 12, "icosa": 20}

    N = 100  # configurations per pair
    K = 50  # timing runs per config/method
    N = 10
    K = 2
    warmup = 10  # warmup runs (discarded)

    methods = ["HDSDF", "CHDSDF", "SDF", "GDF"]  # order as needed
    # Base parameters (everything except what changes per pair/method)
    base_params = {
        "translation_max": [2.0, 2.0, 2.0],
        "translation_min": [-2.0, -2.0, -2.0],
        "rotation_max": [2 * np.pi, 2 * np.pi, 2 * np.pi],
        "rotation_min": [0.0, 0.0, 0.0],
        "max_radius": 1.0,
        "gamma": 0.1,
        "skip_gradients": True,
        "verbose": False,
    }
    # store results: for each pair, list of per-config ratios
    results_HD_over_SDF = {}
    results_CHD_over_SDF = {}
    results_HD_over_GDF = {}

    # ------------------------------
    # Main loops
    # ------------------------------
    for solidA, solidB in itertools.combinations_with_replacement(platonic_solids, 2):
        # complexity measure
        v2 = vertices[solidA] ** 2 * vertices[solidB] ** 2

        # Set the pair-specific face counts
        base_params["n_faces1"] = faces[solidA]
        base_params["n_faces2"] = faces[solidB]

        ratios_hd_sdf = []
        ratios_chd_sdf = []
        ratios_hd_gdf = []

        for seed in range(N):
            # measure mean time per method for this config
            times = {}
            for m in methods:
                print(f"Running {m} for pair ({solidA}, {solidB}), seed {seed}...")
                # Build a fresh params dict for this method
                params = base_params.copy()
                params["mode"] = m
                # warmup
                for _ in range(warmup):
                    elapsed_time, _ = run_single(seed, params)
                # timed runs
                total_time = 0.0
                for _ in range(K):
                    elapsed_time, _ = run_single(seed, params)
                    total_time += elapsed_time
                times[m] = total_time / K

            # compute ratios for this config
            ratios_hd_sdf.append(times["HDSDF"] / times["SDF"])
            ratios_chd_sdf.append(times["CHDSDF"] / times["SDF"])
            ratios_hd_gdf.append(times["HDSDF"] / times["GDF"])

        # aggregate over configurations: geometric mean and 95% CI via bootstrap
        def geom_mean_and_ci(data, n_bootstrap=10000, alpha=0.05):
            data = np.array(data)
            gm = np.exp(np.mean(np.log(data)))
            # bootstrap CI
            boot_means = []
            rng = np.random.default_rng(42)
            for _ in range(n_bootstrap):
                sample = rng.choice(data, size=len(data), replace=True)
                boot_means.append(np.exp(np.mean(np.log(sample))))
            boot_means = np.array(boot_means)
            ci_lo = np.percentile(boot_means, 100 * alpha / 2)
            ci_hi = np.percentile(boot_means, 100 * (1 - alpha / 2))
            return gm, ci_lo, ci_hi

        gm_hd_sdf, lo_hd_sdf, hi_hd_sdf = geom_mean_and_ci(ratios_hd_sdf)
        gm_chd_sdf, lo_chd_sdf, hi_chd_sdf = geom_mean_and_ci(ratios_chd_sdf)
        gm_hd_gdf, lo_hd_gdf, hi_hd_gdf = geom_mean_and_ci(ratios_hd_gdf)

        # store with complexity as key
        results_HD_over_SDF[v2] = (gm_hd_sdf, lo_hd_sdf, hi_hd_sdf)
        results_CHD_over_SDF[v2] = (gm_chd_sdf, lo_chd_sdf, hi_chd_sdf)
        results_HD_over_GDF[v2] = (gm_hd_gdf, lo_hd_gdf, hi_hd_gdf)

    # ---------------------------------------------------------------------------
    # 1.  Convert the result dictionaries into sorted arrays for plotting
    # ---------------------------------------------------------------------------
    # After the loop you have:
    #   results_HD_over_SDF[v2]  = (gm, lo, hi)
    #   results_CHD_over_SDF[v2] = (gm, lo, hi)
    #   results_HD_over_GDF[v2]  = (gm, lo, hi)
    #
    # We assume they share the same keys (all 15 v2 values).

    x_vals = sorted(results_HD_over_SDF.keys())  # complexity values

    # Panel (a) – Slowdown relative to SDF
    hd_sdf_gm = np.array([results_HD_over_SDF[x][0] for x in x_vals])
    hd_sdf_lo = np.array([results_HD_over_SDF[x][1] for x in x_vals])
    hd_sdf_hi = np.array([results_HD_over_SDF[x][2] for x in x_vals])

    chd_sdf_gm = np.array([results_CHD_over_SDF[x][0] for x in x_vals])
    chd_sdf_lo = np.array([results_CHD_over_SDF[x][1] for x in x_vals])
    chd_sdf_hi = np.array([results_CHD_over_SDF[x][2] for x in x_vals])

    # Panel (b) – Slowdown of HD‑SDF relative to GDF
    hd_gdf_gm = np.array([results_HD_over_GDF[x][0] for x in x_vals])
    hd_gdf_lo = np.array([results_HD_over_GDF[x][1] for x in x_vals])
    hd_gdf_hi = np.array([results_HD_over_GDF[x][2] for x in x_vals])

    print("Complexity values (|V_A|^2 * |V_B|^2):", x_vals)
    print("HD‑SDF / SDF geometric means:", hd_sdf_gm)
    print("CHD‑SDF / SDF geometric means:", chd_sdf_gm)
    print("HD‑SDF / GDF geometric means:", hd_gdf_gm)

    # ---------------------------------------------------------------------------
    # 2.  Build a two‑panel figure (vertical subplots, shared x‑axis)
    # ---------------------------------------------------------------------------
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        subplot_titles=(
            "Slowdown relative to SDF",
            "Slowdown of HD‑SDF relative to GDF",
        ),
    )

    # ----- Convenience: add a trace with its own confidence band -----
    def add_trace_with_band(
        fig, x, y, y_lo, y_hi, name, color, row, col, show_legend=True
    ):
        """Add a main line and a filled error band to a subplot."""
        # Main line
        fig.add_trace(
            go.Scatter(
                name=name,
                x=x,
                y=y,
                mode="lines",
                line=dict(color=color),
                showlegend=show_legend,
            ),
            row=row,
            col=col,
        )
        # Upper bound (invisible line, just used for filling)
        fig.add_trace(
            go.Scatter(
                name=f"{name} upper",
                x=x,
                y=y_hi,
                mode="lines",
                marker=dict(color="rgba(68,68,68,0)"),
                line=dict(width=0),
                showlegend=False,
            ),
            row=row,
            col=col,
        )
        # Lower bound – fills to the upper bound
        fig.add_trace(
            go.Scatter(
                name=f"{name} lower",
                x=x,
                y=y_lo,
                mode="lines",
                marker=dict(color="rgba(68,68,68,0)"),
                line=dict(width=0),
                fillcolor=color.replace("rgb", "rgba").replace(")", ",0.2)"),
                fill="tonexty",
                showlegend=False,
            ),
            row=row,
            col=col,
        )

    # ----- Panel (a): HD‑SDF / SDF  and  CHD‑SDF / SDF -----
    add_trace_with_band(
        fig,
        x_vals,
        hd_sdf_gm,
        hd_sdf_lo,
        hd_sdf_hi,
        name="HD‑SDF / SDF",
        color="rgb(31, 119, 180)",  # blue
        row=1,
        col=1,
    )

    add_trace_with_band(
        fig,
        x_vals,
        chd_sdf_gm,
        chd_sdf_lo,
        chd_sdf_hi,
        name="CHD‑SDF / SDF",
        color="rgb(255, 127, 14)",  # orange
        row=1,
        col=1,
    )

    # Parity line (y = 1)
    fig.add_trace(
        go.Scatter(
            x=[min(x_vals), max(x_vals)],
            y=[1, 1],
            mode="lines",
            line=dict(dash="dash", color="black", width=0.8),
            showlegend=False,
        ),
        row=1,
        col=1,
    )

    # ----- Panel (b): HD‑SDF / GDF -----
    add_trace_with_band(
        fig,
        x_vals,
        hd_gdf_gm,
        hd_gdf_lo,
        hd_gdf_hi,
        name="HD‑SDF / GDF",
        color="rgb(44, 160, 44)",  # green
        row=2,
        col=1,
    )

    # Parity line (y = 1)
    fig.add_trace(
        go.Scatter(
            x=[min(x_vals), max(x_vals)],
            y=[1, 1],
            mode="lines",
            line=dict(dash="dash", color="black", width=0.8),
            showlegend=False,
        ),
        row=2,
        col=1,
    )

    # ---------------------------------------------------------------------------
    # 3.  Layout
    # ---------------------------------------------------------------------------
    fig.update_xaxes(
        type="log",
        row=1,
        col=1,
    )

    fig.update_xaxes(
        title_text="Complexity  |V<sub>A</sub>|² &#215; |V<sub>B</sub>|²",
        type="log",
        row=2,
        col=1,
    )

    fig.update_yaxes(
        title_text="Time ratio (method / SDF)",
        type="log",  # <-- remove this line if ratios stay small
        row=1,
        col=1,
    )
    fig.update_yaxes(
        title_text="Time ratio (HD‑SDF / GDF)",
        type="log",  # <-- remove if not needed
        row=2,
        col=1,
    )

    fig.update_layout(
        title=dict(text="Runtime comparison for Platonic‑solid pairs"),
        hovermode="x unified",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
        ),
        template="plotly_white",
    )

    fig.show()
    fig.write_image("runtime_comparison.svg", width=1200, height=800)
