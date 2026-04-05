# %%
import webbrowser
import os
import itertools
import multiprocessing
import uaibot as ub
import numpy as np
import scipy as sp
import uaibot as ub
import plotly.graph_objects as go
import plotly.express as px
import plotly.colors as pc
from polygon import Polytope
from scipy.optimize import linprog
from multiprocessing import Pool

# from uaibot_cpp_bind import expSO3, SmapSO3, SmapSE3, expSE3, ECdistance
from uaibot.simobjects.convexpolytope import ConvexPolytope
from plotly.subplots import make_subplots
from pathlib import Path
from smoothfunctions import (
    holderMeanWithGradient,
    signedDist2Convex,
    smoothMinListWithGradient,
    smoothMin2ElementsGradient,
    smoothMin2ElementsWithGradient,
    smoothMinList,
    smoothMaxList,
    smoothMaxListWithGradient,
    phi,
    ESDF_CGAL,
)


# open browser with html file (neovim workaround)
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


def pR2htm(p, R):
    H = np.eye(4)
    H[:3, :3] = R
    H[:3, -1] = p.ravel()
    return H


def dist_box2box(ubP, ubQ, r):
    wx, wz, wy = ubP.width / 2, ubP.height / 2, ubP.depth / 2
    verts_P = []
    verts_P.append(
        np.array(
            ubP.htm[:3, 0] * wx
            + ubP.htm[:3, 1] * wy
            + ubP.htm[:3, 2] * wz
            + ubP.htm[:3, -1]
        )
    )
    verts_P.append(
        np.array(
            ubP.htm[:3, 0] * wx
            + ubP.htm[:3, 1] * wy
            + ubP.htm[:3, 2] * -wz
            + ubP.htm[:3, -1]
        )
    )
    verts_P.append(
        np.array(
            ubP.htm[:3, 0] * wx
            + ubP.htm[:3, 1] * -wy
            + ubP.htm[:3, 2] * wz
            + ubP.htm[:3, -1]
        )
    )
    verts_P.append(
        np.array(
            ubP.htm[:3, 0] * wx
            + ubP.htm[:3, 1] * -wy
            + ubP.htm[:3, 2] * -wz
            + ubP.htm[:3, -1]
        )
    )
    verts_P.append(
        np.array(
            ubP.htm[:3, 0] * -wx
            + ubP.htm[:3, 1] * wy
            + ubP.htm[:3, 2] * wz
            + ubP.htm[:3, -1]
        )
    )
    verts_P.append(
        np.array(
            ubP.htm[:3, 0] * -wx
            + ubP.htm[:3, 1] * wy
            + ubP.htm[:3, 2] * -wz
            + ubP.htm[:3, -1]
        )
    )
    verts_P.append(
        np.array(
            ubP.htm[:3, 0] * -wx
            + ubP.htm[:3, 1] * -wy
            + ubP.htm[:3, 2] * wz
            + ubP.htm[:3, -1]
        )
    )
    verts_P.append(
        np.array(
            ubP.htm[:3, 0] * -wx
            + ubP.htm[:3, 1] * -wy
            + ubP.htm[:3, 2] * -wz
            + ubP.htm[:3, -1]
        )
    )
    wx, wz, wy = ubQ.width / 2, ubQ.height / 2, ubQ.depth / 2
    verts_Q = []
    verts_Q.append(
        np.array(
            ubQ.htm[:3, 0] * wx
            + ubQ.htm[:3, 1] * wy
            + ubQ.htm[:3, 2] * wz
            + ubQ.htm[:3, -1]
        )
    )
    verts_Q.append(
        np.array(
            ubQ.htm[:3, 0] * wx
            + ubQ.htm[:3, 1] * wy
            + ubQ.htm[:3, 2] * -wz
            + ubQ.htm[:3, -1]
        )
    )
    verts_Q.append(
        np.array(
            ubQ.htm[:3, 0] * wx
            + ubQ.htm[:3, 1] * -wy
            + ubQ.htm[:3, 2] * wz
            + ubQ.htm[:3, -1]
        )
    )
    verts_Q.append(
        np.array(
            ubQ.htm[:3, 0] * wx
            + ubQ.htm[:3, 1] * -wy
            + ubQ.htm[:3, 2] * -wz
            + ubQ.htm[:3, -1]
        )
    )
    verts_Q.append(
        np.array(
            ubQ.htm[:3, 0] * -wx
            + ubQ.htm[:3, 1] * wy
            + ubQ.htm[:3, 2] * wz
            + ubQ.htm[:3, -1]
        )
    )
    verts_Q.append(
        np.array(
            ubQ.htm[:3, 0] * -wx
            + ubQ.htm[:3, 1] * wy
            + ubQ.htm[:3, 2] * -wz
            + ubQ.htm[:3, -1]
        )
    )
    verts_Q.append(
        np.array(
            ubQ.htm[:3, 0] * -wx
            + ubQ.htm[:3, 1] * -wy
            + ubQ.htm[:3, 2] * wz
            + ubQ.htm[:3, -1]
        )
    )
    verts_Q.append(
        np.array(
            ubQ.htm[:3, 0] * -wx
            + ubQ.htm[:3, 1] * -wy
            + ubQ.htm[:3, 2] * -wz
            + ubQ.htm[:3, -1]
        )
    )

    minkowski_verts = [vP - vQ for vP in verts_P for vQ in verts_Q]
    normals_P = [np.array(ubP.htm[:3, i]) for i in range(3)]
    normals_P += [-np.array(ubP.htm[:3, i]) for i in range(3)]
    normals_Q = [np.array(ubQ.htm[:3, i]) for i in range(3)]
    normals_Q += [-np.array(ubQ.htm[:3, i]) for i in range(3)]

    directions = normals_P + normals_Q
    inner_mins = []
    debugs = []
    jacobian = np.zeros((len(directions), len(minkowski_verts)))
    for i, d_ in enumerate(directions):
        d = d_ / np.linalg.norm(d_)
        dotprods = [np.dot(d.ravel(), v.ravel()) for v in minkowski_verts]
        dist, grad = smoothMinListWithGradient(dotprods, r=r)
        inner_mins.append(dist)
        debugs.append(dotprods)
        jacobian[i, :] = grad
    dist, grad = smoothMaxListWithGradient(inner_mins, r=r)
    grad = grad @ jacobian
    return dist, grad, (inner_mins, debugs, verts_Q, directions)
    # return dist, grad, (inner_mins, debugs, minkowski_verts, directions)


def dist_set2set(P: Polytope, Q: Polytope, r):
    """TEST: Compute the approximate distance:
    max_{d in D} min_{vp in P, vq in Q} d^T (vp - vq)"""
    verts_Q = Q.vertices
    verts_P = P.vertices
    minkowski_verts = [vP - vQ for vP in verts_P for vQ in verts_Q]
    normals_P = [a for a in P.A]
    normals_Q = [a for a in Q.A]
    directions = normals_P + normals_Q
    inner_mins = []
    debugs = []
    jacobian = np.zeros((len(directions), len(minkowski_verts)))
    for i, d_ in enumerate(directions):
        d = d_ / np.linalg.norm(d_)
        dotprods = [np.dot(d, v) for v in minkowski_verts]
        dist, grad = smoothMinListWithGradient(dotprods, r=r)
        inner_mins.append(dist)
        debugs.append(dotprods)
        jacobian[i, :] = grad
    dist, grad = smoothMaxListWithGradient(inner_mins, r=r)
    grad = grad @ jacobian
    return dist, grad, (inner_mins, debugs, minkowski_verts, directions)


# Two Cubes
side = 0.1
# A = np.array(
#     [
#         [1.0, 0, 0],
#         [-1.0, 0, 0],
#         [0, 1.0, 0],
#         [0, -1.0, 0],
#         [0, 0, 1.0],
#         [0, 0, -1.0],
#     ]
# )
# b = np.array([side, side, side, side, side, side])
# dx_ = 0.5
# displacement = np.array([-dx_, 0.0, 0]).reshape(-1, 1)
# P = Polytope(A=A, b=b)
# Q = Polytope(A=A, b=1.5 * b + (A @ displacement).ravel())
# PmQ, _ = Polytope.minkowski_subtraction(P, Q)

# ubP = ConvexPolytope(htm=np.eye(4), A=P.A, b=P.b, color="red", opacity=0.8)
# ubQ = ConvexPolytope(htm=np.eye(4), A=Q.A, b=Q.b, color="blue", opacity=0.8)


def run_simulation(case_name, params, T=5.0, dt=1e-3, r=1e-1, return_uaibot_sim=False):
    # Extract parameters with defaults
    log_file = "experiment_log2.txt"
    side = 0.2
    widthP = params.get("widthP", side)
    heightP = params.get("heightP", side)
    depthP = params.get("depthP", side)
    widthQ = params.get("widthQ", 1.5 * side)
    heightQ = params.get("heightQ", 1.5 * side)
    depthQ = params.get("depthQ", 1.5 * side)
    init_pos_Q = np.array(params["init_pos_Q"]).reshape(-1, 1)
    movement_Q = np.array(params.get("movement_Q", [1.0, 0.0, 0.0])).reshape(-1, 1)
    rot_Q_scale = params.get("rot_Q_scale", 8.0)
    rot_P_scale = params.get("rot_P_scale", 4.0)
    trans_P_scale = params.get(
        "trans_P_scale", 0.5
    )  # ubP translates by (t/T)*trans_P_scale along x
    # Create boxes (initial positions)
    ubP = ub.Box(width=widthP, height=heightP, depth=depthP, color="red", opacity=0.8)
    ubQ = ub.Box(
        width=widthQ,
        height=heightQ,
        depth=depthQ,
        color="blue",
        opacity=0.8,
        htm=pR2htm(init_pos_Q, np.eye(3)),
    )
    imax = int(T / dt)
    hist_dist = []
    hist_euc_dist = []
    hist_gradnorm = []
    warning_occurred = False
    debug_entries = []
    if return_uaibot_sim:
        sim = ub.Simulation()
        sim.add([ubP, ubQ])

    for i in range(imax):
        t = i * dt
        frac = t / T

        # ---- ubQ motion ----
        pos_Q = init_pos_Q + movement_Q * frac
        angle_Q = rot_Q_scale * np.pi * frac
        R_Q = (
            ub.Utils.rotx(angle_Q) @ ub.Utils.roty(angle_Q) @ ub.Utils.rotz(2 * angle_Q)
        )
        R_Q = R_Q[:3, :3]
        htm_Q = pR2htm(R_Q @ pos_Q, R_Q)

        # ---- ubP motion ----
        trans_P = np.array([trans_P_scale * frac, 0.0, 0.0]).reshape(-1, 1)
        angle_P = rot_P_scale * np.pi * frac
        htm_P = ub.Utils.trn(trans_P) @ ub.Utils.rotx(angle_P)

        ubP.add_ani_frame(time=t, htm=htm_P)
        ubQ.add_ani_frame(time=t, htm=htm_Q)

        # Compute distances
        dist, grad, (inner, debugs, verts, dirs) = dist_box2box(
            ubP, ubQ, r=r
        )  # assumes function exists
        nearP, nearQ, euclidean_dist, *_ = ubP.compute_dist(ubQ)

        # Check warning condition
        if euclidean_dist == 0.0 and dist > 0.0:
            print(
                f"Warning: {case_name} at t={t:.4f} (step {i}) – dist={dist:.4e}, euc_dist={euclidean_dist}"
            )
            warning_occurred = True

            debug_entries.append(
                (i, t, inner, debugs, verts, dirs, dist, euclidean_dist)
            )
            with open(log_file, "a") as f:
                f.write(
                    f"Warning: {case_name} at t={t:.4f} (step {i}) – dist={dist:.4e}, euc_dist={euclidean_dist}\n"
                )

        hist_dist.append(dist)
        hist_euc_dist.append(euclidean_dist)
        hist_gradnorm.append(np.linalg.norm(grad))

    # ---- Save plots ----
    # Distance plot
    # fig1 = go.Figure()
    # fig1.add_trace(go.Scatter(y=hist_dist, name="Signed Distance"))
    # fig1.add_trace(go.Scatter(y=hist_euc_dist, name="Euclidean Distance"))
    # fig1.update_layout(
    #     title=f"Distances – {case_name}",
    #     xaxis_title="Time step",
    #     yaxis_title="Distance",
    # )
    # fig1.write_image(f"figs/{case_name}_distances.png")

    # # Gradient norm plot
    # fig2 = go.Figure()
    # fig2.add_trace(go.Scatter(y=hist_gradnorm, name='Gradient Norm'))
    # fig2.update_layout(title=f"Gradient Norm – {case_name}",
    #                    xaxis_title="Time step", yaxis_title="Norm")
    # fig2.write_image(f"figs/{case_name}_gradnorm.png")

    if return_uaibot_sim:
        return warning_occurred, debug_entries, sim

    return warning_occurred, debug_entries


os.makedirs("figs2", exist_ok=True)
# widthPs = [1e-3, 0.1, 0.2, 10.0]
# widthQs = [1e-3, 0.1, 0.2, 10.0]
# heightPs = [1e-3, 0.1, 0.2, 10.0]
# heightQs = [1e-3, 0.1, 0.2, 10.0]
# depthPs = [1e-3, 0.1, 0.2, 10.0]
# depthQs = [1e-3, 0.1, 0.2, 10.0]
# init_pos_Qs = [
#     [-0.15, 0.0, 0.0],
#     [-0.05, 0.02, 0.01],
#     [-0.25, 0.05, 0.0],
#     [0.0, 0.0, 0.0],
# ]
# movement_Qs = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 0.01], [-1e-5, 1e-5, -1e-5]]
# rot_Q_scales = [0.0, 2.0, 8.0]
# rot_P_scales = [0.0, 2.0, 8.0]
# trans_P_scales = [0.0, 0.5, 2.0]
#
# # Generate all combinations
# case_counter = 0
# warnings_summary = []
# log_file = "experiment_log.txt"


def run_single_case(args):
    """
    Worker function that runs one simulation case.
    Args:
        args: tuple (index, params)
    Returns:
        dict containing:
            index, case_name, params, warning_occurred, debug_entries, error
    """
    index, params = args
    case_name = f"case_{index+1}"  # 1-based naming
    # log_file = "experiment_log2.txt"

    try:
        warning_occurred, debug_entries = run_simulation(case_name, params)
        error = None
    except Exception as e:
        warning_occurred = False
        debug_entries = []
        error = str(e)

    return {
        "index": index,
        "case_name": case_name,
        "params": params,
        "warning_occurred": warning_occurred,
        "debug_entries": debug_entries,
        "error": error,
    }


def main():
    widthPs = [1e-3, 0.1, 10.0]
    widthQs = [1e-3, 0.1, 10.0]
    heightPs = [1e-3, 0.1, 10.0]
    heightQs = [1e-3, 0.1, 10.0]
    depthPs = [1e-3, 0.1, 10.0]
    depthQs = [1e-3, 0.1, 10.0]
    init_pos_Qs = [
        [-0.15, 0.0, 0.0],
        [-0.05, 0.02, 0.01],
        [-0.25, 0.05, 0.0],
        [0.0, 0.0, 0.0],
    ]
    movement_Qs = [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [1.0, 1.0, 0.01],
        [-1e-5, 1e-5, -1e-5],
    ]
    rot_Q_scales = [0.0, 2.0, 8.0]
    rot_P_scales = [0.0, 2.0, 8.0]
    trans_P_scales = [0.0, 0.5]  # , 2.0]

    # Generate all combinations
    case_counter = 0
    warnings_summary = []
    log_file = "experiment_log2.txt"

    # Generate all parameter combinations as a list
    param_combinations = list(
        itertools.product(
            widthPs,
            widthQs,
            heightPs,
            heightQs,
            depthPs,
            depthQs,
            init_pos_Qs,
            movement_Qs,
            rot_Q_scales,
            rot_P_scales,
            trans_P_scales,
        )
    )

    total_cases = len(param_combinations)
    print(f"Total cases to run: {total_cases}")
    with open(log_file, "w") as f:
        f.write(f"Total cases: {total_cases}\n\n")

    # Prepare input for workers: (index, param_tuple)
    # Each param_tuple is unpacked into the original parameter order
    # inputs = [(i, params) for i, params in enumerate(param_combinations)]
    inputs = []
    for i, (
        widthP,
        widthQ,
        heightP,
        heightQ,
        depthP,
        depthQ,
        init_pos_Q,
        movement_Q,
        rot_Q_scale,
        rot_P_scale,
        trans_P_scale,
    ) in enumerate(param_combinations):
        params = {
            "widthP": widthP,
            "heightP": heightP,
            "depthP": depthP,
            "widthQ": widthQ,
            "heightQ": heightQ,
            "depthQ": depthQ,
            "init_pos_Q": init_pos_Q,
            "movement_Q": movement_Q,
            "rot_Q_scale": rot_Q_scale,
            "rot_P_scale": rot_P_scale,
            "trans_P_scale": trans_P_scale,
        }
        inputs.append((i, params))

    # Determine number of CPUs
    ncpus = multiprocessing.cpu_count()
    print(f"Using {ncpus} CPU cores.")

    # Create a pool of workers
    with Pool(processes=ncpus) as pool:
        # Use imap_unordered to get results as they complete
        results = []
        for result in pool.imap_unordered(run_single_case, inputs):
            results.append(result)

            # Print progress in real time (order may be mixed)
            case_name = result["case_name"]
            if result["error"]:
                print(f"{case_name}: ERROR - {result['error']}")
            elif result["warning_occurred"]:
                print(f"{case_name}: WARNING detected")
            else:
                print(f"{case_name}: OK")

    # Sort results by original index to preserve order in the log file
    # results.sort(key=lambda x: x["index"])

    # Write the log file in order
    # with open(log_file, "w") as f:
    #     for res in results:
    #         case_name = res["case_name"]
    #         params = res["params"]
    #         warning_occurred = res["warning_occurred"]
    #         debug_entries = res["debug_entries"]
    #         error = res["error"]
    #
    #         f.write(f"{case_name}: {params}\n")
    #         if error:
    #             f.write(f"  ERROR in {case_name}: {error}\n")
    #         elif warning_occurred:
    #             for (
    #                 i,
    #                 t,
    #                 inner,
    #                 debugs,
    #                 verts,
    #                 dirs,
    #                 dist,
    #                 euclidean_dist,
    #             ) in debug_entries:
    #                 f.write(
    #                     f"Warning: {case_name} at t={t:.4f} (step {i}) – dist={dist:.4e}, euc_dist={euclidean_dist}\n"
    #                 )
    # else: no warnings or errors, nothing extra

    # Build warnings summary from sorted results
    warnings_summary = [res["case_name"] for res in results if res["warning_occurred"]]

    # Print final summary
    print("\n" + "=" * 60)
    print("Simulation finished.")
    if warnings_summary:
        print("Warnings occurred in the following cases:")
        for name in warnings_summary:
            print(f"  {name}")
    else:
        print("No warnings occurred in any of the tested cases.")
    print("=" * 60)


if __name__ == "__main__":
    main()
# for (
#     widthP,
#     widthQ,
#     heightP,
#     heightQ,
#     depthP,
#     depthQ,
#     init_pos_Q,
#     movement_Q,
#     rot_Q_scale,
#     rot_P_scale,
#     trans_P_scale,
# ) in itertools.product(
#     widthPs,
#     widthQs,
#     heightPs,
#     heightQs,
#     depthPs,
#     depthQs,
#     init_pos_Qs,
#     movement_Qs,
#     rot_Q_scales,
#     rot_P_scales,
#     trans_P_scales,
# ):
#     case_counter += 1
#     case_name = f"case_{case_counter}"
#     params = {
#         "widthP": widthP,
#         "heightP": heightP,
#         "depthP": depthP,
#         "widthQ": widthQ,
#         "heightQ": heightQ,
#         "depthQ": depthQ,
#         "init_pos_Q": init_pos_Q,
#         "movement_Q": movement_Q,
#         "rot_Q_scale": rot_Q_scale,
#         "rot_P_scale": rot_P_scale,
#         "trans_P_scale": trans_P_scale,
#     }
#     print(f"Running {case_name} with params: {params}")
#     with open(log_file, "a") as f:
#         f.write(f"{case_name}: {params}\n")
#     try:
#         warning_occurred, debug_entries = run_simulation(case_name, params)
#         if warning_occurred:
#             warnings_summary.append(case_name)
#
#             print(f"  -> WARNING detected in {case_name}")
#             with open(log_file, "a") as f:
#                 for (
#                     i,
#                     t,
#                     inner,
#                     debugs,
#                     verts,
#                     dirs,
#                     dist,
#                     euclidean_dist,
#                 ) in debug_entries:
#                     f.write(
#                         f"Warning: {case_name} at t={t:.4f} (step {i}) – dist={dist:.4e}, euc_dist={euclidean_dist}\n"
#                     )
#         else:
#             print("  -> OK")
#         warnings_summary.append(case_name)
#     except Exception as e:
#         print(f"Error in {case_name}: {e}")
#         with open(log_file, "a") as f:
#             f.write(f"  ERROR in {case_name}: {e}\n")
#         # warnings_summary.append((case_name, f"Error: {e}"))
#
# # ----------------------------------------------------------------------
# # Summary
# # ----------------------------------------------------------------------
# print("\n" + "=" * 60)
# print("Simulation finished.")
# if warnings_summary:
#     print("Warnings occurred in the following cases:")
#     for name in warnings_summary:
#         print(f"  {name}")
# else:
#     print("No warnings occurred in any of the tested cases.")
# print("=" * 60)
#

# def cfig(data):
#     if isinstance(data, np.ndarray):
#         data = [data]
#     fig = go.Figure()
#     for d in data:
#         fig.add_trace(go.Scatter(y=d))
#     return fig
#
#
# cfig([hist_dist, hist_euc_dist]).show()
# cfig([hist_gradnorm]).show()
#

# %%
# """Visual"""
# # sim.add([ubsum])
# filename = "minkowski"
# sim.set_parameters(width=800, height=600, pixel_ratio=0.7)
# sim.save("./", f"{filename}")
# print("Saved")
#
# # print(dt * (final_time / sim_dt) / decimation / speedup)
#
# open_in_browser(f"./{filename}.html")
