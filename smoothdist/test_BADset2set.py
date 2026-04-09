# %%
import webbrowser
import os
import itertools
import uaibot as ub
import numpy as np
import scipy as sp
import pandas as pd
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
        if t < 1.870 or t > 1.880:
            continue
        # else:
        #     print(t)

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
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(y=hist_dist, name="Signed Distance"))
    fig1.add_trace(go.Scatter(y=hist_euc_dist, name="Euclidean Distance"))
    fig1.update_layout(
        title=f"Distances – {case_name}",
        xaxis_title="Time step",
        yaxis_title="Distance",
    )
    fig1.show()
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

bad_sims = pd.read_csv("./experiment_log2.csv")
# Sort by 'case'
bad_sims = bad_sims.sort_values(by="case").reset_index(drop=True)
# Filter for rows where "dist" > 1e-5
bad_sims = bad_sims[bad_sims["dist"] > 1e-5].reset_index(drop=True)

# num = 113244 # at t=0.9320 (step 932) – dist=1.1620e-01, euc_dist=0.0
bad_num = 1
# 0 -- Blue is a infinite line + cannot see red
# 1 -- Both infinite lines (Clearly non-collision, but Euclidean returns 0)
# 2 -- Cannot see red at all, but dist is 3e-5 (OK)
# 3 -- Both infinite lines, but dist is clearly positive (Euclidean fails ??? but ours work)

check_dist_diff = False  # If false, then cases are where dist seems non-differentiable

if check_dist_diff:
    case = bad_sims.iloc[bad_num, 0]
    # Print all warnings for current case:
    for index, row in bad_sims[bad_sims["case"] == case].iterrows():
        print(
            f"Warning: {row['case']} at t={row['time']:.4f} (step {row['step']}) – dist={row['dist']:.4e}"
        )
else:
    case = 106019 - 1  # Ok
    case = 103419 - 1  # Ok
    case = 10295 - 1  # Ok
    case = (
        100470 - 1
    )  # Ok (Dist is negative although no-collision) due to conservative Set
    case = (
        100471 - 1
    )  # Ok (Dist is negative although no-collision) due to conservative Set
    if case in bad_sims["case"].values:
        print(f"Case {case} is in the bad sims list.")
        for index, row in bad_sims[bad_sims["case"] == case].iterrows():
            print(
                f"Warning: {row['case']} at t={row['time']:.4f} (step {row['step']}) – dist={row['dist']:.4e}"
            )
param_combinations = param_combinations[case]
# Prepare input for workers: (index, param_tuple)
# Each param_tuple is unpacked into the original parameter order
# inputs = [(i, params) for i, params in enumerate(param_combinations)]
inputs = []
(
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
) = param_combinations
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
print(params)

# Determine number of CPUs
warning_occurred, debug_entries, sim = run_simulation(
    "test_case", params, return_uaibot_sim=True, dt=1e-5
)

filename = "BADset2set_test"
sim.save("./", f"{filename}")
open_in_browser(f"./{filename}.html")
