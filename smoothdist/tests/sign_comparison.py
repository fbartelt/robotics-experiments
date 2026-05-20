import numpy as np
import uaibot as ub
import uaibot_cpp_bind as ub_cpp
import plotly.graph_objects as go
import webbrowser
import time
from pathlib import Path


def pR2htm(p, R):
    H = np.eye(4)
    H[:3, :3] = R
    H[:3, -1] = p.ravel()
    return H


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


def run_simulation(case_name, params, T=5.0, dt=1e-3, r=1e-1, return_uaibot_sim=False):
    # Extract parameters with defaults
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
    hist_conserv_dist = []
    hist_euc_dist = []
    hist_gradnorm = []
    times_dsdf = []
    times_dsdf_conserv = []
    times_euc = []

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
        t0 = time.time()
        dist, *gradients = ubP.signed_distance(ubQ, gamma=r, is_conservative=True)
        t1 = time.time()
        times_dsdf.append(t1 - t0)
        t0 = time.time()
        dist_conserv, *grandients_conserv = ubP.signed_distance(
            ubQ, gamma=r, is_conservative=False
        )
        t1 = time.time()
        times_dsdf_conserv.append(t1 - t0)
        t0 = time.time()
        nearP, nearQ, euclidean_dist, *_ = ubP.compute_dist(ubQ)
        t1 = time.time()
        times_euc.append(t1 - t0)

        hist_dist.append(dist)
        hist_conserv_dist.append(dist_conserv)
        hist_euc_dist.append(euclidean_dist)
        # hist_gradnorm.append(np.linalg.norm(grad))

    # Create plot
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=np.arange(imax) * dt,
            y=hist_dist,
            mode="lines",
            name="DSDF (conservative)",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=np.arange(imax) * dt,
            y=hist_conserv_dist,
            mode="lines",
            name="DSDF (full)",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=np.arange(imax) * dt,
            y=hist_euc_dist,
            mode="lines",
            name="Euclidean Distance",
        )
    )
    fig.update_layout(
        title=f"Distance Comparison - {case_name}",
        xaxis_title="Time (s)",
        yaxis_title="Distance (m)",
    )
    fig.show()

    # Averade plus/minus std in 1e-3 seconds
    print(
        f"Average DSDF (conservative) time: {np.mean(times_dsdf):.6E} ± {np.std(times_dsdf):.6E} s"
    )
    print(
        f"Average DSDF (full) time: {np.mean(times_dsdf_conserv):.6E} ± {np.std(times_dsdf_conserv):.6E} s"
    )
    print(
        f"Average Euclidean distance time: {np.mean(times_euc):.6E} ± {np.std(times_euc):.6E} s"
    )
    if return_uaibot_sim:
        return sim

widthP = 0.5
widthQ = 0.1
heightP = 0.2
heightQ = 0.1
depthP = 0.2
depthQ = 0.1
init_pos_Q = [-0.15, 0.0, 0.0]
movement_Q = [1.0, 0.0, 0.0]
rot_Q_scale = 2.0
rot_P_scale = 4.0
trans_P_scale = 0.5

params = {
    "widthP": widthP,
    "widthQ": widthQ,
    "heightP": heightP,
    "heightQ": heightQ,
    "depthP": depthP,
    "depthQ": depthQ,
    "init_pos_Q": init_pos_Q,
    "movement_Q": movement_Q,
    "rot_Q_scale": rot_Q_scale,
    "rot_P_scale": rot_P_scale,
    "trans_P_scale": trans_P_scale,
}

ub_sim = run_simulation("Comparison Case", params, return_uaibot_sim=True)

ub_sim.save("/tmp", "comparison_case")
open_in_browser("/tmp/comparison_case.html")
# mode = 'DSDF_full'
# gamma = 1e-2
#
#
# mode = 'DSDF_full'
# gamma = 1e-2
