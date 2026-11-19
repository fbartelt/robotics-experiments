# animate_problematic_inits.py
"""
Animate the furthest successful P1/P2 init for each curve (method only).

Assumes the multi-run pickle stores every per-run field under the
`run_{rid:03d}_` prefix (dict shown by the user, flattened), e.g.:
    run_000_time, run_000_htm, run_000_curve,
    run_000_idx, run_000_dist, run_000_err_pos, run_000_err_ori,
    run_000_meta_init_type, run_000_meta_htm0, run_000_meta_curve_seed

Every required key is checked with `require()`; a missing key raises
immediately — no fallbacks, no reconstruction, no `.npy` reload.
"""

import os
import pickle
import numpy as np
from scipy.spatial import cKDTree

from uaibot.simobjects import Frame, PointCloud
from uaibot.simobjects.curve import CurveSE3
from uaibot.utils import Utils
from fossen.create_bluerov2 import make_rov
from fossen.ocean_sim import create_sim_underwater

# =============================================================================
# Configuration
# =============================================================================
INIT_TYPES = ("P1", "P2")
CURVE_TYPES = ["ellipse", "vertical_coil", "toroidal_coil"]

results_dir = "./fossen/data/keep_results"
anim_base_path = "./fossen/media/anims"

# --- Success criteria (must match the analysis script) ----------------------
COVERAGE_THRESHOLD = 0.9 * 0.6
MIN_LAPS = 0.6
N_BINS = 100
MAX_DWELL_FRAC = 0.30
MAX_SS_POS = 30.0  # cm
MAX_SS_ORI = 20.0  # deg

# --- Animation --------------------------------------------------------------
SAVE_FOLLOW = True
SHOW_BROWSER = False
WIDTH, HEIGHT = 1920, 1080
INIT_D_THRESHOLD = np.sqrt(2.0) * np.pi  # ≈ 4.4429


# =============================================================================
# I/O helpers
# =============================================================================
def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def list_run_ids(data):
    """Detect run ids from the `run_XXX_idx` keys."""
    return sorted(
        {
            int(k[len("run_") : -len("_idx")])
            for k in data
            if k.startswith("run_") and k.endswith("_idx")
        }
    )


def require(data, key):
    """Return data[key] or raise KeyError naming the missing key."""
    if key not in data:
        raise KeyError(f"Required key missing from pickle: '{key}'")
    return data[key]


# =============================================================================
# Success check (mirrors analyze_run from the analysis script)
# =============================================================================
def analyze_run(idx, dist, err_pos, err_ori, t, N):
    diffs = np.diff(idx.astype(float))
    diffs = np.where(diffs < -N / 2, diffs + N, diffs)
    diffs = np.where(diffs > N / 2, diffs - N, diffs)
    cumulative = np.concatenate([[0.0], np.cumsum(diffs)])
    laps = float(cumulative[-1] / N)

    bins = np.clip(np.floor(idx.astype(float) / N * N_BINS).astype(int), 0, N_BINS - 1)
    coverage = len(np.unique(bins)) / N_BINS

    dt = float(t[1] - t[0]) if len(t) > 1 else 0.01
    dwell = np.zeros(N_BINS)
    np.add.at(dwell, bins, dt)
    total = float(t[-1] - t[0])
    max_dwell = float(dwell.max() / total) if total > 0 else 1.0

    n_ss = max(int(0.8 * len(t)), 1)
    ss_pos = float(np.mean(err_pos[n_ss:]))
    ss_ori = float(np.mean(err_ori[n_ss:]))

    ok_perf = (ss_pos <= MAX_SS_POS) and (ss_ori <= MAX_SS_ORI)
    success = (
        coverage >= COVERAGE_THRESHOLD
        and abs(laps) >= MIN_LAPS
        and max_dwell <= MAX_DWELL_FRAC
        and ok_perf
    )
    return {
        "success": success,
        "laps": laps,
        "coverage": coverage,
        "max_dwell": max_dwell,
        "ss_pos": ss_pos,
        "ss_ori": ss_ori,
    }


# =============================================================================
# Animation
# =============================================================================
def animate_run(time_vec, log_htm, curve, save_name):
    htm0 = log_htm[0]
    auv = make_rov(htm=htm0)
    auv_frame = Frame(htm=htm0)

    camera_start_htm = np.asarray(Utils.trn([-2.0, 0.0, 1.0])) @ htm0
    camera_start_pose = [
        camera_start_htm[0, 3],
        camera_start_htm[1, 3],
        camera_start_htm[2, 3],
        0,
        0,
        0,
        1.0,
    ]

    sim = create_sim_underwater([auv, auv_frame])

    log_position = log_htm[:, :3, 3].reshape(-1, 3)
    followed_path = PointCloud(points=log_position.T, size=0.05, color="magenta")
    sim.add(followed_path)

    curve_ub = CurveSE3(points=curve, color="cyan", size=0.05, num_frames=20)
    sim.add(curve_ub)
    sim.set_parameters(width=WIDTH, height=HEIGHT)

    for i, t_ in enumerate(time_vec):
        t = float(t_)
        htm = log_htm[i]
        auv.add_ani_frame(time=t, htm=htm)
        auv_frame.add_ani_frame(time=t, htm=htm)
        followed_path.add_ani_frame(time=t, initial_ind=0, final_ind=i)

    if SHOW_BROWSER:
        sim.run_in_browser()

    sim.save(anim_base_path, save_name)
    print(f"    saved: {save_name}")

    if SAVE_FOLLOW:
        sim.set_parameters(camera_follow=auv, camera_start_pose=camera_start_pose)
        sim.save(anim_base_path, save_name + "_follow")
        print(f"    saved: {save_name}_follow")


# =============================================================================
# Main
# =============================================================================
def main():
    os.makedirs(anim_base_path, exist_ok=True)

    for curve_type in CURVE_TYPES:
        print("\n" + "=" * 80)
        print(f"Curve: {curve_type}")
        print("=" * 80)

        method_path = os.path.join(results_dir, f"{curve_type}_results.pkl")
        if not os.path.exists(method_path):
            raise FileNotFoundError(method_path)

        data = load_pickle(method_path)

        # --- Collect every P1/P2 run ----------------------------------------
        runs = []
        for rid in list_run_ids(data):
            p = f"run_{rid:03d}_"

            init_type = require(data, p + "meta_init_type")
            if init_type not in INIT_TYPES:
                continue

            htm0 = np.asarray(require(data, p + "meta_htm0"))
            curve_4x4 = np.asarray(require(data, p + "curve"))
            seed = int(require(data, p + "meta_curve_seed"))

            N = len(curve_4x4)

            metrics = analyze_run(
                np.asarray(require(data, p + "idx")),
                np.asarray(require(data, p + "dist")),
                np.asarray(require(data, p + "err_pos")),
                np.asarray(require(data, p + "err_ori")),
                np.asarray(require(data, p + "time")),
                N,
            )

            init_d = float(np.asarray(require(data, p + "dist"))[0])

            runs.append(
                {
                    "rid": rid,
                    "init_type": init_type,
                    "seed": seed,
                    "init_d": float(init_d),
                    **metrics,
                }
            )

        if not runs:
            print("  No P1/P2 runs found.")
            continue

        runs.sort(key=lambda r: r["init_d"])

        # --- Diagnostic table -----------------------------------------------
        print(f"\n  {len(runs)} P1/P2 runs (sorted by initial distance, low → high):")
        print(
            f"    {'run':>4}  {'type':>4}  {'seed':>4}  "
            f"{'init_d':>7}  {'ok':>5}  "
            f"{'lap':>6}  {'cov':>5}  {'dwell':>5}  "
            f"{'ss_pos':>7}  {'ss_ori':>7}"
        )
        for r in runs:
            print(
                f"    {r['rid']:>4}  {r['init_type']:>4}  {r['seed']:>4}  "
                f"{r['init_d']:>7.3f}  {str(r['success']):>5}  "
                f"{r['laps']:>+6.2f}  {r['coverage']:>5.2f}  "
                f"{r['max_dwell']:>5.2f}  "
                f"{r['ss_pos']:>7.2f}  {r['ss_ori']:>7.2f}"
            )

        # --- Furthest successful --------------------------------------------
        successful = [r for r in runs if r["success"]]
        if not successful:
            print("\n  No successful P1/P2 runs — nothing to animate.")
            continue

        # target = successful[-1]
        # print(
        #     f"\n  Furthest SUCCESSFUL init: run={target['rid']}, "
        #     f"type={target['init_type']}, seed={target['seed']}, "
        #     f"init_d={target['init_d']:.3f}"
        # )
        #
        # p = f"run_{target['rid']:03d}_"
        # t_vec = np.asarray(require(data, p + "time"))
        # htm = np.asarray(require(data, p + "htm"))
        # curve_4x4 = np.asarray(require(data, p + "curve"))

        # targets = []
        # for it in INIT_TYPES:  # ("P1", "P2")
        #     group = [r for r in successful if r["init_type"] == it]
        #     if not group:
        #         print(
        #             f"\n  WARNING: no successful {it} runs for {curve_type} — skipping."
        #         )
        #         continue
        #     furthest = group[-1]  # runs is sorted ascending by init_d
        #     print(
        #         f"\n  Furthest SUCCESSFUL {it}: run={furthest['rid']}, "
        #         f"seed={furthest['seed']}, init_d={furthest['init_d']:.3f}"
        #     )
        #     targets.append(furthest)
        targets = []
        for it in INIT_TYPES:  # ("P1", "P2")
            group = [
                r
                for r in successful
                if r["init_type"] == it and r["init_d"] > INIT_D_THRESHOLD
            ]
            if not group:
                print(
                    f"\n  WARNING: no successful {it} runs with "
                    f"init_d > sqrt(2)*pi ≈ {INIT_D_THRESHOLD:.3f} "
                    f"for {curve_type} — skipping."
                )
                continue
            group.sort(key=lambda r: r["init_d"])  # ascending
            chosen = group[0]  # closest above threshold
            print(
                f"\n  Closest SUCCESSFUL {it} above sqrt(2)*pi: "
                f"run={chosen['rid']}, seed={chosen['seed']}, "
                f"init_d={chosen['init_d']:.3f}"
            )
            targets.append(chosen)

        # --- Animate each target -------------------------------------------
        for target in targets:
            p = f"run_{target['rid']:03d}_"
            t_vec = np.asarray(require(data, p + "time"))
            htm = np.asarray(require(data, p + "htm"))
            curve_4x4 = np.asarray(require(data, p + "curve"))

            save_name = (
                f"{curve_type}_{target['init_type']}"
                f"_seed{target['seed']:03d}_run{target['rid']:03d}"
            )
            print(f"\n  Animating {save_name} ({len(t_vec)} steps)…")
            animate_run(t_vec, htm, curve_4x4, save_name)

        # save_name = (
        #     f"{curve_type}_{target['init_type']}"
        #     f"_seed{target['seed']:03d}_run{target['rid']:03d}"
        # )
        # print(f"\n  Animating {save_name} ({len(t_vec)} steps)…")
        # animate_run(t_vec, htm, curve_4x4, save_name)


if __name__ == "__main__":
    main()
