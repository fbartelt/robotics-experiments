# animate_selected_runs.py
"""
Automated animation generation for selected runs (baseline vs method).

For each curve type:
  1. Load baseline and method (default) multi-run pickles.
  2. Keep only nominal-init runs.
  3. Match baseline ↔ method by curve_seed = run_id // N_INITS.
  4. Rank by relative steady-state improvement (baseline − method)/baseline,
     require a minimum absolute gap so near-ties get dropped.
  5. Animate the top-K seeds, both conditions (baseline & method),
     with the already-working animation logic (non-follow + follow).
"""

import os
import pickle
import numpy as np
import uaibot as ub

from uaibot.simobjects import Frame, PointCloud, Ball
from uaibot.simobjects.curve import CurveSE3
from uaibot.utils import Utils
from fossen.create_bluerov2 import make_rov
from fossen.ocean_sim import create_sim_underwater

# =============================================================================
# Configuration
# =============================================================================
N_INITS = 5
INIT_TYPE = "nominal"
CURVE_TYPES = ["ellipse", "vertical_coil", "toroidal_coil"]

curve_base_path = "./fossen/data"
results_dir = "./fossen/data/keep_results"
anim_base_path = "./fossen/media/anims"

# --- Selection thresholds ---------------------------------------------------
# A run qualifies as "baseline much worse than method" if EITHER:
#   * relative improvement on ss_dist >= 20%   AND absolute gap >= 0.1
#   * relative improvement on ss_pos  >= 20%   AND absolute gap >= 2.0 cm
MIN_REL_IMPROVEMENT = 0.20
MIN_ABS_GAP_DIST = 0.10  # units of `dist` (cm on your setup)
MIN_ABS_GAP_POS = 2.0  # cm
TOP_K = 3  # animate this many seeds per curve type

# --- Animation options ------------------------------------------------------
SAVE_FOLLOW = True  # also save the camera-follow version
SHOW_BROWSER = False  # set True to open the browser after each anim
WIDTH, HEIGHT = 1920, 1080

# --- Data source fallbacks --------------------------------------------------
# If `run_XXX_htm` is missing from the multi-run pickle, try per-run pickles
# at: {results_dir}/results_{curve}_seed{seed:03d}_{condition}.pkl
PER_RUN_TEMPLATE = "results_{curve}_seed{seed:03d}_{condition}.pkl"


# =============================================================================
# I/O helpers
# =============================================================================
def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def list_run_ids(data):
    return sorted(
        {
            int(k[len("run_") : -len("_idx")])
            for k in data
            if k.startswith("run_") and k.endswith("_idx")
        }
    )


def get_nominal_runs(data):
    """Map curve_seed -> first nominal run_id found for that seed."""
    runs = {}
    for rid in list_run_ids(data):
        init_type = data.get(f"run_{rid:03d}_meta_init_type", "nominal")
        if init_type != INIT_TYPE:
            continue
        seed = rid // N_INITS
        if seed not in runs:
            runs[seed] = rid
    return runs


def compute_ss_metrics(data, rid):
    """Steady-state metrics over the last 20% of the run."""
    p = f"run_{rid:03d}_"
    t = np.asarray(data[p + "time"])
    dist = np.asarray(data[p + "dist"])
    err_po = np.asarray(data[p + "err_pos"])
    err_or = np.asarray(data[p + "err_ori"])

    n_ss = max(int(0.8 * len(t)), 1)
    return {
        "ss_dist": float(np.mean(dist[n_ss:])),
        "ss_pos": float(np.mean(err_po[n_ss:])),
        "ss_ori": float(np.mean(err_or[n_ss:])),
        "final_dist": float(dist[-1]),
        "n_steps": len(t),
        "duration": float(t[-1] - t[0]),
    }


# =============================================================================
# Selection
# =============================================================================
def select_runs(baseline_data, method_data, top_k=TOP_K, verbose=True):
    """Return (top_selected, all_candidates)."""
    b_runs = get_nominal_runs(baseline_data)
    m_runs = get_nominal_runs(method_data)
    common = sorted(set(b_runs) & set(m_runs))

    if verbose:
        print(
            f"  Nominal seeds: baseline={len(b_runs)}, method={len(m_runs)}, "
            f"common={len(common)}"
        )

    candidates = []
    for seed in common:
        b_id, m_id = b_runs[seed], m_runs[seed]
        bm = compute_ss_metrics(baseline_data, b_id)
        mm = compute_ss_metrics(method_data, m_id)

        abs_gap_d = bm["ss_dist"] - mm["ss_dist"]
        abs_gap_p = bm["ss_pos"] - mm["ss_pos"]
        rel_gap_d = abs_gap_d / (bm["ss_dist"] + 1e-9)
        rel_gap_p = abs_gap_p / (bm["ss_pos"] + 1e-9)

        qualifies = (
            rel_gap_d >= MIN_REL_IMPROVEMENT and abs_gap_d >= MIN_ABS_GAP_DIST
        ) or (rel_gap_p >= MIN_REL_IMPROVEMENT and abs_gap_p >= MIN_ABS_GAP_POS)

        candidates.append(
            {
                "seed": seed,
                "baseline_id": b_id,
                "method_id": m_id,
                "baseline": bm,
                "method": mm,
                "abs_gap_dist": abs_gap_d,
                "abs_gap_pos": abs_gap_p,
                "rel_gap_dist": rel_gap_d,
                "rel_gap_pos": rel_gap_p,
                "score": 0.5 * rel_gap_d + 0.5 * rel_gap_p,
                "qualifies": qualifies,
            }
        )

    qualified = [c for c in candidates if c["qualifies"]]
    qualified.sort(key=lambda c: c["score"], reverse=True)

    if verbose:
        print(
            f"  Qualifying seeds (baseline much worse): {len(qualified)}/{len(candidates)}"
        )

    return qualified[:top_k], candidates


# =============================================================================
# Animation (re-uses the logic from the original script, parameterised)
# =============================================================================
def animate_run(time_vec, log_htm, curve, save_name, traj=None):
    """Port of the original animation script's main loop, parameterised."""
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

    if traj is not None:
        traj0 = traj[0]
        traj_ball = Ball(htm=traj0, radius=0.05, opacity=0.6, color='white')
        traj_frame = Frame(htm=traj0)
        sim.add([traj_ball, traj_frame])

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
        if traj is not None:
            i_traj = min(i + 1, len(time_vec) - 1)
            traj_ball.add_ani_frame(time=t, htm=traj[i_traj])
            traj_frame.add_ani_frame(time=t, htm=traj[i_traj])

    if SHOW_BROWSER:
        sim.run_in_browser()

    sim.save(anim_base_path, save_name)
    print(f"    saved: {save_name}")

    if SAVE_FOLLOW:
        sim.set_parameters(camera_follow=auv, camera_start_pose=camera_start_pose)
        sim.save(anim_base_path, save_name + "_follow")
        print(f"    saved: {save_name}_follow")


def extract_run_htm(data, rid, curve_type, seed, condition):
    """
    Return (time_vec, log_htm, curve) for a given run.

    time/htm resolution order:
      1. Multi-run pickle keys  run_XXX_time / run_XXX_htm
      2. Per-run pickle  {results_dir}/results_{curve}_seedXXX_{cond}.pkl

    curve resolution order:
      1. Per-run key  run_XXX_curve  (preferred — handles randomized curves)
      2. Top-level  data["curve"]    (as you requested)
      3. None (caller warns / falls back)
    """
    p = f"run_{rid:03d}_"

    # --- time + htm ---
    if p + "htm" in data and p + "time" in data:
        t_vec = np.asarray(data[p + "time"])
        htm = np.asarray(data[p + "htm"])
    else:
        per_run = os.path.join(
            results_dir,
            PER_RUN_TEMPLATE.format(curve=curve_type, seed=seed, condition=condition),
        )
        if not os.path.exists(per_run):
            return None, None, None
        d = load_pickle(per_run)
        t_vec = np.asarray(d["time"])
        htm = np.asarray(d["htm"])

    # --- curve from pkl (NOT from .npy) ---
    if p + "curve" in data:
        curve = np.asarray(data[p + "curve"])
    elif "curve" in data:
        curve = np.asarray(data["curve"])
    else:
        curve = None  # caller will warn / fall back

    # --- extract trajectory reference ---
    if condition == "baseline":
        traj = np.asarray(data[p + "target_htm"])
    else:
        traj = None

    return t_vec, htm, curve, traj


# =============================================================================
# Main driver
# =============================================================================
def main():
    os.makedirs(anim_base_path, exist_ok=True)

    for curve_type in CURVE_TYPES:
        print("\n" + "=" * 68)
        print(f"Curve: {curve_type}")
        print("=" * 68)

        baseline_path = os.path.join(results_dir, f"{curve_type}_baseline_results.pkl")
        method_path = os.path.join(results_dir, f"{curve_type}_results.pkl")
        curve_path = os.path.join(curve_base_path, f"{curve_type}.npy")

        missing = [
            p for p in (baseline_path, method_path, curve_path) if not os.path.exists(p)
        ]
        if missing:
            print("  Skipping — missing files:")
            for p in missing:
                print(f"    {p}")
            continue

        curve = np.load(curve_path, allow_pickle=True)
        baseline_data = load_pickle(baseline_path)
        method_data = load_pickle(method_path)

        selected, all_candidates = select_runs(baseline_data, method_data)
        if not selected:
            print("  No qualifying seeds — nothing to animate.")
            continue

        # Pretty-print the selection
        print(f"\n  Top {len(selected)} selected seed(s):")
        for i, c in enumerate(selected):
            print(
                f"    [{i}] seed={c['seed']:3d}  "
                f"ss_dist  {c['baseline']['ss_dist']:8.3f} -> {c['method']['ss_dist']:8.3f}  "
                f"(rel {c['rel_gap_dist']:+6.1%})   "
                f"ss_pos  {c['baseline']['ss_pos']:8.2f} -> {c['method']['ss_pos']:8.2f}  "
                f"(rel {c['rel_gap_pos']:+6.1%})"
            )

        # Optional: print every qualifying candidate for sanity-checking
        print("\n  Full qualifying summary:")
        for c in all_candidates:
            if not c["qualifies"]:
                continue
            print(
                f"    seed={c['seed']:3d}  "
                f"score={c['score']:+.3f}  "
                f"d_gap={c['abs_gap_dist']:+.3f}  p_gap={c['abs_gap_pos']:+.2f}cm"
            )

        # Generate animations
        for c in selected:
            seed = c["seed"]

            for condition, data, rid in (
                ("baseline", baseline_data, c["baseline_id"]),
                ("method", method_data, c["method_id"]),
            ):
                save_name = f"{curve_type}_seed{seed:03d}_{condition}"
                t_vec, htm, curve_run, traj = extract_run_htm(
                    data, rid, curve_type, seed, condition
                )

                if t_vec is None:
                    print(
                        f"\n  WARNING: could not find time/htm for "
                        f"{curve_type} seed={seed} [{condition}] "
                        f"(run_id={rid})."
                    )
                    print(
                        f"           Add `run_{rid:03d}_htm` and `run_{rid:03d}_time` "
                        f"to the multi-run pickle, or place "
                        f"{PER_RUN_TEMPLATE.format(curve=curve_type, seed=seed, condition=condition)} "
                        f"in {results_dir}."
                    )
                    continue
                if curve_run is None:
                    print(
                        f"\n  WARNING: no `curve` stored in pkl for "
                        f"{curve_type} seed={seed} [{condition}]; "
                        f"falling back to npy — the drawn curve may NOT match "
                        f"the randomized one that was tracked."
                    )
                    curve_run = curve  # npy-loaded fallback

                if condition == "baseline":
                    print(f"\n  Animating {save_name} (run_id={rid}, {len(t_vec)} steps)…")
                    animate_run(t_vec, htm, curve_run, save_name, traj)


if __name__ == "__main__":
    main()
