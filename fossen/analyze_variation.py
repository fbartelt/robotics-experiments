import pickle
import numpy as np
import os
from scipy.spatial.transform import Rotation as R
from scipy.spatial import cKDTree


def rand_se3_curve(curve, curve_deriv, seed=None):
    rng = np.random.default_rng(seed)
    N = len(curve)
    yaw_global = rng.uniform(0, 2 * np.pi)
    R_global = R.from_euler("z", yaw_global).as_matrix()
    scale = rng.uniform(0.8, 1.5)
    roll_offset = rng.uniform(-np.pi / 12, np.pi / 12)
    pitch_offset = rng.uniform(-np.pi / 12, np.pi / 12)
    yaw_offset = rng.uniform(-np.pi, np.pi)
    R_offset = R.from_euler("xyz", [roll_offset, pitch_offset, yaw_offset]).as_matrix()

    curve_rand = np.zeros_like(curve)
    curve_deriv_rand = np.zeros_like(curve_deriv)

    for i in range(N):
        H = curve[i].copy()
        H[:3, :3] = R_global @ H[:3, :3] @ R_offset
        H[:3, 3] = scale * (R_global @ H[:3, 3])
        curve_rand[i] = H

        dH = curve_deriv[i].copy()
        dH[:3, :3] = R_global @ dH[:3, :3] @ R_offset
        dH[:3, 3] = scale * (R_global @ dH[:3, 3])
        curve_deriv_rand[i] = dH

    return curve_rand, curve_deriv_rand


def recompute_from_htm(htm_array, curve, K_dup=200):
    """
    Recompute (idx, err_pos, err_ori) for each true pose in htm_array,
    against the true (unscaled) curve. Uses a cKDTree with seam duplication
    for fast and correct nearest-point lookup on a closed curve.
    """
    N = len(curve)
    curve_pos = curve[:, :3, 3]  # (N, 3)
    curve_rot = curve[:, :3, :3]  # (N, 3, 3)

    curve_pos_ext = np.vstack([curve_pos[-K_dup:], curve_pos, curve_pos[:K_dup]])
    tree = cKDTree(curve_pos_ext)

    positions = htm_array[:, :3, 3]  # (T, 3)
    rotations = htm_array[:, :3, :3]  # (T, 3, 3)

    _, idx_ext = tree.query(positions)
    idx = ((idx_ext - K_dup) % N).astype(int)

    delta = curve_pos[idx] - positions
    err_pos = np.linalg.norm(delta, axis=1) * 100.0  # cm

    R_near = curve_rot[idx]  # (T, 3, 3)
    tr = np.einsum("tij,tij->t", R_near, rotations)
    err_ori = np.degrees(np.arccos(np.clip((tr - 1.0) / 2.0, -1.0, 1.0)))

    return idx, err_pos, err_ori


def analyze_pickle_recomputed_v2(
    pkl_path, curve_base_npy, dcurve_base_npy, **analyzer_kwargs
):
    """
    Regenerates the actual per-run randomized curve using the same seed that
    the simulation used, then recomputes idx / err_pos / err_ori against it.
    The saved 'dist' (SE(3), scaled units) is preserved and trusted.
    """
    print("\n" + "=" * 68)
    print(f"Analyzing (RECOMPUTED v2): {pkl_path}")
    print("=" * 68)

    curve_base = np.load(curve_base_npy, allow_pickle=True)
    dcurve_base = np.load(dcurve_base_npy, allow_pickle=True)
    N = len(curve_base)
    print(f"N = {N}")

    with open(pkl_path, "rb") as f:
        data = pickle.load(f)

    run_ids = sorted(
        {
            k[len("run_") : -len("_htm")]
            for k in data
            if k.startswith("run_") and k.endswith("_htm")
        }
    )
    print(
        f"Recomputing metrics for {len(run_ids)} runs against their own randomized curves..."
    )

    results = []
    for run_id in run_ids:
        seed = int(run_id)
        curve_i, _ = rand_se3_curve(curve_base, dcurve_base, seed=seed)

        htm = data[f"run_{run_id}_htm"]
        time_arr = data.get(f"run_{run_id}_time", np.arange(len(htm)) * 0.01)
        dist_arr = data.get(f"run_{run_id}_dist", np.zeros(len(htm)))

        idx, err_pos, err_ori = recompute_from_htm(htm, curve_i)
        results.append(
            analyze_run(idx, dist_arr, err_pos, err_ori, time_arr, N, **analyzer_kwargs)
        )

    successes = [r for r in results if r["success"]]
    failures = [(i, r) for i, r in enumerate(results) if not r["success"]]
    print(
        f"\nSuccess rate: {len(successes)}/{len(run_ids)} = "
        f"{100.0 * len(successes) / len(run_ids):.1f}%"
    )

    if successes:
        print("\nMetrics (successful runs only):")
        for metric in [
            "ss_dist",
            "ss_pos",
            "ss_ori",
            "max_dist_ss",
            "first_lap_time",
            "laps",
            "coverage",
            "max_dwell_frac",
        ]:
            vals = np.array([r[metric] for r in successes])
            vals = vals[~np.isnan(vals)]
            if len(vals):
                print(
                    f"  {metric:16s}: mean={vals.mean():9.4f}  std={vals.std():9.4f}  "
                    f"median={np.median(vals):9.4f}  min={vals.min():9.4f}  max={vals.max():9.4f}"
                )

    if failures:
        print(f"\nFailed runs ({len(failures)}):")
        for i, r in failures:
            reasons = []
            if r["coverage"] < analyzer_kwargs.get("coverage_threshold", 0.90):
                reasons.append(f"coverage={r['coverage']:.2f}")
            if r["laps"] < analyzer_kwargs.get("min_laps", 1.0):
                reasons.append(f"laps={r['laps']:.2f}")
            if r["is_stuck"]:
                reasons.append(f"stuck(dwell={r['max_dwell_frac']:.2f})")
            if not r.get("performance_ok", True):
                reasons.append(
                    f"poor_tracking(pos={r['ss_pos']:.1f}cm, ori={r['ss_ori']:.1f}°)"
                )
            print(
                f"  Run {int(run_ids[i]):3d}: "
                + ", ".join(reasons)
                + f"  | ss_dist={r['ss_dist']:.3f}  ss_pos={r['ss_pos']:.1f}cm"
            )

    return results


def unwrap_closed_index(idx_seq, N):
    """Unwrap the circular index sequence to track signed continuous advancement."""
    diffs = np.diff(idx_seq.astype(float))
    diffs = np.where(diffs < -N / 2, diffs + N, diffs)
    diffs = np.where(diffs > N / 2, diffs - N, diffs)
    return diffs


def analyze_run(
    idx,
    dist,
    err_pos,
    err_ori,
    t,
    N,
    coverage_threshold=0.90,
    min_laps=1.0,
    n_bins=100,
    max_dwell_frac=0.30,
    max_ss_pos=30.0,  # NEW: Maximum steady-state position error in cm
    max_ss_ori=20.0,  # NEW: Maximum steady-state orientation error in degrees
):
    """
    Analyze a single run.

    Success criteria:
      1) coverage >= coverage_threshold: the trajectory visited >=90% of the
         curve (using coarse bins, so skipped samples don't count against it)
      2) laps >= min_laps: the net signed progress through the curve is at
         least one full loop (so we know it went all the way around, not just
         oscillated in one region)
      3) max_dwell_frac <= 0.30: no single bin of the curve consumed more
         than 30% of the total sim time (catches "getting stuck" anywhere,
         not just at the end)
    """
    # --- 1. Signed laps with wrap correction ---
    diffs = unwrap_closed_index(idx, N)
    cumulative = np.concatenate([[0.0], np.cumsum(diffs)])
    laps = float(cumulative[-1] / N)

    # --- 2. Coverage via coarse bins ---
    bins = np.clip(np.floor(idx.astype(float) / N * n_bins).astype(int), 0, n_bins - 1)
    coverage = len(np.unique(bins)) / n_bins

    # --- 3. Dwell-time based stuck detection ---
    dt = float(t[1] - t[0]) if len(t) > 1 else 0.01
    dwell = np.zeros(n_bins)
    np.add.at(dwell, bins, dt)
    total_time = float(t[-1] - t[0])
    max_dwell_frac_actual = float(dwell.max() / total_time) if total_time > 0 else 1.0
    is_stuck = max_dwell_frac_actual > max_dwell_frac

    # --- 4. Steady-state metrics (last 20% of run) ---
    n = len(t)
    n_ss = int(0.8 * n)
    ss_dist = float(dist[n_ss:].mean())
    ss_pos = float(err_pos[n_ss:].mean())
    ss_ori = float(err_ori[n_ss:].mean())
    max_dist = float(dist[n_ss:].max())

    # --- 5. Time to first full lap ---
    lap_progress = cumulative / N
    if lap_progress[-1] >= 1.0:
        first_lap_time = float(t[np.argmax(lap_progress >= 1.0)])
    else:
        first_lap_time = float("nan")

    performance_ok = (ss_pos <= max_ss_pos) and (ss_ori <= max_ss_ori)
    # success = coverage >= coverage_threshold and laps >= min_laps and not is_stuck
    success = (
        coverage >= coverage_threshold
        and laps >= min_laps
        and not is_stuck
        and performance_ok
    )  # Added performance check

    return {
        "coverage": coverage,
        "laps": laps,
        "max_dwell_frac": max_dwell_frac_actual,
        "is_stuck": is_stuck,
        "ss_dist": ss_dist,
        "ss_pos": ss_pos,
        "ss_ori": ss_ori,
        "max_dist_ss": max_dist,
        "final_dist": float(dist[-1]),
        "first_lap_time": first_lap_time,
        "performance_ok": performance_ok,  # Log this for debugging
        "success": success,
    }
    # return {
    #     "coverage": coverage,
    #     "laps": laps,
    #     "max_dwell_frac": max_dwell_frac_actual,
    #     "is_stuck": is_stuck,
    #     "ss_dist": ss_dist,
    #     "ss_pos": ss_pos,
    #     "ss_ori": ss_ori,
    #     "max_dist_ss": max_dist,
    #     "final_dist": float(dist[-1]),
    #     "first_lap_time": first_lap_time,
    #     "success": success,
    # }


def analyze_pickle(pkl_path, curve_npy_path=None, include_types=("nominal",), **kwargs):
    print("\n" + "=" * 68)
    print(f"Analyzing: {pkl_path}")
    if include_types is not None:
        print(f"  Filter: only runs with init_type in {include_types}")
    else:
        print(f"  Filter: none (all runs included)")
    print("=" * 68)

    with open(pkl_path, "rb") as f:
        data = pickle.load(f)

    # --- extract run IDs that actually exist in the pickle ---
    all_run_ids = sorted(
        {
            int(k[len("run_") : -len("_idx")])
            for k in data
            if k.startswith("run_") and k.endswith("_idx")
        }
    )
    print(f"  Runs present in file: {len(all_run_ids)}")

    # --- filter by init_type (treat missing meta as 'nominal' for backward compat) ---
    run_ids = []
    for rid in all_run_ids:
        init_type = data.get(f"run_{rid:03d}_meta_init_type", "nominal")
        if include_types is None or init_type in include_types:
            run_ids.append(rid)
    print(
        f"  Using runs: {len(run_ids)}  "
        f"(ids: {run_ids[:5]}{'...' if len(run_ids) > 5 else ''})"
    )

    if not run_ids:
        print("  No runs matched the filter. Aborting analysis for this curve.")
        return []

    # --- N from curve file if available, otherwise infer ---
    if curve_npy_path and os.path.exists(curve_npy_path):
        curve = np.load(curve_npy_path, allow_pickle=True)
        N = len(curve)
        print(f"  N = {N} (from curve file)")
    else:
        sample = np.concatenate([data[f"run_{rid:03d}_idx"] for rid in run_ids[:20]])
        N = int(sample.max()) + 1
        print(f"  N = {N} (inferred from a sample of runs)")

    # --- evaluate each run ---
    results = []
    for rid in run_ids:
        prefix = f"run_{rid:03d}_"
        r = analyze_run(
            data[prefix + "idx"],
            data[prefix + "dist"],
            data[prefix + "err_pos"],
            data[prefix + "err_ori"],
            data[prefix + "time"],
            N,
            **kwargs,
        )
        r["run_id"] = rid
        r["init_type"] = data.get(f"{prefix}meta_init_type", "nominal")
        results.append(r)

    successes = [r for r in results if r["success"]]
    failures = [r for r in results if not r["success"]]

    print(
        f"\nSuccess rate: {len(successes)}/{len(results)} = "
        f"{100.0 * len(successes) / len(results):.1f}%"
    )

    if successes:
        print("\nMetrics (successful runs only):")
        for metric in [
            "ss_dist",
            "ss_pos",
            "ss_ori",
            "max_dist_ss",
            "first_lap_time",
            "laps",
            "coverage",
            "max_dwell_frac",
        ]:
            vals = np.array([r[metric] for r in successes])
            vals = vals[~np.isnan(vals)]
            if len(vals):
                print(
                    f"  {metric:16s}: mean={vals.mean():9.4f}  std={vals.std():9.4f}  "
                    f"median={np.median(vals):9.4f}  min={vals.min():9.4f}  max={vals.max():9.4f}"
                )

    # --- breakdown by init_type (informative once problematic runs are merged in) ---
    types_present = sorted({r["init_type"] for r in results})
    if len(types_present) > 1:
        print("\nBreakdown by init_type:")
        for t in types_present:
            group = [r for r in results if r["init_type"] == t]
            n_ok = sum(1 for r in group if r["success"])
            print(f"  {t:10s}: {n_ok}/{len(group)} = {100.0 * n_ok / len(group):5.1f}%")

    if failures:
        print(f"\nFailed runs ({len(failures)}):")
        for r in failures:
            reasons = []
            if r["coverage"] < kwargs.get("coverage_threshold", 0.90):
                reasons.append(f"coverage={r['coverage']:.2f}")
            if r["laps"] < kwargs.get("min_laps", 1.0):
                reasons.append(f"laps={r['laps']:.2f}")
            if r["is_stuck"]:
                reasons.append(f"stuck(dwell={r['max_dwell_frac']:.2f})")
            if not r.get("performance_ok", True):
                reasons.append(
                    f"poor_tracking(pos={r['ss_pos']:.1f}cm, ori={r['ss_ori']:.1f}°)"
                )
            # --- Extract the scale factor and base radii for this curve ---
            curve_seed = r["run_id"] // 5  # N_INITS=5
            rng = np.random.default_rng(curve_seed)
            _ = rng.uniform(0, 2 * np.pi)  # first draw: global yaw
            scale = rng.uniform(0.8, 1.5)  # second draw: scale

            # Base ellipse radii from your simulation script (A=2.0, B=1.0)
            a = 2.0 * scale
            b = 1.0 * scale

            print(
                f"  Run {r['run_id']:3d} [{r['init_type']:10s}]: "
                + ", ".join(reasons)
                + f"  | ss_dist={r['ss_dist']:.3f}  ss_pos={r['ss_pos']:.1f}cm"
                + f"  | curve_seed={curve_seed} (a={a:.2f}, b={b:.2f})"
            )

        seed_stats = {}
        for r in results:
            seed = r["run_id"] // 5  # N_INITS=5
            s = seed_stats.setdefault(seed, {"ok": 0, "total": 0})
            s["total"] += 1
            s["ok"] += int(r["success"])

        print(f"\nPer-seed summary (sorted by scale):")
        print(
            f"  {'seed':>4}  {'scale':>5}  {'a':>5}  {'b':>5}  {'roll°':>6}  {'pitch°':>7}  {'yaw_off°':>7}  {'success':>9}"
        )
        rows = []
        for seed, s in seed_stats.items():
            rng = np.random.default_rng(seed)
            _ = rng.uniform(0, 2 * np.pi)  # yaw_global
            scale = rng.uniform(0.8, 1.5)
            roll = np.degrees(rng.uniform(-np.pi / 12, np.pi / 12))
            pitch = np.degrees(rng.uniform(-np.pi / 12, np.pi / 12))
            yaw_off = np.degrees(rng.uniform(-np.pi, np.pi))  # <-- add this
            rows.append(
                (
                    scale,
                    seed,
                    3.0 * scale,
                    2.0 * scale,
                    roll,
                    pitch,
                    yaw_off,
                    s["ok"],
                    s["total"],
                )
            )

        rows.sort()
        for scale, seed, a, b, roll, pitch, yaw_off, ok, tot in rows:
            print(
                f"  {seed:>4}  {scale:>5.2f}  {a:>5.2f}  {b:>5.2f}  {roll:>6.1f}  {pitch:>7.1f}  {yaw_off:>7.1f}  {ok:>4}/{tot}"
            )

    return results


if __name__ == "__main__":
    results_dir = "./fossen/data"
    curves_dir = "./fossen/data"

    for curve_type in ["ellipse", "trefoil", "vertical_coil"]:
        pkl_path = os.path.join(results_dir, f"{curve_type}_results.pkl")
        npy_path = os.path.join(curves_dir, f"{curve_type}.npy")
        if os.path.exists(pkl_path):
            analyze_pickle(pkl_path, npy_path, min_laps=0.6,)
        else:
            print(f"Skipping {curve_type}: {pkl_path} not found")

print(2)
# if __name__ == "__main__":
#     results_dir = "./fossen/data"
#     curves_dir = "./fossen/data"
#
#     for curve_type in ["ellipse", "trefoil", "vertical_coil"]:
#         pkl = os.path.join(results_dir, f"{curve_type}_results.pkl")
#         npy = os.path.join(curves_dir, f"{curve_type}.npy")
#         dnpy = os.path.join(curves_dir, f"{curve_type}_derivative.npy")
#         if os.path.exists(pkl):
#             analyze_pickle_recomputed_v2(pkl, npy, dnpy)
#         else:
#             print(f"Skipping {curve_type}: {pkl} not found")
