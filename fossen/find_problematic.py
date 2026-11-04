#!/usr/bin/env python3
import time
import argparse
import os
import pickle
import numpy as np
from scipy.spatial.transform import Rotation as R
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
import heapq

try:
    import uaibot as ub
    import uaibot_cpp_bind as ub_cpp

    HAS_UAIBOT = True
except ImportError:
    HAS_UAIBOT = False
    print(
        "[warn] uaibot not available; P2 will use the numeric fallback "
        "(position-only Hessian, ignores orientation coupling)."
    )

# =============================================================================
# CONFIG  — edit this block to change behaviour
# =============================================================================
CURVE_TYPES = ["ellipse", "trefoil", "vertical_coil"]
CURVE_SEEDS = [0]  # e.g. [0]  or  list(range(20))  for all simulation seeds
N_CANDIDATES = 50000
DATA_DIR = "./data"
RADIUS_FRAC = 1.5
TOP_K = 10
TOP_SAVE = 100  # how many HTMs to persist per score in the .pkl
PROGRESS_EVERY = 1000


# --------------------------------------------------------------- parallel worker
_CURVE = None
_SEED = None


def _init_worker(curve, seed):
    global _CURVE, _SEED
    _CURVE = curve
    _SEED = seed


def _worker_chunk(chunk_indices):
    """Evaluate candidates in `chunk_indices`, return all scores + local top-N."""
    curve = _CURVE
    seed = _SEED
    use_cpp = HAS_UAIBOT

    top_p1 = []
    top_p2 = []
    results = []

    for k in chunk_indices:
        # Per-candidate deterministic RNG so chunk order does not matter
        cand_rng = np.random.default_rng(np.array([seed, int(k)], dtype=np.uint64))
        htm = sample_random_htm(curve, cand_rng, radius_frac=RADIUS_FRAC)

        s1 = score_p1(htm, curve)
        s2 = score_p2_cpp(htm, curve) if use_cpp else score_p2_fallback(htm, curve)
        results.append((int(k), float(s1), float(s2)))

        heapq.heappush(top_p1, (s1, int(k), htm))
        if len(top_p1) > TOP_SAVE:
            heapq.heappop(top_p1)

        heapq.heappush(top_p2, (s2, int(k), htm))
        if len(top_p2) > TOP_SAVE:
            heapq.heappop(top_p2)

    return results, top_p1, top_p2


# ---------------------------------------------------------------- curve gen
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
    out = np.zeros_like(curve)
    for i in range(N):
        H = curve[i].copy()
        H[:3, :3] = R_global @ H[:3, :3] @ R_offset
        H[:3, 3] = scale * (R_global @ H[:3, 3])
        out[i] = H
    return out


# --------------------------------------------------------------- scoring P1
def score_p1(htm, curve, min_idx_diff_k=5, eps=1e-12):
    """HIGH = ambiguous (a distant sample is nearly as close as the closest)."""
    N = len(curve)
    dists = [ub_cpp.ee_distance(htm, Hd) for Hd in curve]
    u = int(np.argmin(dists))
    D_min = dists[u]

    best_r = np.inf
    for v in range(N):
        if v == u:
            continue
        idx_diff = abs(u - v)
        min_idx_diff_v = min(idx_diff, N - idx_diff)
        if min_idx_diff_v <= min_idx_diff_k:
            continue
        r_v = abs(D_min - dists[v]) / min_idx_diff_v
        if r_v < best_r:
            best_r = r_v

    return 1.0 / (best_r + eps)


# --------------------------------------------------------------- scoring P2
def get_LvDhat(htm, curve, ds=1.0 / 30000):
    xi, _, _ = ub.Robot.vector_field_SE3(
        htm,
        curve,
        kt1=0.0,
        kt2=1.0,
        kt3=1.0,
        kn1=1.0,
        kn2=1e9,
        ds=ds,
    )
    return np.asarray(xi).reshape(-1)


def score_p2_cpp(htm, curve, eps=5e-3):
    base_L = get_LvDhat(htm, curve)
    J = np.zeros((6, 6))
    for i in range(6):
        xi = np.zeros(6)
        xi[i] = eps
        variation = ub_cpp.expSE3(ub_cpp.SmapSE3(xi)) @ htm
        L_i = get_LvDhat(variation, curve)
        J[:, i] = (L_i - base_L) / eps
    return float(np.linalg.norm(J, "fro"))


def score_p2_fallback(htm, curve, eps=5e-3):
    """Position-only numerical Hessian of distance-to-curve. Ignores orientation."""
    p = htm[:3, 3]
    curve_pos = curve[:, :3, 3]

    def d(q):
        return float(np.linalg.norm(curve_pos - q, axis=1).min())

    H = np.zeros((3, 3))
    for i in range(3):
        for j in range(3):
            ei = np.zeros(3)
            ei[i] = eps
            ej = np.zeros(3)
            ej[j] = eps
            H[i, j] = (
                d(p + ei + ej) - d(p + ei - ej) - d(p - ei + ej) + d(p - ei - ej)
            ) / (4 * eps * eps)
    # Pad to 6x6 so scale matches the C++ version roughly
    H6 = np.zeros((6, 6))
    H6[:3, :3] = H
    return float(np.linalg.norm(H6, "fro"))


# --------------------------------------------------------------- sampling
def sample_random_htm(curve, rng, radius_frac=1.5):
    """Position uniform in a ball containing the curve; orientation = yaw only, full 360°."""
    curve_pos = curve[:, :3, 3]
    centroid = curve_pos.mean(axis=0)
    ball_radius = radius_frac * np.linalg.norm(curve_pos - centroid, axis=1).max()

    # Uniform inside ball
    dir_vec = rng.standard_normal(3)
    dir_vec /= np.linalg.norm(dir_vec)
    r = ball_radius * rng.uniform(0, 1) ** (1 / 3)
    p = centroid + dir_vec * r

    # Random yaw only — roll and pitch locked to 0
    yaw = rng.uniform(-np.pi, np.pi)
    R_rand = R.from_euler("xyz", [0.0, 0.0, yaw]).as_matrix()

    htm = np.eye(4)
    htm[:3, :3] = R_rand
    htm[:3, 3] = p
    return htm


# --------------------------------------------------------------- main
def run_search(curve_type, seed):
    base_path = os.path.join(DATA_DIR, f"{curve_type}.npy")
    dbase_path = os.path.join(DATA_DIR, f"{curve_type}_derivative.npy")
    if not (os.path.exists(base_path) and os.path.exists(dbase_path)):
        raise FileNotFoundError(f"Missing {base_path} or {dbase_path}.")

    curve_base = np.load(base_path, allow_pickle=True)
    dcurve_base = np.load(dbase_path, allow_pickle=True)
    curve = rand_se3_curve(curve_base, dcurve_base, seed=seed)
    N = len(curve)

    use_cpp = HAS_UAIBOT
    scorer_name = "C++ L-operator" if use_cpp else "numeric fallback"

    n_workers = min(multiprocessing.cpu_count(), N_CANDIDATES)
    chunks = np.array_split(np.arange(N_CANDIDATES), n_workers)

    print(f"\n{'='*72}")
    print(f"  Curve: {curve_type}   seed: {seed}   N = {N}")
    print(f"  Candidates: {N_CANDIDATES}   P2 scorer: {scorer_name}")
    print(f"  Workers: {n_workers}   chunks: {len(chunks)}")
    print(f"{'='*72}\n")

    # --- parallel evaluation ---
    t0 = time.time()
    chunk_results = []
    with ProcessPoolExecutor(
        max_workers=n_workers,
        initializer=_init_worker,
        initargs=(curve, seed),
    ) as ex:
        futures = {ex.submit(_worker_chunk, chunk): i for i, chunk in enumerate(chunks)}
        for j, fut in enumerate(as_completed(futures)):
            chunk_results.append(fut.result())
            elapsed = time.time() - t0
            print(
                f"  [{curve_type} seed={seed}] chunk {j+1}/{len(chunks)} done "
                f"({elapsed/60:.1f} min elapsed)",
                flush=True,
            )

    print(f"\n  done in {(time.time() - t0)/60:.1f} min")

    # --- merge chunk results ---
    all_scores = np.full((N_CANDIDATES, 2), np.nan)
    top_p1 = []
    top_p2 = []
    for results, tp1, tp2 in chunk_results:
        for k, s1, s2 in results:
            all_scores[k, 0] = s1
            all_scores[k, 1] = s2
        top_p1.extend(tp1)
        top_p2.extend(tp2)

    top_p1 = sorted(top_p1, key=lambda x: -x[0])[:TOP_SAVE]
    top_p2 = sorted(top_p2, key=lambda x: -x[0])[:TOP_SAVE]

    p1_scores = all_scores[:, 0]
    p2_scores = all_scores[:, 1]

    # --- reporting (unchanged) ---
    print(
        f"\nP1: min={p1_scores.min():.4e}  median={np.median(p1_scores):.4e}  "
        f"p90={np.percentile(p1_scores, 90):.4e}  p99={np.percentile(p1_scores, 99):.4e}  "
        f"max={p1_scores.max():.4e}"
    )
    print(
        f"P2: min={p2_scores.min():.4e}  median={np.median(p2_scores):.4e}  "
        f"p90={np.percentile(p2_scores, 90):.4e}  p99={np.percentile(p2_scores, 99):.4e}  "
        f"max={p2_scores.max():.4e}"
    )

    order1 = np.argsort(-p1_scores)
    order2 = np.argsort(-p2_scores)
    rank1 = np.empty_like(p1_scores, dtype=int)
    rank1[order1] = np.arange(len(order1))
    rank2 = np.empty_like(p2_scores, dtype=int)
    rank2[order2] = np.arange(len(order2))
    combined = rank1 + rank2
    j_worst = int(np.argmin(combined))

    worst_htm = None
    for _, k, h in top_p1 + top_p2:
        if k == j_worst:
            worst_htm = h
            break

    top_p1_list = sorted(top_p1, key=lambda x: -x[0])[:TOP_K]
    top_p2_list = sorted(top_p2, key=lambda x: -x[0])[:TOP_K]

    print(f"\nTop {TOP_K} by P1 (most ambiguous):")
    for s, k, h in top_p1_list:
        p = h[:3, 3]
        print(
            f"  idx={k:>6}  P1={p1_scores[k]:.4e}  P2={p2_scores[k]:.4e}  "
            f"pos=[{p[0]:.3f}, {p[1]:.3f}, {p[2]:.3f}]"
        )

    print(f"\nTop {TOP_K} by P2 (highest L-operator Jacobian):")
    for s, k, h in top_p2_list:
        p = h[:3, 3]
        print(
            f"  idx={k:>6}  P2={p2_scores[k]:.4e}  P1={p1_scores[k]:.4e}  "
            f"pos=[{p[0]:.3f}, {p[1]:.3f}, {p[2]:.3f}]"
        )

    if worst_htm is not None:
        p = worst_htm[:3, 3]
        print(f"\nWorst combined (in top-{TOP_SAVE} of both):")
        print(
            f"  idx={j_worst}  P1={p1_scores[j_worst]:.4e}  P2={p2_scores[j_worst]:.4e}  "
            f"pos=[{p[0]:.3f}, {p[1]:.3f}, {p[2]:.3f}]"
        )

    out = f"./problematic_{curve_type}_seed{seed}.pkl"
    with open(out, "wb") as f:
        pickle.dump(
            {
                "curve_type": curve_type,
                "curve_seed": seed,
                "curve": curve,
                "n_candidates": N_CANDIDATES,
                "all_p1": p1_scores,
                "all_p2": p2_scores,
                "top_p1_htms": [h for _, _, h in top_p1_list],
                "top_p1_scores": [s for s, _, _ in top_p1_list],
                "top_p2_htms": [h for _, _, h in top_p2_list],
                "top_p2_scores": [s for s, _, _ in top_p2_list],
                "worst_both_htm": worst_htm,
                "worst_both_idx": j_worst,
            },
            f,
            protocol=pickle.HIGHEST_PROTOCOL,
        )
    print(f"\nSaved to {out}\n")


# def run_search(curve_type, seed):
#     base_path = os.path.join(DATA_DIR, f"{curve_type}.npy")
#     dbase_path = os.path.join(DATA_DIR, f"{curve_type}_derivative.npy")
#     if not (os.path.exists(base_path) and os.path.exists(dbase_path)):
#         raise FileNotFoundError(f"Missing {base_path} or {dbase_path}.")
#
#     curve_base = np.load(base_path, allow_pickle=True)
#     dcurve_base = np.load(dbase_path, allow_pickle=True)
#     curve = rand_se3_curve(curve_base, dcurve_base, seed=seed)
#     N = len(curve)
#
#     use_cpp = HAS_UAIBOT
#     scorer_name = "C++ L-operator" if use_cpp else "numeric fallback"
#
#     print(f"\n{'='*72}")
#     print(f"  Curve: {curve_type}   seed: {seed}   N = {N}")
#     print(f"  Candidates: {N_CANDIDATES}   P2 scorer: {scorer_name}")
#     print(f"{'='*72}\n")
#
#     rng = np.random.default_rng(seed + 10_000)
#
#     # Pre-allocate, keep only top-Top_save HTMs in memory at any time
#     p1_scores = np.zeros(N_CANDIDATES)
#     p2_scores = np.zeros(N_CANDIDATES)
#
#     import heapq
#
#     top_p1 = []  # heap of (-score, idx, htm)
#     top_p2 = []
#     worst_both = (
#         []
#     )  # heap of (combined_rank_lower_is_better_proxy, idx, htm) -- filled later
#
#     t0 = time.time()
#     for k in range(N_CANDIDATES):
#         htm = sample_random_htm(curve, rng, radius_frac=RADIUS_FRAC)
#         s1 = score_p1(htm, curve)
#         s2 = score_p2_cpp(htm, curve) if use_cpp else score_p2_fallback(htm, curve)
#         p1_scores[k] = s1
#         p2_scores[k] = s2
#
#         heapq.heappush(top_p1, (s1, k, htm))
#         if len(top_p1) > TOP_SAVE:
#             heapq.heappop(top_p1)
#
#         heapq.heappush(top_p2, (s2, k, htm))
#         if len(top_p2) > TOP_SAVE:
#             heapq.heappop(top_p2)
#
#         if (k + 1) % PROGRESS_EVERY == 0:
#             elapsed = time.time() - t0
#             rate = (k + 1) / elapsed
#             eta = (N_CANDIDATES - k - 1) / rate
#             print(
#                 f"  {k+1:>6}/{N_CANDIDATES}   "
#                 f"{rate:6.0f} eval/s   ETA {eta/60:6.1f} min"
#             )
#
#     print(f"\n  done in {(time.time() - t0)/60:.1f} min")
#
#     print(
#         f"\nP1: min={p1_scores.min():.4e}  median={np.median(p1_scores):.4e}  "
#         f"p90={np.percentile(p1_scores, 90):.4e}  p99={np.percentile(p1_scores, 99):.4e}  "
#         f"max={p1_scores.max():.4e}"
#     )
#     print(
#         f"P2: min={p2_scores.min():.4e}  median={np.median(p2_scores):.4e}  "
#         f"p90={np.percentile(p2_scores, 90):.4e}  p99={np.percentile(p2_scores, 99):.4e}  "
#         f"max={p2_scores.max():.4e}"
#     )
#
#     # Rank-based worst-on-both
#     order1 = np.argsort(-p1_scores)
#     order2 = np.argsort(-p2_scores)
#     rank1 = np.empty_like(p1_scores, dtype=int)
#     rank1[order1] = np.arange(len(order1))
#     rank2 = np.empty_like(p2_scores, dtype=int)
#     rank2[order2] = np.arange(len(order2))
#     combined = rank1 + rank2
#     j_worst = int(np.argmin(combined))
#     htm_worst = sample_random_htm(
#         curve, np.random.default_rng(-1), RADIUS_FRAC
#     )  # placeholder, overwritten below
#     # Recover actual HTM for j_worst from top_p1/top_p2 or re-sample deterministically is not possible;
#     # instead save rank arrays and just report the position, which requires re-deriving.
#     # Simplest: keep a lightweight htm array only if j_worst is in our top_save set.
#     worst_htm = None
#     for s, k, h in top_p1 + top_p2:
#         if k == j_worst:
#             worst_htm = h
#             break
#
#     top_p1_list = sorted(top_p1, key=lambda x: -x[0])[:TOP_K]
#     top_p2_list = sorted(top_p2, key=lambda x: -x[0])[:TOP_K]
#
#     print(f"\nTop {TOP_K} by P1 (most ambiguous):")
#     for s, k, h in top_p1_list:
#         p = h[:3, 3]
#         print(
#             f"  idx={k:>6}  P1={p1_scores[k]:.4e}  P2={p2_scores[k]:.4e}  "
#             f"pos=[{p[0]:.3f}, {p[1]:.3f}, {p[2]:.3f}]"
#         )
#
#     print(f"\nTop {TOP_K} by P2 (highest L-operator Jacobian):")
#     for s, k, h in top_p2_list:
#         p = h[:3, 3]
#         print(
#             f"  idx={k:>6}  P2={p2_scores[k]:.4e}  P1={p1_scores[k]:.4e}  "
#             f"pos=[{p[0]:.3f}, {p[1]:.3f}, {p[2]:.3f}]"
#         )
#
#     if worst_htm is not None:
#         p = worst_htm[:3, 3]
#         print(f"\nWorst combined (in top-{TOP_SAVE} of both):")
#         print(
#             f"  idx={j_worst}  P1={p1_scores[j_worst]:.4e}  P2={p2_scores[j_worst]:.4e}  "
#             f"pos=[{p[0]:.3f}, {p[1]:.3f}, {p[2]:.3f}]"
#         )
#
#     out = f"./problematic_{curve_type}_seed{seed}.pkl"
#     with open(out, "wb") as f:
#         pickle.dump(
#             {
#                 "curve_type": curve_type,
#                 "curve_seed": seed,
#                 "curve": curve,
#                 "n_candidates": N_CANDIDATES,
#                 "all_p1": p1_scores,
#                 "all_p2": p2_scores,
#                 "top_p1_htms": [h for _, _, h in top_p1_list],
#                 "top_p1_scores": [s for s, _, _ in top_p1_list],
#                 "top_p2_htms": [h for _, _, h in top_p2_list],
#                 "top_p2_scores": [s for s, _, _ in top_p2_list],
#                 "worst_both_htm": worst_htm,
#                 "worst_both_idx": j_worst,
#             },
#             f,
#             protocol=pickle.HIGHEST_PROTOCOL,
#         )
#     print(f"\nSaved to {out}\n")


def main():
    for curve_type in CURVE_TYPES:
        for seed in CURVE_SEEDS:
            run_search(curve_type, seed)


if __name__ == "__main__":
    main()
