#!/usr/bin/env python3
import torch
import time
import os
import pickle
import numpy as np
from scipy.spatial.transform import Rotation as R

from torch_vf import SE3VectorField

# =============================================================================
# CONFIG  — edit this block to change behaviour
# =============================================================================
CURVE_TYPES = ["ellipse", "trefoil", "vertical_coil"]
CURVE_SEEDS = list(range(20))  # e.g. [0]  or  list(range(20))
N_CANDIDATES = 50000  # raise as high as your GPU memory allows
DATA_DIR = "./data"
RADIUS_FRAC = 1.5
TOP_K = 10

# ----------------------- FIXED PARAMETERS -----------------------------
# Torch configuration — do NOT force CPU here.  This script is only ever
# used for the L-operator score and always uses SE3VectorField.
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float64

# Instantiate the solver once (delta / ds match the C++ constants c_delta / c_ds)
vf_solver = SE3VectorField(delta=1e-3, ds=1e-3)


# =============================================================================
# Curve generation
# =============================================================================
def rand_se3_curve(curve, curve_deriv, seed=None):
    rng = np.random.default_rng(seed)
    N = len(curve)
    yaw_global = rng.uniform(0, 2 * np.pi)
    R_global = R.from_euler("z", yaw_global).as_matrix()
    scale = rng.uniform(0.8, 1.5)
    roll_offset = rng.uniform(-np.pi / 12, np.pi / 12)
    pitch_offset = rng.uniform(-np.pi / 12, np.pi / 12)
    yaw_offset = rng.uniform(0, 2 * np.pi)
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


def sample_random_htm(curve, rng, radius_frac=1.5):
    """Position uniform in a ball containing the curve; orientation = yaw only."""
    curve_pos = curve[:, :3, 3]
    centroid = curve_pos.mean(axis=0)
    ball_radius = radius_frac * np.linalg.norm(curve_pos - centroid, axis=1).max()

    dir_vec = rng.standard_normal(3)
    dir_vec /= np.linalg.norm(dir_vec)
    r = ball_radius * rng.uniform(0, 1) ** (1 / 3)
    p = centroid + dir_vec * r

    yaw = rng.uniform(-np.pi, np.pi)
    R_rand = R.from_euler("xyz", [0.0, 0.0, yaw]).as_matrix()

    htm = np.eye(4)
    htm[:3, :3] = R_rand
    htm[:3, 3] = p
    return htm


# =============================================================================
# Full SE(3)-distance matrix
# (needed by P1; SE3VectorField.ecdistance only returns the min)
# =============================================================================
def eedistance_matrix(vf, states, curve, chunk_size=512):
    """Return the full (N, M) SE(3)-distance matrix, chunked over N.

    Mirrors SE3VectorField.ecdistance but keeps every pair instead of argmin.
    """
    N, M = states.shape[0], curve.shape[0]
    device, dtype = states.device, states.dtype

    R_V = states[..., :3, :3]
    p_V = states[..., :3, 3]
    R_V_T = R_V.transpose(-1, -2)

    R_W = curve[..., :3, :3]
    p_W = curve[..., :3, 3]

    out = torch.empty(N, M, device=device, dtype=dtype)

    for start in range(0, N, chunk_size):
        end = min(start + chunk_size, N)

        R_V_c = R_V[start:end]
        R_V_T_c = R_V_T[start:end]
        p_V_c = p_V[start:end]

        # trace(Q) = <R_V, R_W>_F : (C, M)
        trace_Q = torch.einsum("cij,mij->cm", R_V_c, R_W)

        # t = R_V^T (p_W - p_V) : (C, M, 3)
        d_W = p_W.unsqueeze(0) - p_V_c.unsqueeze(1)  # (C, M, 3)
        t = torch.einsum("cij,cmj->cmi", R_V_T_c, d_W)
        t_sq = (t * t).sum(-1)

        # t^T Q t = d_W · (R_W t)
        RW_t = torch.einsum("mij,cmj->cmi", R_W, t)
        tQt = (d_W * RW_t).sum(-1)

        # angle theta
        cos_theta = ((trace_Q - 1.0) * 0.5).clamp(-1.0, 1.0)
        sin_theta = torch.sqrt((1.0 - cos_theta**2).clamp_min(0.0))
        theta = torch.atan2(sin_theta, cos_theta)

        # alpha with numeric safety near cos_theta ~ 1
        mask = cos_theta > vf.max_cos_theta
        one_c = (1.0 - cos_theta).clamp_min(1e-12)
        alpha = torch.where(
            mask,
            torch.full_like(theta, -1.0 / 12.0),
            (2.0 - 2.0 * cos_theta - theta**2) / (4.0 * one_c**2),
        )

        # d^2 = 2*theta^2 + t^T M t
        tMt = 2.0 * alpha * tQt + (1.0 - 2.0 * alpha) * t_sq
        out[start:end] = torch.sqrt((2.0 * theta**2 + tMt).clamp_min(0.0))

    return out


# =============================================================================
# Batched P1 score
# =============================================================================
def score_p1_batch(
    htm_batch_t, curve_t, vf, min_idx_diff_k=5, eps=1e-12, chunk_size=512
):
    """htm_batch_t: (N, 4, 4) on device. Returns (N,) numpy scores.

    HIGH = ambiguous (a distant sample is nearly as close as the closest).
    """
    N, M = htm_batch_t.shape[0], curve_t.shape[0]
    out = np.empty(N, dtype=np.float64)

    idx = torch.arange(M, device=htm_batch_t.device)

    with torch.no_grad():
        for start in range(0, N, chunk_size):
            end = min(start + chunk_size, N)
            htm_c = htm_batch_t[start:end]  # (C, 4, 4)

            dists = eedistance_matrix(
                vf, htm_c, curve_t, chunk_size=chunk_size
            )  # (C, M)
            u = torch.argmin(dists, dim=1)  # (C,)
            D_min = dists.gather(1, u.unsqueeze(1)).squeeze(1)  # (C,)

            idx_diff = (idx.unsqueeze(0) - u.unsqueeze(1)).abs()  # (C, M)
            min_idx_diff = torch.minimum(idx_diff, M - idx_diff)  # (C, M)
            valid = min_idx_diff > min_idx_diff_k  # (C, M)

            denom = min_idx_diff.to(dists.dtype).clamp_min(1.0)
            r_v = (dists - D_min.unsqueeze(1)).abs() / denom
            r_v = torch.where(valid, r_v, torch.full_like(r_v, float("inf")))

            best_r = r_v.min(dim=1).values  # (C,)
            out[start:end] = (1.0 / (best_r + eps)).cpu().numpy()

    return out


# =============================================================================
# Batched P2 score
# =============================================================================
def score_p2_batch(htm_batch_t, curve_t, vf, eps=5e-3, chunk_size=512):
    """htm_batch_t: (N, 4, 4) on device. Returns (N,) numpy Frobenius norms."""
    N = htm_batch_t.shape[0]
    out = np.empty(N, dtype=np.float64)

    I6 = torch.eye(6, dtype=htm_batch_t.dtype, device=htm_batch_t.device)
    basis = vf.smap_se3(I6)  # (6, 4, 4), unit

    with torch.no_grad():
        for start in range(0, N, chunk_size):
            end = min(start + chunk_size, N)
            htm_c = htm_batch_t[start:end]
            C = end - start

            # Base L-operator for all candidates in one VF call.
            base_L, _, _ = vf.vectorfield_se3(
                htm_c,
                curve_t,
                kt1=0.0,
                kt2=1.0,
                kt3=1.0,
                kn1=1.0,
                kn2=1e9,
            )  # (C, 6)

            # 6 Lie-algebra perturbations, batched across candidates:
            #   variations[k, i] = exp(eps * basis_k) @ htm_c[i]
            variations = vf.exp_se3(basis.unsqueeze(1) * eps) @ htm_c.unsqueeze(
                0
            )  # (6, C, 4, 4)
            variations_flat = variations.reshape(6 * C, 4, 4)

            # One VF call for all 6*C perturbations.
            Ls, _, _ = vf.vectorfield_se3(
                variations_flat,
                curve_t,
                kt1=0.0,
                kt2=1.0,
                kt3=1.0,
                kn1=1.0,
                kn2=1e9,
            )  # (6C, 6)
            Ls = Ls.reshape(6, C, 6)

            # J[:, i, :] is the 6x6 Jacobian rows for candidate i.
            J = (Ls - base_L.unsqueeze(0)) / eps  # (6, C, 6)
            out[start:end] = torch.sqrt((J**2).sum(dim=(0, 2))).cpu().numpy()

    return out


# =============================================================================
# Main search
# =============================================================================
def run_search(curve_type, seed):
    base_path = os.path.join(DATA_DIR, f"{curve_type}.npy")
    dbase_path = os.path.join(DATA_DIR, f"{curve_type}_derivative.npy")
    if not (os.path.exists(base_path) and os.path.exists(dbase_path)):
        raise FileNotFoundError(f"Missing {base_path} or {dbase_path}.")

    curve_base = np.load(base_path, allow_pickle=True)
    dcurve_base = np.load(dbase_path, allow_pickle=True)
    curve, curve_deriv = rand_se3_curve(curve_base, dcurve_base, seed=seed)
    N = len(curve)

    print(f"\n{'='*72}")
    print(f"  Curve: {curve_type}   seed: {seed}   N = {N}")
    print(f"  Candidates: {N_CANDIDATES}   DEVICE: {DEVICE}   dtype: {DTYPE}")
    print(f"{'='*72}\n")

    # ---- 1. Sample all candidates (per-candidate RNG for reproducibility) ----
    t0 = time.time()
    htm_all = np.empty((N_CANDIDATES, 4, 4), dtype=np.float64)
    for k in range(N_CANDIDATES):
        cand_rng = np.random.default_rng(np.array([seed, int(k)], dtype=np.uint64))
        htm_all[k] = sample_random_htm(curve, cand_rng, radius_frac=RADIUS_FRAC)
    print(f"  sampled {N_CANDIDATES} HTMs in {time.time()-t0:.2f}s")

    # ---- 2. Batched GPU scoring ---------------------------------------------
    curve_t = torch.as_tensor(curve, dtype=DTYPE, device=DEVICE)
    htm_all_t = torch.as_tensor(htm_all, dtype=DTYPE, device=DEVICE)

    t0 = time.time()
    p1_scores = score_p1_batch(htm_all_t, curve_t, vf_solver)
    print(f"  P1 done in {time.time()-t0:.2f}s")

    t0 = time.time()
    p2_scores = score_p2_batch(htm_all_t, curve_t, vf_solver)
    print(f"  P2 done in {time.time()-t0:.2f}s")

    # ---- 3. Reporting -------------------------------------------------------
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

    top_p1_idx = order1[:TOP_K]
    top_p2_idx = order2[:TOP_K]

    print(f"\nTop {TOP_K} by P1 (most ambiguous):")
    for k in top_p1_idx:
        p = htm_all[k, :3, 3]
        print(
            f"  idx={k:>6}  P1={p1_scores[k]:.4e}  P2={p2_scores[k]:.4e}  "
            f"pos=[{p[0]:.3f}, {p[1]:.3f}, {p[2]:.3f}]"
        )

    print(f"\nTop {TOP_K} by P2 (highest L-operator Jacobian):")
    for k in top_p2_idx:
        p = htm_all[k, :3, 3]
        print(
            f"  idx={k:>6}  P2={p2_scores[k]:.4e}  P1={p1_scores[k]:.4e}  "
            f"pos=[{p[0]:.3f}, {p[1]:.3f}, {p[2]:.3f}]"
        )

    p = htm_all[j_worst, :3, 3]
    print(f"\nWorst combined (rank sum of P1 and P2):")
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
                "top_p1_htms": htm_all[top_p1_idx].copy(),
                "top_p1_scores": p1_scores[top_p1_idx].copy(),
                "top_p2_htms": htm_all[top_p2_idx].copy(),
                "top_p2_scores": p2_scores[top_p2_idx].copy(),
                "worst_both_htm": htm_all[j_worst].copy(),
                "worst_both_idx": j_worst,
            },
            f,
            protocol=pickle.HIGHEST_PROTOCOL,
        )
    print(f"\nSaved to {out}\n")


def main():
    for curve_type in CURVE_TYPES:
        for seed in CURVE_SEEDS:
            run_search(curve_type, seed)


if __name__ == "__main__":
    main()
