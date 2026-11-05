import pickle
import os
import multiprocessing
import functools
import torch
import numpy as np
import uaibot as ub
import uaibot_cpp_bind as ub_cpp
from tqdm import tqdm
from scipy.spatial.transform import Rotation as R
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from torch_vf import SE3VectorField, ec_distance_and_pos_ori
from baseline_adriano import (
    se3_curve_to_r3t3,
    rezende_vf_r3t3,
    rezende_distance_gradient,
    vf_to_fossen_twist,
)

mp_ctx = multiprocessing.get_context("spawn")
curve_base_path = "./data"
results_dir = curve_base_path
curve_types = ["ellipse", "trefoil", "vertical_coil"]

# ----------------------- FIXED PARAMETERS -----------------------------
gains = 6 * np.array([2.0, 2.0, 2.0, 4.0, 4.0, 4.0])
dt = 0.01
T = 300
scale_pos = 1 / np.sqrt(3.0)

kn1 = 1
kn2 = 1.0
kt1 = 0.03 * 0.8
kt2 = 1.0
kt3 = kn2

N_CURVES = 20
N_INITS = 5
PROBLEMATIC_INIT_SEED = 0
n_samples = N_CURVES * N_INITS

RUN_BASELINE = True  # whether to run Lie-VF or Rezende VF
# "both" (default), "nominal" (skip problematic), or "problematic" (only problematic)
RUN_MODE = "nominal"
# Rejection-sampling effort for problematic seeds
N_PROBLEMATIC_TRIES = 50000

# Torch configuration used inside each worker.
# CPU + float64 keeps things consistent with your numpy path and avoids
# 100 processes fighting over one GPU.
# ---- Torch device / dtype (float64 to match numpy) -----------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if RUN_BASELINE:
    DEVICE = torch.device("cpu")
DTYPE = torch.float64

# Instantiate the solver once (delta / ds match the C++ constants c_delta / c_ds)
vf_solver = SE3VectorField(delta=1e-3, ds=1e-3)


def get_params(noise_seed: int = 42):
    """Creates a fresh FossenParameters object to avoid multiprocessing pickling issues."""
    params = ub_cpp.FossenParameters()
    params.dt = dt
    params.velocity_gains = gains
    params.model_compensation = False
    params.thruster_time_constant = 0
    params.thrust_max = 5.25 * 9.80665 * np.ones((8,))
    params.thrust_min = -4.10 * 9.80665 * np.ones((8,))

    # Measurement Noise Configuration (converted from C++ to NumPy arrays)
    params.measurement_noise.enabled = True
    params.measurement_noise.seed = int(noise_seed)
    params.measurement_noise.sample_period = 0  # 1 / 30

    params.measurement_noise.white_variances = np.array(
        [
            # Position (x, y, z) [m]
            0.005**2,
            0.005**2,
            0.002**2,
            # Orientation (roll, pitch, yaw) [rad]
            0.002**2,
            0.002**2,
            0.002**2,
            # Linear velocity (u, v, w) [m/s]
            0.002**2,
            0.002**2,
            0.002**2,
            # Angular velocity (p, q, r) [rad/s]
            0.003**2,
            0.003**2,
            0.003**2,
        ],
        dtype=np.float64,
    )

    params.measurement_noise.colored_variances = np.array(
        [
            0.01**2,
            0.01**2,
            0.01**2,  # Position: ~1 cm drift
            0.005**2,
            0.005**2,
            0.005**2,
            0.005**2,
            0.005**2,
            0.005**2,
            0.01**2,
            0.01**2,
            0.01**2,
        ],
        dtype=np.float64,
    )

    # params.measurement_noise.correlation_times = 0.2 * np.ones((12,))
    params.measurement_noise.correlation_times = np.array(
        [10.0, 10.0, 10.0, 1.0, 1.0, 2.0, 0.8, 0.8, 0.8, 0.3, 0.3, 0.3],
        dtype=np.float64,
    )

    return params


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


def get_LvDhat(htm_t, curve_t, vf_solver):
    """L-operator via torch VF: kt1=0 kills tangent, kn2=1e9 saturates tanh."""
    twist_t, _, _ = vf_solver.vectorfield_se3(
        htm_t.unsqueeze(0),  # (1, 4, 4)
        curve_t,  # (N, 4, 4)
        kt1=0.0,
        kt2=1.0,
        kt3=1.0,
        kn1=1.0,
        kn2=1e9,
    )
    return twist_t.squeeze(0)  # (6,)


def score_p1(htm_t, curve_t, vf_solver, min_idx_diff_k=5, eps=1e-12):
    """HIGH = ambiguous (a distant sample is nearly as close as the closest)."""
    N = curve_t.shape[0]
    with torch.no_grad():
        # (N,) SE(3) distance from htm_t to each curve sample.
        dists = vf_solver.eedistance(htm_t.unsqueeze(0), curve_t)

        u = int(torch.argmin(dists).item())
        D_min = dists[u]

        # Circular index distances from the argmin.
        idx = torch.arange(N, device=DEVICE)
        idx_diff = (idx - u).abs()
        min_idx_diff_v = torch.minimum(idx_diff, N - idx_diff).to(dists.dtype)

        # Only consider points with circular index-distance > min_idx_diff_k.
        valid = min_idx_diff_v > float(min_idx_diff_k)
        r_v = (dists - D_min).abs() / min_idx_diff_v.clamp_min(1.0)
        r_v = torch.where(valid, r_v, torch.full_like(r_v, float("inf")))

        best_r = float(r_v.min().item())

    return 1.0 / (best_r + eps)


def score_p2(htm_t, curve_t, vf_solver, eps=5e-3):
    """Hessian of the L-operator, batched over the 6 Lie-algebra directions."""
    with torch.no_grad():
        base_L = get_LvDhat(htm_t, curve_t, vf_solver)  # (6,)

        I6 = torch.eye(6, dtype=DTYPE, device=DEVICE)
        basis = vf_solver.smap_se3(I6)  # (6, 4, 4)
        # exp(eps * basis_i) @ htm  for i = 0..5   →   (6, 4, 4)
        variations = vf_solver.exp_se3(basis * eps) @ htm_t.unsqueeze(0)

        # Single batched call for all 6 perturbed states → (6, 6).
        Ls, _, _ = vf_solver.vectorfield_se3(
            variations,
            curve_t,
            kt1=0.0,
            kt2=1.0,
            kt3=1.0,
            kn1=1.0,
            kn2=1e9,
        )

        J = (Ls - base_L.unsqueeze(0)) / eps  # (6, 6)
        return float(torch.linalg.norm(J, "fro").item())


def _sample_random_init(curve, rng):
    htm = curve[0].copy()
    offset_dir = rng.standard_normal(3)
    offset_dir /= np.linalg.norm(offset_dir)
    htm[:3, 3] += offset_dir * rng.uniform(0.5, 2.0)
    R_err = R.from_euler("xyz", rng.uniform(-np.pi / 6, np.pi / 6, 3)).as_matrix()
    htm[:3, :3] = htm[:3, :3] @ R_err
    return htm


def sample_init(curve, rng, kind="nominal"):
    """'P1'/'P2' use rejection sampling; 'nominal' samples uniformly.

    Note: `curve` is the *unscaled* curve (same as before); scoring stays
    in the unscaled space. Torch tensors are built once per call, not per try.
    """
    if kind == "nominal":
        return _sample_random_init(curve, rng), np.nan

    vf_solver = SE3VectorField()
    curve_t = torch.as_tensor(curve, dtype=DTYPE, device=DEVICE)

    best_htm, best_score = None, -np.inf
    for _ in range(N_PROBLEMATIC_TRIES):
        htm = _sample_random_init(curve, rng)
        htm_t = torch.as_tensor(htm, dtype=DTYPE, device=DEVICE)

        if kind == "P1":
            s = score_p1(htm_t, curve_t, vf_solver)
        elif kind == "P2":
            s = score_p2(htm_t, curve_t, vf_solver)
        else:
            raise ValueError(f"unknown init kind: {kind}")

        if s > best_score:
            best_score, best_htm = s, htm
    return best_htm, best_score


# ---------------------------------------------------------------------------
# Main worker: one physics + torch-vector-field run
# ---------------------------------------------------------------------------
def run_single_sim(curve_type, i, curve_base, curve_derivative_base):
    """Function executed by each parallel worker for a single run."""
    # Keep torch quiet / single-threaded per worker.
    torch.set_num_threads(1)
    torch.set_grad_enabled(False)

    noise_seed = int((i * 7919 + 13) % (2**31 - 1))
    params = get_params(noise_seed=noise_seed)
    CURVE_POINTS = len(curve_base)

    curve_seed = i // N_INITS
    init_seed = i % N_INITS
    if init_seed != PROBLEMATIC_INIT_SEED:
        init_type = "nominal"
    else:
        init_type = "P1" if curve_seed % 2 == 0 else "P2"

    # ---- Random rigid transform of the curve (same as before) -----------
    curve, curve_derivative = rand_se3_curve(
        curve_base, curve_derivative_base, seed=curve_seed
    )
    curve_scale = curve.copy()
    curve_scale[:, :3, 3] *= scale_pos

    local_rng = np.random.default_rng(i + 1000)
    htm0, target_score = sample_init(curve, local_rng, kind=init_type)

    actual_twist = np.zeros(6)

    state = ub_cpp.FossenState()
    state.pose = htm0.copy()
    state.twist = actual_twist.copy()
    state.measurement.pose = htm0.copy()
    state.measurement.twist = actual_twist.copy()
    state.measurement.valid = True

    # ---- Torch setup (once per worker) ---------------------------------
    vf_solver = SE3VectorField()
    curve_scale_t = torch.as_tensor(curve_scale, dtype=DTYPE, device=DEVICE)
    deriv_t = torch.as_tensor(curve_derivative, dtype=DTYPE, device=DEVICE)

    time_vec = np.arange(0, T, dt)

    # Pre-allocate logs
    log_dist, log_idx = [], []
    log_twist, log_xi, log_htm = [], [], []
    log_requested_wrench, log_allocated_wrench = [], []
    log_applied_wrench, log_allocation_residual = [], []
    log_thrust_setpoints, log_saturated_thrusters = [], []
    log_meas_htm, log_meas_twist = [], []

    prev_idx = 0
    window_size = int(0.1 * CURVE_POINTS)
    dist = 1e3
    print(f"begin {i}", end="; ", flush=True)

    for t in time_vec:
        half_win = window_size // 2
        if dist < 0.5:
            start = (prev_idx - half_win) % CURVE_POINTS
            end = (prev_idx + half_win) % CURVE_POINTS
        else:
            start = 0
            end = CURVE_POINTS

        # 1. CONTROLLER FEEDBACK (noisy measurement)
        ctrl_pose = state.measurement.pose if state.measurement.valid else state.pose
        ctrl_twist = state.measurement.twist if state.measurement.valid else state.twist

        scale_pose = ctrl_pose.copy()
        scale_pose[:3, 3] *= scale_pos

        # ---- Windowed curve on device ---------------------------------
        if start < end:
            curve_window_t = curve_scale_t[start:end]
            deriv_window_t = deriv_t[start:end]
        else:
            curve_window_t = torch.cat(
                (curve_scale_t[start:], curve_scale_t[:end]), dim=0
            )
            deriv_window_t = torch.cat((deriv_t[start:], deriv_t[:end]), dim=0)

        # ---- Single batched torch vector-field call -------------------
        scale_pose_t = torch.as_tensor(
            scale_pose, dtype=DTYPE, device=DEVICE
        ).unsqueeze(0)

        twist_t, dist_t, idx_t = vf_solver.vectorfield_se3(
            states=scale_pose_t,
            curve=curve_window_t,
            kt1=kt1,
            kt2=kt2,
            kt3=kt3,
            kn1=kn1,
            kn2=kn2,
            curve_derivative=deriv_window_t,
        )

        xi = twist_t.squeeze(0).cpu().numpy()  # (6,) v || ω
        dist = float(dist_t.item())
        local_idx = int(idx_t.item())

        global_idx = (start + local_idx) % CURVE_POINTS
        prev_idx = global_idx
        closest_idx = global_idx

        v, omega = xi[:3], xi[3:]
        p_ctrl = ctrl_pose[:3, 3]

        target_twist = np.empty(6)
        target_twist[:3] = v + np.cross(omega, p_ctrl)
        target_twist[3:] = omega

        # 2. PHYSICS STEP
        state = ub_cpp.fossen_step(target_twist, state, params)
        diagnostics = state.diagnostics

        # 3. LOGGING (uses ground truth)
        true_pose = state.pose
        true_twist = state.twist
        rot = true_pose[:3, :3].T

        log_dist.append(dist)
        log_idx.append(closest_idx)
        # matches the convention used in your test file
        log_xi.append(target_twist.copy())
        log_twist.append(true_twist.copy())
        log_htm.append(true_pose.copy())

        log_requested_wrench.append(diagnostics.requested_wrench)
        log_allocated_wrench.append(diagnostics.allocated_wrench)
        log_applied_wrench.append(diagnostics.applied_wrench)
        log_allocation_residual.append(diagnostics.allocation_residual)
        log_thrust_setpoints.append(diagnostics.thrust_setpoints)
        log_saturated_thrusters.append(diagnostics.saturated_thrusters)
        log_meas_htm.append(state.measurement.pose.copy())
        log_meas_twist.append(state.measurement.twist.copy())

    print(f"end {i}", end="$$ ", flush=True)

    # ------------------------------------------------------------------
    # Post-hoc true-pose metrics on the FULL curve (batched, on device).
    # ------------------------------------------------------------------
    htm_batch = torch.as_tensor(np.array(log_htm), dtype=DTYPE, device=DEVICE)
    htm_batch_scl = htm_batch.clone()
    htm_batch_scl[..., :3, 3] *= scale_pos

    curve_full_t = torch.as_tensor(curve, dtype=DTYPE, device=DEVICE)  # unscaled
    curve_full_scl_t = torch.as_tensor(
        curve_scale, dtype=DTYPE, device=DEVICE
    )  # scaled

    dists_true, idx_true, pos_err_true_cm, ori_err_true_deg = ec_distance_and_pos_ori(
        vf_solver,
        states_scaled=htm_batch_scl,
        curve_scaled=curve_full_scl_t,
        states_unscaled=htm_batch,
        curve_unscaled=curve_full_t,
        chunk_size=None,  # auto-chunk (~5e6 pairs)
    )

    return {
        "time": np.array(time_vec),
        "dist_noisy": np.array(log_dist),  # in-loop (windowed, noisy)
        "dist": dists_true.cpu().numpy(),  # post-hoc (full curve)
        "err_pos": pos_err_true_cm.cpu().numpy(),  # post-hoc true-pose
        "err_ori": ori_err_true_deg.cpu().numpy(),  # post-hoc true-pose
        "curve": curve,
        "curve_derivative": curve_derivative,
        "idx": np.array(log_idx),
        "idx_true": idx_true.cpu().numpy(),
        "xi": np.array(log_xi),
        "twist": np.array(log_twist),
        "htm": np.array(log_htm),
        "requested_wrench": np.array(log_requested_wrench),
        "allocated_wrench": np.array(log_allocated_wrench),
        "applied_wrench": np.array(log_applied_wrench),
        "allocation_residual": np.array(log_allocation_residual),
        "thrust_setpoints": np.array(log_thrust_setpoints),
        "saturated_thrusters": np.array(log_saturated_thrusters),
        "noise_seed": noise_seed,
        # metadata
        "meta_curve_seed": curve_seed,
        "meta_init_seed": init_seed,
        "meta_run_id": i,
        "meta_init_rng_seed": i + 1000,
        "meta_init_type": init_type,
        "meta_target_score": target_score,
        "meta_htm0": htm0,
        "meta_config": {
            "gains": gains,
            "dt": dt,
            "T": T,
            "scale_pos": scale_pos,
            "kt1": kt1,
            "kt2": kt2,
            "kt3": kt3,
            "kn1": kn1,
            "kn2": kn2,
            "vf_solver": {
                "delta": vf_solver.delta,
                "ds": vf_solver.ds,
                "max_cos_theta": vf_solver.max_cos_theta,
                "theta_zero": vf_solver.theta_zero,
            },
            "thrust_max": params.thrust_max,
            "thrust_min": params.thrust_min,
            "noise_white_var": params.measurement_noise.white_variances,
            "noise_colored_var": params.measurement_noise.colored_variances,
            "noise_corr_times": params.measurement_noise.correlation_times,
            "curve_type": curve_type,
        },
    }


def run_single_baseline(curve_type, i, curve_base, curve_derivative_base):
    """Rezende baseline on R^3 x T^3. CPU-only, no torch VF in the loop."""
    # One thread per worker — critical under ProcessPoolExecutor.
    torch.set_num_threads(1)
    torch.set_grad_enabled(False)

    noise_seed = int((i * 7919 + 13) % (2**31 - 1))
    params = get_params(noise_seed=noise_seed)

    curve_seed = i // N_INITS
    init_seed = i % N_INITS
    if init_seed != PROBLEMATIC_INIT_SEED:
        init_type = "nominal"
    else:
        init_type = "P1" if curve_seed % 2 == 0 else "P2"

    curve, curve_derivative = rand_se3_curve(
        curve_base, curve_derivative_base, seed=curve_seed
    )
    curve_scale = curve.copy()
    curve_scale[:, :3, 3] *= scale_pos

    local_rng = np.random.default_rng(i + 1000)
    htm0, target_score = sample_init(curve, local_rng, kind=init_type)

    state = ub_cpp.FossenState()
    state.pose = htm0.copy()
    state.twist = np.zeros(6)
    state.measurement.pose = htm0.copy()
    state.measurement.twist = np.zeros(6)
    state.measurement.valid = True

    # ---- Rezende curve representation (once) ---------------------------
    # Test-script convention: build from the UNSCALED curve; scale_pos is
    # applied to both p_ctrl and p_curve at VF-call time.
    p_curve, phi_curve = se3_curve_to_r3t3(curve)
    p_curve_scl = p_curve * scale_pos  # hoisted out of loop

    time_vec = np.arange(0, T, dt)

    # ---- Logs (same names as run_single_sim where it makes sense) ------
    log_dist_rez, log_dist_ctrl_rez, log_idx = [], [], []
    log_twist, log_xi, log_htm = [], [], []
    log_requested_wrench, log_allocated_wrench = [], []
    log_applied_wrench, log_allocation_residual = [], []
    log_thrust_setpoints, log_saturated_thrusters = [], []
    log_meas_htm, log_meas_twist = [], []

    print(f"begin {i}", end="; ", flush=True)

    for t in time_vec:
        # 1. Controller feedback (noisy measurement)
        ctrl_pose = state.measurement.pose if state.measurement.valid else state.pose
        p_ctrl = ctrl_pose[:3, 3]
        phi_ctrl = R.from_matrix(ctrl_pose[:3, :3]).as_euler("xyz")

        psi = rezende_vf_r3t3(
            p_ctrl * scale_pos,
            phi_ctrl,
            p_curve_scl,
            phi_curve,
            v_r=1.0,
            kn1=kn1,
            kn2=kn2,
            kt1=kt1 * 10,
            kt2=kt2,
            kt3=kt3,
        )
        target_twist = vf_to_fossen_twist(psi, phi_ctrl)

        # 2. Physics
        state = ub_cpp.fossen_step(target_twist, state, params)
        diagnostics = state.diagnostics

        # 3. In-loop logging ---------------------------------------------
        true_pose = state.pose
        true_twist = state.twist
        p_true = true_pose[:3, 3]
        phi_true = R.from_matrix(true_pose[:3, :3]).as_euler("xyz")

        # True-pose Rezende metric (matches test script)
        dist_true_rez, closest_idx, _ = rezende_distance_gradient(
            p_true, phi_true, p_curve, phi_curve
        )
        # Perceived Rezende metric (noisy ctrl pose) — analogous to
        # run_single_sim's `dist_noisy` (extra full search per step; drop
        # this call if you don't need the perceived-vs-true split).
        dist_ctrl_rez, _, _ = rezende_distance_gradient(
            p_ctrl, phi_ctrl, p_curve, phi_curve
        )

        log_dist_rez.append(float(dist_true_rez))
        log_dist_ctrl_rez.append(float(dist_ctrl_rez))
        log_idx.append(int(closest_idx))
        log_xi.append(target_twist.copy())
        log_twist.append(true_twist.copy())
        log_htm.append(true_pose.copy())
        log_requested_wrench.append(diagnostics.requested_wrench)
        log_allocated_wrench.append(diagnostics.allocated_wrench)
        log_applied_wrench.append(diagnostics.applied_wrench)
        log_allocation_residual.append(diagnostics.allocation_residual)
        log_thrust_setpoints.append(diagnostics.thrust_setpoints)
        log_saturated_thrusters.append(diagnostics.saturated_thrusters)
        log_meas_htm.append(state.measurement.pose.copy())
        log_meas_twist.append(state.measurement.twist.copy())

    print(f"end {i}", end="$$ ", flush=True)

    # ------------------------------------------------------------------
    # Post-hoc pos/ori errors (NumPy-only, reusing Rezende closest_idx)
    # ------------------------------------------------------------------
    htm_arr = np.array(log_htm)  # (T, 4, 4)
    idx_arr = np.array(log_idx)  # (T,)
    closest_arr = curve[idx_arr]  # (T, 4, 4) unscaled

    p_true_all = htm_arr[:, :3, 3]
    R_true_all = htm_arr[:, :3, :3]
    p_near_all = closest_arr[:, :3, 3]
    R_near_all = closest_arr[:, :3, :3]

    err_pos_cm = np.linalg.norm(p_near_all - p_true_all, axis=1) * 100.0
    trace = (R_near_all * R_true_all).sum(axis=(1, 2))
    cos_ang = np.clip((trace - 1.0) / 2.0, -1.0, 1.0)
    err_ori_deg = np.degrees(np.arccos(cos_ang))

    return {
        "time": np.array(time_vec),
        # distance: perceived (noisy) and true-pose Rezende
        "dist_noisy": np.array(log_dist_ctrl_rez),  # Rezende on ctrl_pose
        "dist": np.array(log_dist_rez),  # Rezende on true pose
        # pos/ori errors (unscaled, closest point from Rezende)
        "err_pos": err_pos_cm,
        "err_ori": err_ori_deg,
        # inputs / traces
        "curve": curve,
        "curve_derivative": curve_derivative,
        "idx": idx_arr,
        "idx_true": idx_arr,  # same source as err_*
        "xi": np.array(log_xi),
        "twist": np.array(log_twist),
        "htm": htm_arr,
        "meas_htm": np.array(log_meas_htm),
        "meas_twist": np.array(log_meas_twist),
        "requested_wrench": np.array(log_requested_wrench),
        "allocated_wrench": np.array(log_allocated_wrench),
        "applied_wrench": np.array(log_applied_wrench),
        "allocation_residual": np.array(log_allocation_residual),
        "thrust_setpoints": np.array(log_thrust_setpoints),
        "saturated_thrusters": np.array(log_saturated_thrusters),
        "noise_seed": noise_seed,
        # metadata
        "meta_curve_seed": curve_seed,
        "meta_init_seed": init_seed,
        "meta_run_id": i,
        "meta_init_rng_seed": i + 1000,
        "meta_init_type": init_type,
        "meta_target_score": target_score,
        "meta_htm0": htm0,
        "meta_baseline": "rezende_r3t3",
        "meta_config": {
            "gains": gains,
            "dt": dt,
            "T": T,
            "scale_pos": scale_pos,
            "kt1": kt1,
            "kt2": kt2,
            "kt3": kt3,
            "kn1": kn1,
            "kn2": kn2,
            "kt1_effective": kt1 * 10,  # baseline uses 10× kt1
            "v_r": 1.0,
            "thrust_max": params.thrust_max,
            "thrust_min": params.thrust_min,
            "noise_white_var": params.measurement_noise.white_variances,
            "noise_colored_var": params.measurement_noise.colored_variances,
            "noise_corr_times": params.measurement_noise.correlation_times,
            "curve_type": curve_type,
        },
    }


def _save_merged(results_dir, curve_type, all_run_results, baseline=False):
    """Merge this batch into {curve_type}_results.pkl atomically."""
    save_dict = {}
    for run_id, run_data in all_run_results:
        for key, value in run_data.items():
            save_dict[f"run_{run_id:03d}_{key}"] = value

    if baseline:
        save_path = os.path.join(results_dir, f"{curve_type}_baseline_results.pkl")
    else:
        save_path = os.path.join(results_dir, f"{curve_type}_results.pkl")
    if os.path.exists(save_path):
        with open(save_path, "rb") as f:
            existing = pickle.load(f)
        existing.update(save_dict)
        save_dict = existing

    tmp_path = save_path + ".tmp"
    with open(tmp_path, "wb") as f:
        pickle.dump(save_dict, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp_path, save_path)  # atomic on POSIX
    return save_path


if __name__ == "__main__":
    os.makedirs(results_dir, exist_ok=True)

    SAVE_EVERY_K = 10  # checkpoint every K completed runs
    is_cuda = DEVICE.type == "cuda"
    tag = "baseline" if RUN_BASELINE else "lie"
    worker_fn = run_single_baseline if RUN_BASELINE else run_single_sim

    if is_cuda:
        print(f"[{tag}] Device: {DEVICE}, dtype: {DTYPE}  ->  sequential")
    else:
        max_cores = multiprocessing.cpu_count()
        mp_ctx = multiprocessing.get_context("spawn")
        print(
            f"[{tag}] Device: {DEVICE}, dtype: {DTYPE}  ->  parallel "
            f"({max_cores} cores, spawn)"
        )

    for curve_type in curve_types:
        print(f"\n================ Processing curve: {curve_type} ================")
        curve_path = os.path.join(curve_base_path, f"{curve_type}.npy")
        dcurve_path = os.path.join(curve_base_path, f"{curve_type}_derivative.npy")
        curve_base = np.load(curve_path, allow_pickle=True)
        curve_derivative_base = np.load(dcurve_path, allow_pickle=True)

        if RUN_MODE == "nominal":
            run_ids = [
                i for i in range(n_samples) if i % N_INITS != PROBLEMATIC_INIT_SEED
            ]
        elif RUN_MODE == "problematic":
            run_ids = [
                i for i in range(n_samples) if i % N_INITS == PROBLEMATIC_INIT_SEED
            ]
        else:
            run_ids = list(range(n_samples))

        n_total = len(run_ids)
        all_run_results = []
        print(
            f"[{curve_type}] START  {datetime.now().strftime('%H:%M:%S')}  "
            f"(mode={RUN_MODE}, {n_total} runs)",
            flush=True,
        )

        if is_cuda:
            # ---------------- SEQUENTIAL (single GPU) ----------------
            for k, run_id in enumerate(run_ids, start=1):
                res = run_single_sim(
                    curve_type, run_id, curve_base, curve_derivative_base
                )
                all_run_results.append((run_id, res))
                print(
                    f"[{curve_type}] {datetime.now().strftime('%H:%M:%S')}  "
                    f"run_id={run_id}  ({k}/{n_total})",
                    flush=True,
                )
                if (k % SAVE_EVERY_K == 0) or (k == n_total):
                    path = _save_merged(
                        results_dir, curve_type, all_run_results, baseline=RUN_BASELINE
                    )
                    print(f"  -> checkpoint saved: {path}", flush=True)

        else:
            # ---------------- PARALLEL (CPU) ------------------------
            worker_func = functools.partial(
                # run_single_sim,
                worker_fn,
                curve_type,
                curve_base=curve_base,
                curve_derivative_base=curve_derivative_base,
            )
            with ProcessPoolExecutor(
                max_workers=max_cores, mp_context=mp_ctx
            ) as executor:
                for k, (run_id, res) in enumerate(
                    zip(run_ids, executor.map(worker_func, run_ids)), start=1
                ):
                    all_run_results.append((run_id, res))
                    print(
                        f"[{curve_type}] {datetime.now().strftime('%H:%M:%S')}  "
                        f"run_id={run_id}  ({k}/{n_total})",
                        flush=True,
                    )
                    if (k % SAVE_EVERY_K == 0) or (k == n_total):
                        path = _save_merged(
                            results_dir,
                            curve_type,
                            all_run_results,
                            baseline=RUN_BASELINE,
                        )
                        print(f"  -> checkpoint saved: {path}", flush=True)

        print(
            f"[{curve_type}] END    {datetime.now().strftime('%H:%M:%S')}", flush=True
        )

    print("\nAll simulations complete.")
