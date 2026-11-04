import pickle
import os
import numpy as np
import uaibot as ub
import uaibot_cpp_bind as ub_cpp
from tqdm import tqdm
from scipy.spatial.transform import Rotation as R
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
import multiprocessing
import functools

curve_base_path = "./data"
curve_types = ["ellipse", "trefoil", "vertical_coil"]

# ----------------------- FIXED PARAMETERS -----------------------------
gains = 6 * np.array([2.0, 2.0, 2.0, 4.0, 4.0, 4.0])
dt = 0.01
T = 300
scale_pos = 1 / np.sqrt(3)

kn1 = 1
kn2 = 1.0
kt1 = 0.03 / 2
kt2 = 1.0
kt3 = kn2

# --- Run structure: 20 curves × 5 inits = 100 ---
N_CURVES = 20
N_INITS = 5
PROBLEMATIC_INIT_SEED = 0  # init_seed==0 is the problematic one
n_samples = N_CURVES * N_INITS  # 100

# "both" (default), "nominal" (skip problematic), or "problematic" (only problematic)
RUN_MODE = "nominal"

# Rejection-sampling effort for problematic seeds
N_PROBLEMATIC_TRIES = 100


def get_params():
    """Creates a fresh FossenParameters object to avoid multiprocessing pickling issues."""
    params = ub_cpp.FossenParameters()
    params.dt = dt
    params.velocity_gains = gains
    params.model_compensation = False
    params.thruster_time_constant = 0
    params.thrust_max = 5.25 * 9.80665 * np.ones((8,))
    params.thrust_min = -4.10 * 9.80665 * np.ones((8,))

    params.measurement_noise.enabled = True
    params.measurement_noise.seed = 42
    params.measurement_noise.sample_period = 0.01 * 0

    params.measurement_noise.white_variances = np.zeros((12,))
    params.measurement_noise.colored_variances = np.array(
        [
            0.05**2,
            0.05**2,
            0.05**2,
            0.005**2,
            0.005**2,
            0.005**2,
            0.005**2,
            0.005**2,
            0.005**2,
            0.001**2,
            0.001**2,
            0.001**2,
        ],
        dtype=np.float64,
    )
    params.measurement_noise.correlation_times = np.array(
        [10.0, 10.0, 10.0, 1.0, 1.0, 5.0, 0.2, 0.2, 0.2, 30.0, 30.0, 30.0],
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


def get_LvDhat(htm, curve, ds=1.0 / 30000):
    """Extract the C++ L-operator via the VF: kt1=0 kills tangent, kn2=1e9 saturates tanh."""
    xi, _, _ = ub.Robot.vector_field_SE3(
        htm,
        curve,
        kt1=0.0,
        kt2=1.0,
        kt3=1.0,
        kn1=1.0,
        kn2=1e9,
        # ds=ds,
    )
    return np.asarray(xi).reshape(-1)


def score_p1(htm, curve, min_idx_diff_k=5, eps=1e-12):
    """Ambiguity score: HIGH when a distant sample has nearly equal distance.

    For each v with circular index-distance > min_idx_diff_k, compute
        r_v = |D_min - D_v| / min_idx_diff_v
    (rate of distance growth per unit arc-length).
    A flat landscape (small r_v) means multiple near-minima -> ambiguous.
    Return 1 / (min_v r_v + eps), so higher = more ambiguous.
    """
    p = htm[:3, 3]
    curve_pos = curve[:, :3, 3]
    N = len(curve)
    dists = np.linalg.norm(curve_pos - p, axis=1)
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


def score_p2(htm, curve, eps=5e-3):
    """Hessian of the L-operator, via the C++ VF (no need to rebind)."""
    base_L = get_LvDhat(htm, curve)
    J = np.zeros((6, 6))
    for i in range(6):
        xi = np.zeros(6)
        xi[i] = eps
        variation = ub_cpp.expSE3(ub_cpp.SmapSE3(xi)) @ htm
        L_i = get_LvDhat(variation, curve)
        J[:, i] = (L_i - base_L) / eps
    return float(np.linalg.norm(J, "fro"))


def _sample_random_init(curve, rng):
    htm = curve[0].copy()
    offset_dir = rng.standard_normal(3)
    offset_dir /= np.linalg.norm(offset_dir)
    htm[:3, 3] += offset_dir * rng.uniform(0.5, 2.0)
    R_err = R.from_euler("xyz", rng.uniform(-np.pi / 6, np.pi / 6, 3)).as_matrix()
    htm[:3, :3] = htm[:3, :3] @ R_err
    return htm


def sample_init(curve, rng, kind="nominal"):
    """'P1'/'P2' use rejection sampling; 'nominal' samples uniformly."""
    if kind == "nominal":
        return _sample_random_init(curve, rng), np.nan

    best_htm, best_score = None, -np.inf
    for _ in range(N_PROBLEMATIC_TRIES):
        htm = _sample_random_init(curve, rng)
        s = score_p1(htm, curve) if kind == "P1" else score_p2(htm, curve)
        if s > best_score:
            best_score, best_htm = s, htm
    return best_htm, best_score


def run_single_sim(curve_type, i, curve_base, curve_derivative_base):
    """Function executed by each parallel worker for a single run."""
    # Initialize parameters inside the worker to ensure safety
    # params = get_params()
    # CURVE_POINTS = len(curve_base)
    #
    # curve, curve_derivative = rand_se3_curve(curve_base, curve_derivative_base, seed=i)
    # curve_scale = curve.copy()
    # curve_scale[:, :3, 3] *= scale_pos
    #
    # htm0 = curve[0].copy()
    # # Use a local RNG per worker to avoid seed collisions
    # local_rng = np.random.default_rng(i + 1000)
    #
    # offset_dir = local_rng.standard_normal(3)
    # offset_dir /= np.linalg.norm(offset_dir)
    # htm0[:3, 3] += offset_dir * local_rng.uniform(0.5, 2.0)
    #
    # R_err = R.from_euler("xyz", local_rng.uniform(-np.pi / 6, np.pi / 6, 3)).as_matrix()
    # htm0[:3, :3] = htm0[:3, :3] @ R_err
    #
    # actual_twist = np.zeros(6)
    params = get_params()
    CURVE_POINTS = len(curve_base)

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

    actual_twist = np.zeros(6)

    state = ub_cpp.FossenState()
    state.pose = htm0.copy()
    state.twist = actual_twist.copy()
    state.measurement.pose = htm0.copy()
    state.measurement.twist = actual_twist.copy()
    state.measurement.valid = True

    time_vec = np.arange(0, T, dt)

    # Pre-allocate lists
    log_dist, log_err_pos, log_err_ori, log_idx = [], [], [], []
    log_twist, log_xi, log_htm = [], [], []
    log_requested_wrench, log_allocated_wrench = [], []
    log_applied_wrench, log_allocation_residual = [], []
    log_thrust_setpoints, log_saturated_thrusters = [], []

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

        if start < end:
            curve_window = curve_scale[start:end]
            deriv_window = curve_derivative[start:end]
        else:
            curve_window = np.vstack((curve_scale[start:], curve_scale[:end]))
            deriv_window = np.vstack((curve_derivative[start:], curve_derivative[:end]))

        ctrl_pose = state.measurement.pose if state.measurement.valid else state.pose
        ctrl_twist = state.measurement.twist if state.measurement.valid else state.twist

        scale_pose = ctrl_pose.copy()
        scale_pose[:3, 3] *= scale_pos

        xi, _, local_idx = ub.Robot.vector_field_SE3(
            scale_pose,
            curve_window,
            kt1=kt1,
            kt2=kt2,
            kt3=kt3,
            kn1=kn1,
            kn2=kn2,
            curve_derivative=deriv_window,
        )

        global_idx = (start + local_idx) % CURVE_POINTS
        prev_idx = global_idx

        v, omega = xi[:3], xi[3:]
        p_ctrl = ctrl_pose[:3, 3]

        target_twist = np.empty(6)
        target_twist[:3] = v + np.cross(omega, p_ctrl)
        target_twist[3:] = omega

        state = ub_cpp.fossen_step(target_twist, state, params)
        diagnostics = state.diagnostics

        true_pose = state.pose
        true_twist = state.twist
        rot = true_pose[:3, :3].T

        true_pose_scale = true_pose.copy()
        true_pose_scale[:3, 3] *= scale_pos

        _, dist, closest_local_idx = ub.Robot.vector_field_SE3(
            true_pose_scale,
            curve_window,
            kt1=kt1,
            kt2=kt2,
            kt3=kt3,
            kn1=kn1,
            kn2=kn2,
            curve_derivative=curve_derivative,
        )

        closest_idx = (start + closest_local_idx) % CURVE_POINTS
        # closest_point = curve[closest_idx]
        closest_point = curve_window[closest_local_idx]
        p_near = closest_point[:3, 3]
        ori_near = closest_point[:3, :3]
        p_curr = true_pose_scale[:3, 3]
        ori_curr = true_pose_scale[:3, :3]

        trace_ = np.trace(ori_near @ np.linalg.inv(ori_curr))
        acos = np.arccos(np.clip((trace_ - 1) / 2, -1.0, 1.0))
        if np.isnan(acos):
            acos = 0

        log_dist.append(dist)
        log_idx.append(closest_idx)
        log_xi.append(xi.copy())
        log_twist.append((rot @ true_twist[3:].reshape(-1, 1)).ravel())
        log_htm.append(true_pose.copy())
        log_err_pos.append(np.linalg.norm(p_near - p_curr) * 100)
        log_err_ori.append(acos * 180 / np.pi)

        log_requested_wrench.append(diagnostics.requested_wrench)
        log_allocated_wrench.append(diagnostics.allocated_wrench)
        log_applied_wrench.append(diagnostics.applied_wrench)
        log_allocation_residual.append(diagnostics.allocation_residual)
        log_thrust_setpoints.append(diagnostics.thrust_setpoints)
        log_saturated_thrusters.append(diagnostics.saturated_thrusters)

    print(f"end {i}", end="$$ ", flush=True)

    return {
        "time": time_vec,
        "dist": np.array(log_dist),
        "err_pos": np.array(log_err_pos),
        "err_ori": np.array(log_err_ori),
        "idx": np.array(log_idx),
        "xi": np.array(log_xi),
        "twist": np.array(log_twist),
        "htm": np.array(log_htm),
        "requested_wrench": np.array(log_requested_wrench),
        "allocated_wrench": np.array(log_allocated_wrench),
        "applied_wrench": np.array(log_applied_wrench),
        "allocation_residual": np.array(log_allocation_residual),
        "thrust_setpoints": np.array(log_thrust_setpoints),
        "saturated_thrusters": np.array(log_saturated_thrusters),
        # --- add these lines to the returned dict ---
        "meta_curve_seed": curve_seed,
        "meta_init_seed": init_seed,
        "meta_init_type": init_type,
        "meta_target_score": target_score,
        "meta_htm0": htm0,
    }


if __name__ == "__main__":
    # Required for multiprocessing on some platforms
    results_dir = curve_base_path
    os.makedirs(results_dir, exist_ok=True)

    max_cores = multiprocessing.cpu_count()
    print(f"Starting parallel execution using {max_cores} cores.")

    for curve_type in curve_types:
        print(f"\n================ Processing curve: {curve_type} ================")
        curve_path = os.path.join(curve_base_path, f"{curve_type}.npy")
        dcurve_path = os.path.join(curve_base_path, f"{curve_type}_derivative.npy")
        curve_base = np.load(curve_path, allow_pickle=True)
        curve_derivative_base = np.load(dcurve_path, allow_pickle=True)

        # Create a partial function that locks in the curve data
        worker_func = functools.partial(
            run_single_sim,
            curve_type,
            curve_base=curve_base,
            curve_derivative_base=curve_derivative_base,
        )

        # # Run n_samples in parallel across all cores
        # all_run_results = []
        # print(
        #     f"[{curve_type}] START  {datetime.now().strftime('%H:%M:%S')}", flush=True
        # )

        # --- select which runs to execute based on RUN_MODE ---
        if RUN_MODE == "nominal":
            run_ids = [
                i for i in range(n_samples) if i % N_INITS != PROBLEMATIC_INIT_SEED
            ]
        elif RUN_MODE == "problematic":
            run_ids = [
                i for i in range(n_samples) if i % N_INITS == PROBLEMATIC_INIT_SEED
            ]
        else:  # "both"
            run_ids = list(range(n_samples))

        all_run_results = []
        print(
            f"[{curve_type}] START  {datetime.now().strftime('%H:%M:%S')}  "
            f"(mode={RUN_MODE}, {len(run_ids)} runs)",
            flush=True,
        )

        # with ProcessPoolExecutor(max_workers=max_cores) as executor:
        #     for res in executor.map(worker_func, range(n_samples)):
        #         all_run_results.append(res)
        #         print(
        #             f"[{curve_type}] {datetime.now().strftime('%H:%M:%S')}  run {len(all_run_results)}/{n_samples}",
        #             flush=True,
        #         )

        with ProcessPoolExecutor(max_workers=max_cores) as executor:
            for run_id, res in zip(run_ids, executor.map(worker_func, run_ids)):
                all_run_results.append((run_id, res))
                print(
                    f"[{curve_type}] {datetime.now().strftime('%H:%M:%S')}  "
                    f"run_id={run_id}  ({len(all_run_results)}/{len(run_ids)})",
                    flush=True,
                )

        print(
            f"[{curve_type}] END    {datetime.now().strftime('%H:%M:%S')}", flush=True
        )

        # with ProcessPoolExecutor(max_workers=max_cores) as executor:
        #     # executor.map returns results in the order they were submitted
        #     for res in tqdm(
        #         executor.map(worker_func, range(n_samples)),
        #         total=n_samples,
        #         desc=f"Simulating {curve_type}",
        #     ):
        #         all_run_results.append(res)

        # Format data for a single .npz file per curve
        # save_dict = {}
        # for run_idx, run_data in enumerate(all_run_results):
        #     for key, value in run_data.items():
        #         save_dict[f"run_{run_idx:03d}_{key}"] = value
        #
        # save_path = os.path.join(results_dir, f"{curve_type}_results.pkl")
        # # np.savez_compressed(save_path, **save_dict)
        # with open(save_path, "wb") as f:
        #     pickle.dump(save_dict, f, protocol=pickle.HIGHEST_PROTOCOL)
        # print(f"Saved aggregated data to {save_path}")
        save_dict = {}
        for run_id, run_data in all_run_results:
            for key, value in run_data.items():
                save_dict[f"run_{run_id:03d}_{key}"] = value

        save_path = os.path.join(results_dir, f"{curve_type}_results.pkl")
        # Merge with existing file if present (allows running modes separately)
        if os.path.exists(save_path):
            with open(save_path, "rb") as f:
                existing = pickle.load(f)
            existing.update(save_dict)
            save_dict = existing

        with open(save_path, "wb") as f:
            pickle.dump(save_dict, f, protocol=pickle.HIGHEST_PROTOCOL)

    print("\nAll simulations complete.")
