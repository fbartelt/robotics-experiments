import os
import numpy as np
import uaibot as ub
import uaibot_cpp_bind as ub_cpp
from tqdm import tqdm
from scipy.spatial.transform import Rotation as R
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
import functools

curve_base_path = "./fossen/data"
curve_types = ["ellipse", "trefoil", "vertical_coil"]

# ----------------------- FIXED PARAMETERS -----------------------------
gains = 6 * np.array([2.0, 2.0, 2.0, 4.0, 4.0, 4.0])
dt = 0.01
T = 300
n_samples = 100
scale_pos = 1 / np.sqrt(3)

kn1 = 1
kn2 = 1.0
kt1 = 0.03 / 2
kt2 = 1.0
kt3 = kn2


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


def run_single_sim(curve_type, i, curve_base, curve_derivative_base):
    """Function executed by each parallel worker for a single run."""
    # Initialize parameters inside the worker to ensure safety
    params = get_params()
    CURVE_POINTS = len(curve_base)

    curve, curve_derivative = rand_se3_curve(curve_base, curve_derivative_base, seed=i)
    curve_scale = curve.copy()
    curve_scale[:, :3, 3] *= scale_pos

    htm0 = curve[0].copy()
    # Use a local RNG per worker to avoid seed collisions
    local_rng = np.random.default_rng(i + 1000)

    offset_dir = local_rng.standard_normal(3)
    offset_dir /= np.linalg.norm(offset_dir)
    htm0[:3, 3] += offset_dir * local_rng.uniform(0.5, 2.0)

    R_err = R.from_euler("xyz", local_rng.uniform(-np.pi / 6, np.pi / 6, 3)).as_matrix()
    htm0[:3, :3] = htm0[:3, :3] @ R_err

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

        _, dist, closest_idx = ub.Robot.vector_field_SE3(
            true_pose_scale,
            curve,
            kt1=kt1,
            kt2=kt2,
            kt3=kt3,
            kn1=kn1,
            kn2=kn2,
            curve_derivative=curve_derivative,
        )

        closest_point = curve[closest_idx]
        p_near = closest_point[:3, 3]
        ori_near = closest_point[:3, :3]
        p_curr = true_pose[:3, 3]
        ori_curr = true_pose[:3, :3]

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

        # Run n_samples in parallel across all cores
        all_run_results = []
        with ProcessPoolExecutor(max_workers=max_cores) as executor:
            # executor.map returns results in the order they were submitted
            for res in tqdm(
                executor.map(worker_func, range(n_samples)),
                total=n_samples,
                desc=f"Simulating {curve_type}",
            ):
                all_run_results.append(res)

        # Format data for a single .npz file per curve
        save_dict = {}
        for run_idx, run_data in enumerate(all_run_results):
            for key, value in run_data.items():
                save_dict[f"run_{run_idx:03d}_{key}"] = value

        save_path = os.path.join(results_dir, f"{curve_type}_results.npz")
        np.savez_compressed(save_path, **save_dict)
        print(f"Saved aggregated data to {save_path}")

    print("\nAll simulations complete.")
