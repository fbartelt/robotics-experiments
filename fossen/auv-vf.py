import torch
import pickle
import os
import numpy as np
import uaibot as ub
import uaibot_cpp_bind as ub_cpp
import plotly.graph_objects as go
from tqdm import tqdm
from uaibot.utils import Utils
from uaibot.simobjects import Box, Frame, Ball
from scipy.spatial.transform import Rotation as R
from uaibot.simobjects.curve import CurveSE3
from fossen.curves import build_curve
from fossen.create_bluerov2 import make_rov
from fossen.ocean_sim import create_sim_underwater
from plotly.subplots import make_subplots
from fossen.torch_vf import SE3VectorField, ec_distance_and_pos_ori

# ---- Torch device / dtype (float64 to match numpy) -----------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float64

# Instantiate the solver once (delta / ds match the C++ constants c_delta / c_ds)
vf_solver = SE3VectorField(
    delta=1e-3,
    ds=1e-3,
    normalize_tangent=True,
    normalize_normal=False,
    # scale_pos=1.0 / np.sqrt(3),
    scale_pos=1.0,
)

# ---------- Initial conditions (mirror of the C++ example) ---------------
htm0 = np.eye(4)
htm0[2, 3] = -2.0  # C++: pose(2, 3) = -3.   (0-based in NumPy)

actual_twist = np.zeros(6)
gains = 6 * np.array([2.0, 2.0, 2.0, 4.0, 4.0, 4.0]) * 1.6
# gains = np.array([180., 180., 88., 716., 768., 360.])

dt = 0.01  # C++: const double dt = 0.01
T = 120
curve_base_path = "./fossen/data"
scale_pos = 1 / np.sqrt(3.0)
scale_pos = 1.0
curve_seed = None
# curve_seed = 61  # used to check failed runs in variations

kn1 = 1.0
kn2 = 0.7
# kt1 = 1.0
kt1 = 0.7 # 0.9 is too high, 0.4 is tpp slow
kt2 = 0.7
kt3 = kn2

CURVE_X0, CURVE_Y0, CURVE_Z0 = -2.0 * 0, -2.0 * 0, -4.0
CURVE_POINTS = 15000 * 2

# lissajous_3d
# circle
# ellipse
# trefoil
# vertical_coil
CURVE_TYPE = "trefoil"

scl = (0.8 / 1.5)
scl = 1.0
TEST_RADIUS = 3.0 * scl
TEST_MINOR_RADIUS = 2.0 * scl
TEST_N_COILS = 4.0
TEST_ROLLS = 1.0

CIRCLE_RADIUS = 3.0
CURVE_N_ROLLS = 1.0

# Ellipse
ELLIPSE_A = 3.0 * 1
ELLIPSE_B = 2.0 * 1

# Lissajous
LISSAJOUS_A = 4.0
LISSAJOUS_B = 3.5
LISSAJOUS_C = 0.5

# Trefoil
TREFOIL_R = 1.0 * 2
TREFOIL_Z_AMP = 0.5 * 1

# Cylinder surface
CYLINDER_R = 2.0
CYLINDER_Z_AMP = 1.0
CYLINDER_N_OSCILLATIONS = 3


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


kwargs = {
    "n_points": CURVE_POINTS,
    "z0": CURVE_Z0,
    "center_xy": (CURVE_X0, CURVE_Y0),
}

if CURVE_TYPE == "circle":
    kwargs["radius"] = CIRCLE_RADIUS
    kwargs["n_rolls"] = CURVE_N_ROLLS
elif CURVE_TYPE == "ellipse":
    kwargs["a"] = ELLIPSE_A
    kwargs["b"] = ELLIPSE_B
elif CURVE_TYPE == "lissajous_3d":
    kwargs["A"] = LISSAJOUS_A
    kwargs["B"] = LISSAJOUS_B
    kwargs["C"] = LISSAJOUS_C
elif CURVE_TYPE == "trefoil":
    kwargs["R"] = TREFOIL_R
    kwargs["z_amp"] = TREFOIL_Z_AMP
elif CURVE_TYPE == "vertical_coil":
    kwargs["R"] = CYLINDER_R
    kwargs["z_amp"] = CYLINDER_Z_AMP
    kwargs["N_oscillations"] = CYLINDER_N_OSCILLATIONS
elif CURVE_TYPE == "test" or CURVE_TYPE == "gimbal":
    kwargs["R"] = TEST_RADIUS
    kwargs["r"] = TEST_MINOR_RADIUS
    kwargs["n_coils"] = TEST_N_COILS
    kwargs["n_rolls"] = TEST_ROLLS


results_file_name = f"results_{CURVE_TYPE}_default.pkl"
results_file = os.path.join(curve_base_path, results_file_name)
curve_path = os.path.join(curve_base_path, f"{CURVE_TYPE}.npy")
dcurve_path = os.path.join(curve_base_path, f"{CURVE_TYPE}_derivative.npy")

if os.path.exists(curve_path):
    curve = np.load(curve_path, allow_pickle=True)
    curve_derivative = np.load(dcurve_path, allow_pickle=True)
else:
    print(f"Curve {curve_path} not found! Generating {CURVE_POINTS}.")
    curve, curve_derivative = build_curve(CURVE_TYPE, **kwargs)

if curve_seed is None:
    pass
else:
    curve, curve_derivative = rand_se3_curve(curve, curve_derivative, seed=curve_seed)

curve_scale = curve.copy()
curve_scale[:, :3, 3] *= scale_pos
# Convert curves to torch tensors ONCE (kept on the same device for the whole run)
curve_scale_t = torch.as_tensor(curve_scale, dtype=DTYPE, device=DEVICE)  # (M,4,4)
deriv_t = torch.as_tensor(curve_derivative, dtype=DTYPE, device=DEVICE)  # (M,4,4)

# ---------- Scene object -------------------------------------------------
# htm0 = curve[int(CURVE_POINTS / 2)]
# htm0 = curve[0]
closest_p = Ball(htm=np.eye(4), opacity=0.8, radius=0.05, color="pink")
# auv = Box(htm=htm0, depth=338e-3, height=254e-3, width=457e-3, color="red", opacity=0.3)
auv = make_rov(htm=htm0)
auv_frame = Frame(htm=htm0)
# sim = ub.Simulation([auv, auv_frame])
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
# sim.set_parameters(camera_follow=auv, camera_start_pose=camera_start_pose)

# sim.add(closest_p)
curve_ub = CurveSE3(points=curve, color="cyan")
sim.add(curve_ub)
sim.set_parameters(width=1480, height=920)

# ---------- Integrate physics + build animation frames -------------------
htm_next = htm0.copy()
time_vec = np.arange(0, T, dt)

# ---------- Parameters & Noise Setup ---------------
params = ub_cpp.FossenParameters()
params.dt = dt
params.velocity_gains = gains
params.model_compensation = False
params.thruster_time_constant = 0
params.thrust_max = 5.25 * 9.80665 * np.ones((8,)) * 2
params.thrust_min = -4.10 * 9.80665 * np.ones((8,)) * 2

# Measurement Noise Configuration (converted from C++ to NumPy arrays)
params.measurement_noise.enabled = True
params.measurement_noise.seed = 42
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
    [10.0, 10.0, 10.0, 1.0, 1.0, 2.0, 0.8, 0.8, 0.8, 0.3, 0.3, 0.3], dtype=np.float64
)

# ---------- Initialize State ---------------
# We initialize the State object and seed it with the starting pose
state = ub_cpp.FossenState()
state.pose = htm0.copy()
state.twist = actual_twist.copy()

# Initialize the measurement manually for the first iteration
state.measurement.pose = htm0.copy()
state.measurement.twist = actual_twist.copy()
state.measurement.valid = True

# ---------- Integrate physics + build animation frames -------------------
time_vec = np.arange(0, T, dt)
log_dist, log_err_pos, log_err_ori, log_idx = [], [], [], []
log_twist, log_xi, log_htm = [], [], []
log_requested_wrench, log_allocated_wrench = [], []
log_applied_wrench, log_allocation_residual = [], []
log_thrust_setpoints, log_saturated_thrusters = [], []

prev_idx = 0
window_size = int(0.1 * CURVE_POINTS)
dist = 1e3

for t in tqdm(time_vec):
    half_win = window_size // 2

    if dist < 0.5:
        start = (prev_idx - half_win) % CURVE_POINTS
        end = (prev_idx + half_win) % CURVE_POINTS
    else:
        start = 0
        end = CURVE_POINTS

    # 1. CONTROLLER FEEDBACK (Uses Noisy Measurement)
    # If measurement isn't valid yet, fallback to ground truth
    ctrl_pose = state.measurement.pose if state.measurement.valid else state.pose
    ctrl_twist = state.measurement.twist if state.measurement.valid else state.twist

    scale_pose = ctrl_pose.copy()
    scale_pose[:3, 3] *= scale_pos

    # ---- Build window on device (preserves the exact wrap-around logic) ------
    if start < end:
        curve_window_t = curve_scale_t[start:end]
        deriv_window_t = deriv_t[start:end]
    else:
        curve_window_t = torch.cat((curve_scale_t[start:], curve_scale_t[:end]), dim=0)
        deriv_window_t = torch.cat((deriv_t[start:], deriv_t[:end]), dim=0)

    # ---- Single state -> batched (1, 4, 4) -----------------------------------
    scale_pose_t = torch.as_tensor(scale_pose, dtype=DTYPE, device=DEVICE).unsqueeze(0)

    # ---- Batched torch vector field ------------------------------------------
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
    # Apply a constant thrust to the vertical thrusters to pitch the vehicle

    # ---- Back to numpy for downstream physics / logging ----------------------
    xi = twist_t.squeeze(0).cpu().numpy()  # (6,)   v || omega
    dist = float(dist_t.item())
    local_idx = int(idx_t.item())

    # 4. Map the local index back to the global curve index
    global_idx = (start + local_idx) % CURVE_POINTS
    prev_idx = global_idx
    closest_idx = global_idx

    v, omega = xi[:3], xi[3:]
    p_ctrl = ctrl_pose[:3, 3]

    target_twist = np.empty(6)
    target_twist[:3] = v + np.cross(omega, p_ctrl)
    target_twist[3:] = omega

    # 2. PHYSICS STEP (Updates state.pose, state.twist, and state.measurement)
    state = ub_cpp.fossen_step(target_twist, state, params)
    diagnostics = state.diagnostics

    # 3. LOGGING (Uses Ground Truth)
    true_pose = state.pose
    true_twist = state.twist
    rot = true_pose[:3, :3].T
    true_pose_scale = true_pose.copy()
    true_pose_scale[:3, 3] *= scale_pos

    log_dist.append(dist)
    log_idx.append(closest_idx)
    # log_err_ori.append(acos * 180 / np.pi)
    log_xi.append((rot @ target_twist[3:].reshape(-1, 1)).ravel())
    log_twist.append((rot @ true_twist[3:].reshape(-1, 1)).ravel())
    log_htm.append(true_pose.copy())
    log_requested_wrench.append(diagnostics.requested_wrench)
    log_allocated_wrench.append(diagnostics.allocated_wrench)
    log_applied_wrench.append(diagnostics.applied_wrench)
    log_allocation_residual.append(diagnostics.allocation_residual)
    log_thrust_setpoints.append(diagnostics.thrust_setpoints)
    log_saturated_thrusters.append(diagnostics.saturated_thrusters)

    auv.add_ani_frame(time=float(t), htm=true_pose)
    auv_frame.add_ani_frame(time=float(t), htm=true_pose)
    # closest_p.add_ani_frame(time=float(t), htm=closest_point)

# ------------------------------------------------------------------
# True-pose metrics, computed post-hoc on the full curve (batched).
# ------------------------------------------------------------------

htm_batch = torch.as_tensor(np.array(log_htm), dtype=DTYPE, device=DEVICE)
htm_batch_scl = htm_batch.clone()
htm_batch_scl[..., :3, 3] *= scale_pos

curve_full_t = torch.as_tensor(curve, dtype=DTYPE, device=DEVICE)  # unscaled
curve_full_scl_t = torch.as_tensor(curve_scale, dtype=DTYPE, device=DEVICE)  # scaled

dists_true, idx_true, pos_err_true_cm, ori_err_true_deg = ec_distance_and_pos_ori(
    vf_solver,
    states_scaled=htm_batch_scl,
    curve_scaled=curve_full_scl_t,
    states_unscaled=htm_batch,
    curve_unscaled=curve_full_t,
    chunk_size=None,  # auto-chunks to ~5e6 pairs
)

# Store alongside the in-loop (noisy-derived) logs for direct comparison
# Optional: overwrite the "primary" metrics with the true ones for plots
data = {
    "time": time_vec,
    "dist": np.array(log_dist),
    "dist_true": dists_true.cpu().numpy(),
    # "err_pos": np.array(log_err_pos),
    # "err_ori": np.array(log_err_ori),
    "err_pos": pos_err_true_cm.cpu().numpy(),
    "err_ori": ori_err_true_deg.cpu().numpy(),
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
}

log_dist = data["dist_true"]
log_err_pos = data["err_pos"]
log_err_ori = data["err_ori"]

# ---------- Show ---------------------------------------------------------
sim.run_in_browser()
with open(results_file, "wb") as f:
    pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)


def create_fig():
    # Create a figure with three plots, one above another. First the distance,
    # then position error, and the orientation error
    fig = make_subplots(rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.02)
    fig.add_trace(
        go.Scatter(x=time_vec, y=log_dist, showlegend=False, line=dict(width=3)),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(x=time_vec, y=log_err_pos, showlegend=False, line=dict(width=3)),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Scatter(x=time_vec, y=log_err_ori, showlegend=False, line=dict(width=3)),
        row=3,
        col=1,
    )
    fig.add_trace(
        go.Scatter(x=time_vec, y=log_idx, showlegend=False, line=dict(width=3)),
        row=4,
        col=1,
    )
    fig.update_xaxes(
        title_text="Time (s)", gridcolor="gray", zerolinecolor="gray", row=3, col=1
    )
    fig.update_xaxes(
        title_text="", gridcolor="gray", zerolinecolor="gray", row=1, col=1
    )
    fig.update_xaxes(
        title_text="", gridcolor="gray", zerolinecolor="gray", row=2, col=1
    )
    fig.update_yaxes(
        title_text="Distance D",
        gridcolor="gray",
        zerolinecolor="gray",
        row=1,
        col=1,
        title_standoff=30,
    )
    fig.update_yaxes(
        title_text="Pos. error (cm)",
        gridcolor="gray",
        zerolinecolor="gray",
        row=2,
        col=1,
        title_standoff=30,
    )
    fig.update_yaxes(
        title_text="Ori. error (deg)",
        gridcolor="gray",
        zerolinecolor="gray",
        row=3,
        col=1,
        title_standoff=30,
    )
    fig.update_yaxes(
        title_text="Closest index",
        gridcolor="gray",
        zerolinecolor="gray",
        row=4,
        col=1,
        title_standoff=30,
    )

    # fig.update_layout(width=718.110, height=605.9155)
    fig.update_layout(width=1280, height=900)
    return fig


create_fig().show()

ss_idxs = np.where(time_vec > 20)[0]
ss_D = log_dist[ss_idxs].mean()
ss_pos = log_err_pos[ss_idxs].mean()
ss_ori = log_err_ori[ss_idxs].mean()
print(f"Steady state errors-- D: {ss_D}, pos: {ss_pos} cm, ori: {ss_ori}°")
# %%

fig = go.Figure()
for i in range(3):
    fig.add_trace(go.Scatter(y=np.array(log_xi)[:, i]))
for i in range(3):
    fig.add_trace(go.Scatter(y=np.array(log_twist)[:, i]))
fig.show()
print(2)
