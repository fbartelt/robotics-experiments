import pickle
import os
import numpy as np
import uaibot as ub
import uaibot_cpp_bind as ub_cpp
import plotly.graph_objects as go
from tqdm import tqdm
from uaibot.utils import Utils
from uaibot.simobjects import Box, Frame, Ball
from uaibot.simobjects.curve import CurveSE3
from fossen.curves import build_curve
from fossen.create_bluerov2 import make_rov
from fossen.ocean_sim import create_sim_underwater
from fossen.baseline_adriano import (
    se3_curve_to_r3t3,
    rezende_vf_r3t3,
    rezende_distance_gradient,
    vf_to_fossen_twist,
)
from scipy.spatial.transform import Rotation as R
from plotly.subplots import make_subplots

# ---------- Initial conditions (mirror of the C++ example) ---------------
htm0 = np.eye(4)
htm0[2, 3] = -2.0  # C++: pose(2, 3) = -3.   (0-based in NumPy)

actual_twist = np.zeros(6)
gains = 6 * np.array([2.0, 2.0, 2.0, 4.0, 4.0, 4.0]) * 1.6
# gains = np.array([180., 180., 88., 716., 768., 360.])

dt = 0.01  # C++: const double dt = 0.01
T = 300
curve_base_path = "./fossen/data"

kn1 = 1.0
kn2 = 0.7
# kt1 = 1.0
kt1 = 0.7  # 0.7 is cool for toroidal coil
# kt1 = 0.3
kt2 = 0.7
kt3 = kn2

CURVE_X0, CURVE_Y0, CURVE_Z0 = -2.0 * 0, -2.0 * 0, -4.0
CURVE_POINTS = 15000 * 2

# lissajous_3d
# circle
# ellipse
# trefoil
# vertical_coil
# toroidal_coil
CURVE_TYPE = "toroidal_coil"

scl = 0.8 / 1.5
scl = 1.0
TEST_RADIUS = 3.0 * scl
TEST_MINOR_RADIUS = 2.0 * scl
TEST_N_COILS = 4.0
TEST_ROLLS = 2.0 * 0


# TEST_RADIUS = 2.0
# TEST_ROLLS = 2.0 * 0

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
elif CURVE_TYPE == "toroidal_coil":
    kwargs["R"] = TEST_RADIUS
    kwargs["r"] = TEST_MINOR_RADIUS
    kwargs["n_coils"] = TEST_N_COILS
    kwargs["n_rolls"] = TEST_ROLLS


results_file_name = f"results_{CURVE_TYPE}_baseline.pkl"
results_file = os.path.join(curve_base_path, results_file_name)
curve_path = os.path.join(curve_base_path, f"{CURVE_TYPE}.npy")
dcurve_path = os.path.join(curve_base_path, f"{CURVE_TYPE}_derivative.npy")

# if os.path.exists(curve_path):
#     curve = np.load(curve_path, allow_pickle=True)
#     curve_derivative = np.load(dcurve_path, allow_pickle=True)
# else:
#     curve, curve_derivative = build_curve(CURVE_TYPE, **kwargs)
if os.path.exists(curve_path):
    curve = np.load(curve_path, allow_pickle=True)
    curve_derivative = np.load(dcurve_path, allow_pickle=True)
else:
    curve, curve_derivative = build_curve(CURVE_TYPE, **kwargs)

# ---- Rezende baseline: precompute the R^3 x T^3 coordinate curve ----
p_curve, phi_curve = se3_curve_to_r3t3(curve)

# VF tuning (Rezende's v_r is a mixed m+rad speed; 0.4 ≈ 0.4 m/s surge)
REZENDE_V_R = 1.0
scale_pos = 1 / np.sqrt(3.0)
scale_pos = 1.0
curve_scale = curve.copy()
curve_scale[:, :3, 3] *= scale_pos

# ---------- Scene object -------------------------------------------------
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
    # ---- 1. Controller feedback (Rezende VF on R^3 x T^3) ----
    ctrl_pose = state.measurement.pose if state.measurement.valid else state.pose
    p_ctrl = ctrl_pose[:3, 3]
    phi_ctrl = R.from_matrix(ctrl_pose[:3, :3]).as_euler("xyz")

    psi = rezende_vf_r3t3(
        p_ctrl * scale_pos,
        phi_ctrl,
        p_curve * scale_pos,
        phi_curve,
        v_r=REZENDE_V_R,
        kn1=kn1,
        kn2=kn2,
        kt1=kt1,
        kt2=kt2,
        kt3=kt3,
    )
    target_twist = vf_to_fossen_twist(psi, phi_ctrl)

    # ---- 2. Physics step ----
    state = ub_cpp.fossen_step(target_twist, state, params)
    diagnostics = state.diagnostics

    # ---- 3. Logging (uses ground truth and the Rezende metric) ----
    true_pose = state.pose
    true_twist = state.twist
    rot = true_pose[:3, :3].T

    p_true = true_pose[:3, 3]
    phi_true = R.from_matrix(true_pose[:3, :3]).as_euler("xyz")
    dist, closest_idx, _ = rezende_distance_gradient(
        p_true, phi_true, p_curve, phi_curve
    )

    closest_point = curve[closest_idx]
    p_near = closest_point[:3, 3]
    ori_near = closest_point[:3, :3]
    p_curr = true_pose[:3, 3]
    ori_curr = true_pose[:3, :3]

    log_err_pos.append(np.linalg.norm(p_near - p_curr) * 100)
    trace_ = np.trace(ori_near @ np.linalg.inv(ori_curr))
    acos = np.arccos(np.clip((trace_ - 1) / 2, -1.0, 1.0))
    if np.isnan(acos):
        acos = 0

    log_dist.append(dist)
    log_idx.append(closest_idx)
    log_xi.append((rot @ target_twist[3:].reshape(-1, 1)).ravel())
    log_twist.append((rot @ true_twist[3:].reshape(-1, 1)).ravel())
    log_err_ori.append(acos * 180 / np.pi)
    log_htm.append(true_pose.copy())
    log_requested_wrench.append(diagnostics.requested_wrench)
    log_allocated_wrench.append(diagnostics.allocated_wrench)
    log_applied_wrench.append(diagnostics.applied_wrench)
    log_allocation_residual.append(diagnostics.allocation_residual)
    log_thrust_setpoints.append(diagnostics.thrust_setpoints)
    log_saturated_thrusters.append(diagnostics.saturated_thrusters)

    auv.add_ani_frame(time=float(t), htm=true_pose)
    auv_frame.add_ani_frame(time=float(t), htm=true_pose)
    closest_p.add_ani_frame(time=float(t), htm=closest_point)

data = {
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

# %%

fig = go.Figure()
for i in range(3):
    fig.add_trace(go.Scatter(y=np.array(log_xi)[:, i]))
for i in range(3):
    fig.add_trace(go.Scatter(y=np.array(log_twist)[:, i]))
fig.show()
print(2)
