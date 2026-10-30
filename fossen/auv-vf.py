import numpy as np
import uaibot as ub
import uaibot_cpp_bind as ub_cpp
import plotly.graph_objects as go
from tqdm import tqdm
from uaibot.simobjects import Box, Frame, Ball
from uaibot.simobjects.curve import CurveSE3
from fossen.curves import build_curve
from plotly.subplots import make_subplots

# ---------- Initial conditions (mirror of the C++ example) ---------------
htm0 = np.eye(4)
htm0[2, 3] = -2.0  # C++: pose(2, 3) = -3.   (0-based in NumPy)

actual_twist = np.zeros(6)
gains = 6 * np.array([2.0, 2.0, 2.0, 4.0, 4.0, 4.0])
# gains = np.array([180., 180., 88., 716., 768., 360.])

dt = 0.01  # C++: const double dt = 0.01
T = 70

CURVE_X0, CURVE_Y0, CURVE_Z0 = -2.0 * 0, -2.0 * 0, -4.0
CURVE_POINTS = 15000 * 2

# lissajous_3d
# circle
# ellipse
# trefoil
# vertical_coil
CURVE_TYPE = "trefoil"

CURVE_RADIUS = 1.5 * 2
CURVE_N_ROLLS = 1.0

# Ellipse
ELLIPSE_A = 2.0 * 1
ELLIPSE_B = 1.0 * 1

# Lissajous
LISSAJOUS_A = 4.0
LISSAJOUS_B = 2.0
LISSAJOUS_C = 1.0

# Trefoil
TREFOIL_R = 1.0 * 2
TREFOIL_Z_AMP = 0.5 * 1

# Cylinder surface
CYLINDER_R = 2.0
CYLINDER_Z_AMP = 1.0
CYLINDER_N_OSCILLATIONS = 3


kn1 = 1
kn2 = 1.0
kt1 = 0.03 / 2
kt2 = 1.0
kt3 = kn2

kwargs = {
    "n_points": CURVE_POINTS,
    "z0": CURVE_Z0,
    "center_xy": (CURVE_X0, CURVE_Y0),
}

if CURVE_TYPE == "circle":
    kwargs["radius"] = CURVE_RADIUS
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


curve, curve_derivative = build_curve(CURVE_TYPE, **kwargs)
scale_pos = 1 / np.sqrt(3)
curve_scale = curve.copy()
curve_scale[:, :3, 3] *= scale_pos

# ---------- Scene object -------------------------------------------------
# htm0 = curve[0]
closest_p = Ball(htm=np.eye(4), opacity=0.8, radius=0.05, color="pink")
auv = Box(htm=htm0, depth=338e-3, height=254e-3, width=457e-3, color="red", opacity=0.3)
auv_frame = Frame(htm=htm0)
sim = ub.Simulation([auv, auv_frame])
sim.add(closest_p)
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
params.thrust_max = 5.25 * 9.80665 * np.ones((8,))
params.thrust_min = -4.10 * 9.80665 * np.ones((8,))

# Measurement Noise Configuration (converted from C++ to NumPy arrays)
params.measurement_noise.enabled = True
params.measurement_noise.seed = 42
params.measurement_noise.sample_period = 0.01 * 0

params.measurement_noise.white_variances = np.array(
    [
        0 * 0.02**2,
        0 * 0.02**2,
        0 * 0.003**2,
        0 * 0.002**2,
        0 * 0.002**2,
        0 * 0.005**2,
        0 * 0.005**2,
        0 * 0.005**2,
        0 * 0.005**2,
        0 * 0.001**2,
        0 * 0.001**2,
        0 * 0.001**2,
    ],
    dtype=np.float64,
)

params.measurement_noise.colored_variances = np.array(
    [
        0 * 0.01**2,
        0 * 0.01**2,
        0 * 0.005**2,  # Position: ~1 cm drift
        0 * 0.002**2,
        0 * 0.002**2,
        0 * 0.005**2,
        0.005**2,
        0.005**2,
        0.005**2,
        0.001**2,
        0.001**2,
        0.001**2,
        # 0.05**2,
        # 0.05**2,
        # 0.01**2,
        # 0.005**2,
        # 0.005**2,
        # 0.02**2,
        # 0.01**2,
        # 0.01**2,
        # 0.01**2,
        # 0.003**2,
        # 0.003**2,
        # 0.003**2,
    ],
    dtype=np.float64,
)

# params.measurement_noise.correlation_times = 0.2 * np.ones((12,))
params.measurement_noise.correlation_times = np.array(
    [10.0, 10.0, 10.0, 1.0, 1.0, 5.0, 0.2, 0.2, 0.2, 30.0, 30.0, 30.0], dtype=np.float64
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
log_twist, log_xi = [], []

prev_idx = 0
window_size = int(0.1 * CURVE_POINTS)
dist = 1e3

noise_xi_H = np.zeros((6,))
std_noise = np.array(
    [
        0.05,
        0.05,
        0.05,
        0.005,
        0.005,
        0.005,
    ]
)

A = np.exp(-dt / 5.0)

for t in tqdm(time_vec):
    half_win = window_size // 2
    noise_xi_H = A * noise_xi_H + std_noise * np.sqrt(1 - A**2) * np.random.normal(
        size=(6,)
    )

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
        # Handle wrap-around at the seam
        curve_window = np.vstack((curve_scale[start:], curve_scale[:end]))
        deriv_window = np.vstack((curve_derivative[start:], curve_derivative[:end]))

    # 1. CONTROLLER FEEDBACK (Uses Noisy Measurement)
    # If measurement isn't valid yet, fallback to ground truth
    ctrl_pose = state.measurement.pose if state.measurement.valid else state.pose
    ctrl_twist = state.measurement.twist if state.measurement.valid else state.twist

    ctrl_pose = ub_cpp.expSE3(ub_cpp.SmapSE3(noise_xi_H)) @ state.pose
    scale_pose = ctrl_pose.copy()
    scale_pose[:3, 3] *= scale_pos

    xi, dist, local_idx = ub.Robot.vector_field_SE3(
        scale_pose,
        # curve,
        curve_window,
        kt1=kt1,
        kt2=kt2,
        kt3=kt3,
        kn1=kn1,
        kn2=kn2,
        # curve_derivative=curve_derivative,
        curve_derivative=deriv_window,
    )

    # 4. Map the local index back to the global curve index
    global_idx = (start + local_idx) % CURVE_POINTS
    prev_idx = global_idx
    idx = global_idx

    v, omega = xi[:3], xi[3:]
    p_ctrl = ctrl_pose[:3, 3]

    target_twist = np.empty(6)
    target_twist[:3] = v + np.cross(omega, p_ctrl)
    target_twist[3:] = omega

    # 2. PHYSICS STEP (Updates state.pose, state.twist, and state.measurement)
    state = ub_cpp.fossen_step(target_twist, state, params)

    # 3. LOGGING (Uses Ground Truth)
    true_pose = state.pose
    true_twist = state.twist
    rot = true_pose[:3, :3].T

    log_dist.append(dist)
    log_idx.append(idx)
    log_xi.append((rot @ target_twist[3:].reshape(-1, 1)).ravel())
    log_twist.append((rot @ true_twist[3:].reshape(-1, 1)).ravel())

    closest_point = curve[idx]
    p_near = closest_point[:3, 3]
    ori_near = closest_point[:3, :3]
    p_curr = true_pose[:3, 3]
    ori_curr = true_pose[:3, :3]

    log_err_pos.append(np.linalg.norm(p_near - p_curr) * 100)
    trace_ = np.trace(ori_near @ np.linalg.inv(ori_curr))
    acos = np.arccos(np.clip((trace_ - 1) / 2, -1.0, 1.0))  # Clip for numerical safety
    if np.isnan(acos):
        acos = 0
    log_err_ori.append(acos * 180 / np.pi)

    auv.add_ani_frame(time=float(t), htm=true_pose)
    auv_frame.add_ani_frame(time=float(t), htm=true_pose)
    closest_p.add_ani_frame(time=float(t), htm=closest_point)

# ---------- Show ---------------------------------------------------------
sim.run_in_browser()


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
