import numpy as np
import uaibot as ub
import uaibot_cpp_bind as ub_cpp
import plotly.graph_objects as go
from tqdm import tqdm
from uaibot.simobjects import Box, Frame
from uaibot.simobjects.curve import CurveSE3
from fossen.curves import build_curve
from plotly.subplots import make_subplots

# ---------- Initial conditions (mirror of the C++ example) ---------------
htm0 = np.eye(4)
htm0[2, 3] = -2.0  # C++: pose(2, 3) = -3.   (0-based in NumPy)

actual_twist = np.zeros(6)
gains = 60 * np.array([2.0, 2.0, 2.0, 4.0, 4.0, 4.0])
# gains = np.array([180., 180., 88., 716., 768., 360.])

dt = 0.01  # C++: const double dt = 0.01
T = 60

CURVE_X0, CURVE_Y0, CURVE_Z0 = -2.0 * 0, -2.0 * 0, -4.0
CURVE_POINTS = 15000

# lissajous_3d
# circle
# ellipse
# trefoil
# vertical_coil
CURVE_TYPE = "circle"

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
TREFOIL_R = 1.0
TREFOIL_Z_AMP = 1.0

# Cylinder surface
CYLINDER_R = 2.0
CYLINDER_Z_AMP = 1.0
CYLINDER_N_OSCILLATIONS = 3


kn1 = 1.0
kn2 = 1.0
kt1 = 0.03
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
    kwargs["z_amp"] = TREFOIL_R
elif CURVE_TYPE == "vertical_coil":
    kwargs["R"] = CYLINDER_R
    kwargs["z_amp"] = CYLINDER_Z_AMP
    kwargs["N_oscillations"] = CYLINDER_N_OSCILLATIONS


curve, curve_derivative = build_curve(CURVE_TYPE, **kwargs)

# ---------- Scene object -------------------------------------------------
# htm0 = curve[0]
auv = Box(htm=htm0, depth=338e-3, height=254e-3, width=457e-3, color="red")
auv_frame = Frame(htm=htm0)
sim = ub.Simulation([auv, auv_frame])
curve_ub = CurveSE3(points=curve, color="cyan")
sim.add(curve_ub)
sim.set_parameters(width=1480, height=920)

# ---------- Integrate physics + build animation frames -------------------
htm_next = htm0.copy()
time_vec = np.arange(0, T, dt)
params = ub_cpp.FossenParameters()
params.dt = dt
params.velocity_gains = gains
params.model_compensation = False

log_dist, log_err_pos, log_err_ori, log_idx = [], [], [], []
log_twist, log_xi = [], []

for t in tqdm(time_vec):
    # Same call as C++: bluerov2::step(target, pose, twist, gains, dt)
    xi, dist, idx = ub.Robot.vector_field_SE3(
        htm_next,
        curve,
        kt1=kt1,
        kt2=kt2,
        kt3=kt3,
        kn1=kn1,
        kn2=kn2,
        curve_derivative=curve_derivative,
        # delta=dt,
    )
    v, omega = xi[:3], xi[3:]
    p = htm_next[:3, 3]
    target_twist = np.empty(6)
    target_twist[:3] = v + np.cross(omega, p)
    target_twist[3:] = omega
    # state = ub_cpp.fossen_step_direct(target_twist, htm_next, actual_twist, gains, dt)
    state = ub_cpp.fossen_step_stateless(target_twist, htm_next, actual_twist, params)

    log_dist.append(dist)
    log_idx.append(idx)

    closest_point = curve[idx]
    htm_next
    p_near = closest_point[:3, 3]
    ori_near = closest_point[:3, :3]
    p_curr = htm_next[:3, 3]
    ori_curr = htm_next[:3, :3]
    log_err_pos.append(np.linalg.norm(p_near - p_curr) * 100)
    trace_ = np.trace(ori_near @ np.linalg.inv(ori_curr))
    acos = np.arccos((trace_ - 1) / 2)
    # checks if acos is nan
    if np.isnan(acos):
        acos = 0
    log_err_ori.append(acos * 180 / np.pi)

    htm_next = state.pose
    actual_twist = state.twist
    auv.add_ani_frame(time=float(t), htm=htm_next)
    auv_frame.add_ani_frame(time=float(t), htm=htm_next)

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
#
