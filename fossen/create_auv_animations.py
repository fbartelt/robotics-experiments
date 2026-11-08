import pickle
import os
import numpy as np
import uaibot as ub
import uaibot_cpp_bind as ub_cpp
import plotly.graph_objects as go
from tqdm import tqdm
from uaibot.utils import Utils
from uaibot.simobjects import Box, Frame, Ball, PointCloud
from uaibot.simobjects.curve import CurveSE3
from fossen.curves import build_curve
from fossen.create_bluerov2 import make_rov
from fossen.ocean_sim import create_sim_underwater
from plotly.subplots import make_subplots

curve_base_path = "./fossen/data"
anim_base_path = "./fossen/media"
# ellipse
# trefoil
# vertical_coil
# toroidal_coil
CURVE_TYPE = "toroidal_coil"

RUN_BASELINE = True

if RUN_BASELINE:
    results_file_name = f"results_{CURVE_TYPE}_baseline.pkl"
else:
    results_file_name = f"results_{CURVE_TYPE}_default.pkl"

results_file = os.path.join(curve_base_path, results_file_name)
curve_path = os.path.join(curve_base_path, f"{CURVE_TYPE}.npy")
dcurve_path = os.path.join(curve_base_path, f"{CURVE_TYPE}_derivative.npy")

if os.path.exists(curve_path):
    curve = np.load(curve_path, allow_pickle=True)
    curve_derivative = np.load(dcurve_path, allow_pickle=True)
else:
    raise FileExistsError("Curve was not generated")

with open(results_file, "rb") as f:
    data = pickle.load(f)

time_vec = data["time"]
log_dist = data["dist"]
log_err_pos = data["err_pos"]
log_err_ori = data["err_ori"]
log_idx = data["idx"]
log_xi = data["xi"]
log_twist = data["twist"]
log_htm = np.array(data["htm"])
requested_wrench = data["requested_wrench"]
allocated_wrench = data["allocated_wrench"]
applied_wrench = data["applied_wrench"]
allocation_residual = data["allocation_residual"]
thrust_setpoints = data["thrust_setpoints"]
saturated_thrusters = data["saturated_thrusters"]
requested_wrench

htm0 = log_htm[0]
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

log_position = log_htm[:, :3, 3].reshape(-1, 3)
followed_path = PointCloud(points=log_position.T, size=0.05, color="magenta")
sim.add(followed_path)

# sim.set_parameters(camera_follow=auv, camera_start_pose=camera_start_pose)

# sim.add(closest_p)
curve_ub = CurveSE3(points=curve, color="cyan", size=0.05, num_frames=20)
sim.add(curve_ub)
sim.set_parameters(width=1920, height=1080)


for i, t_ in enumerate(time_vec):
    t = float(t_)
    htm = log_htm[i]
    auv.add_ani_frame(time=t, htm=htm)
    auv_frame.add_ani_frame(time=t, htm=htm)
    followed_path.add_ani_frame(time=t, initial_ind=0, final_ind=i)


sim.run_in_browser()
bsl_name = "_baseline"if RUN_BASELINE else ""
for fol in range(2):
    if fol == 0:
        sim.save(anim_base_path, f"{CURVE_TYPE}{bsl_name}")
    else:
        sim.set_parameters(camera_follow=auv, camera_start_pose=camera_start_pose)
        sim.save(anim_base_path, f"{CURVE_TYPE}{bsl_name}_follow")


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


# create_fig().show()

# %%

fig = go.Figure()
for i in range(3):
    fig.add_trace(go.Scatter(y=np.array(log_xi)[:, i]))
for i in range(3):
    fig.add_trace(go.Scatter(y=np.array(log_twist)[:, i]))
fig.show()
print(2)
