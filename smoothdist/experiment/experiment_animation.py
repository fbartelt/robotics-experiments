import pickle
import numpy as np
import uaibot as ub
import plotly.graph_objects as go
from franka_tests.create_franka_emika_3_mod import create_franka_emika_3_mod
import webbrowser
from pathlib import Path


# Save  the simulation to see the results (open the html file control_sim.html that will
# be generated in the same folder the script was ran)
def open_in_browser(filename: str):
    """
    Opens an HTML file in the system's default web browser.
    Works cross-platform (Linux, macOS, Windows).
    """
    path = Path(filename).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    # Convert to file:// URL and open
    webbrowser.open_new_tab(path.as_uri())


# -------------------------------------------------------------------- #
# Mode 0 -> create Euclidean simulation
# Mode 1 -> create HD-SDF simulation
mode = 0
# Whether to plot or not the witness points (top box-> claw)
add_witness = True
# -------------------------------------------------------------------- #

esdf_data = "./experiment_data/mode_0/video_7_data/data.pickle"
hdsdf_data = "./experiment_data/mode_1/video_1_data/data.pickle"

with open(esdf_data, "rb") as f:
    esdf = pickle.load(f)

with open(hdsdf_data, "rb") as f:
    hdsdf = pickle.load(f)

data = hdsdf if mode == 1 else esdf

real_obstacles = []
real_obstacles.append(
    ub.Box(
        htm=ub.Utils.trn([0.53, 0.16, 0.45]),
        width=0.35,
        depth=0.05,
        height=0.90,
        # color="magenta",
        color="#A8CEAC",
    )
)
real_obstacles.append(
    ub.Box(
        htm=ub.Utils.trn([0.53, -0.16, 0.45]),
        width=0.35,
        depth=0.05,
        height=0.90,
        # color="magenta",
        color="#A8CEAC",
    )
)
real_obstacles.append(
    ub.Box(
        htm=ub.Utils.trn([0.53, 0.00, 0.925]),
        width=0.35,
        depth=0.35,
        height=0.05,
        # color="magenta",
        color="#E8DD94",
    )
)

# Agumented obstacles
obstacles = []
expand = 5e-2

expand_opacity = 0.5
expand_color = "cyan"
obstacles.append(
    ub.Box(
        htm=ub.Utils.trn([0.53, 0.16, 0.45]),
        width=0.35 + expand,
        depth=0.05 + expand,
        height=0.90 + expand,
        # color=expand_color,
        color="#A8CEAC",
        opacity=expand_opacity,
    )
)
obstacles.append(
    ub.Box(
        htm=ub.Utils.trn([0.53, -0.16, 0.45]),
        width=0.35 + expand,
        depth=0.05 + expand,
        height=0.90 + expand,
        color="#A8CEAC",
        # color=expand_color,
        opacity=expand_opacity,
    )
)
obstacles.append(
    ub.Box(
        htm=ub.Utils.trn([0.53, 0.00, 0.925]),
        width=0.35 + expand,
        depth=0.35 + expand,
        height=0.05 + expand,
        color="#E8DD94",
        # color=expand_color,
        opacity=expand_opacity,
    )
)

# Initial configuration (rad)
q = np.matrix([[1.0582, -1.3811, 0.3629, -1.9647, -0.959, 1.4881, -0.1534]]).T

# Target pose
htm_tg = (
    ub.Utils.trn([0.64, 0, 0.75]) * ub.Utils.roty(np.pi / 2) * ub.Utils.rotz(np.pi / 2)
)


franka = create_franka_emika_3_mod()
sim = ub.Simulation.create_sim_grid(franka)
franka.add_ani_frame(0, q)

if mode == 1:
    for obs in obstacles:
        sim.add(obs)

for obs in real_obstacles:
    sim.add(obs)

if add_witness:
    col_model = None
    for i, link in enumerate(franka.links):
        for j, (col_obj, _) in enumerate(link.col_objects):
            if i == 6 and j == 0:
                col_model = col_obj
                sim.add(col_obj)

frame_tg = ub.Frame(htm=htm_tg, size=0.1)
sim.add(frame_tg)

if add_witness:
    ball_claw = ub.Ball(color='magenta', radius=1e-2)
    ball_obs = ub.Ball(color='cyan', radius=1e-2)
    sim.add([ball_obs, ball_claw])

time_ = data["timestamp"]
t0 = time_[0]
time = np.array([t - t0 for t in time_])
dist = np.array(data["hist_dist"]).reshape(-1, 1)
q_hist = np.array(data["q"]).reshape(-1, 7)
qdot_hist = np.array(data["hist_dq"]).reshape(-1, 7)
# u_hist = qdot_hist[:, i]

for i, t in enumerate(time):
    q = q_hist[i]
    no_iter_max = 300
    tol = 2e-4

    # For plotting witness
    if add_witness:
        for obs in real_obstacles[2:]:
            p_col, p_obs, *_ = col_model.compute_dist(
                obj=obs, h=1e-6, eps=0, no_iter_max=no_iter_max, tol=tol
            )
            ball_claw.add_ani_frame(time=t.item(), htm=ub.Utils.trn(p_col))
            ball_obs.add_ani_frame(time=t.item(), htm=ub.Utils.trn(p_obs))

    franka.add_ani_frame(q=q, time=float(t))

file_name = f"experiment_mode_{mode}"
sim.save(file_name=file_name)
open_in_browser(file_name + ".html")
