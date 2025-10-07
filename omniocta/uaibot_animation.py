# %%
import re
import pickle
import os
import webbrowser
import numpy as np
import scipy as sp
import uaibot as ub
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import plotly.colors as pc
from uaibot_cpp_bind import expSO3, SmapSO3, SmapSE3, expSE3, ECdistance
from plotly.subplots import make_subplots
from pathlib import Path


def progress_bar(i, imax):
    """Prints a progress bar in the terminal."""
    bar_len = 60
    filled_len = int(round(bar_len * i / float(imax)))

    percents = round(100.0 * i / float(imax), 1)
    bar = "=" * filled_len + "-" * (bar_len - filled_len)

    print(f"[{bar}] {percents}%\r", end="")
    if i == imax:
        print()


def pose2htm(p, R):
    """Homogeneous transformation matrix from position and rotation."""
    p = np.array(p)
    htm = np.eye(4)
    htm[0:3, 0:3] = R
    htm[0:3, 3] = p.ravel()
    return htm


def hd(s, r=1, b=1, d=0.2):
    """Curve parametrization used in paper. This is based on the hyperbolic
    paraboloid.

    Parameters
    ----------
    s : float
        Parameter of the curve. It must be in the interval [0, 1].
    r : float, optional
        Radius of the curve in XY plane. The default is 1.
    b : float, optional
        Height of the curve. The default is 1.
    d : float, optional
        Curvature of the curve. The default is 0.2.

    Returns
    -------
    hds : np.array
        Homogeneous transformation matrix of the curve evaluated at parameter s.
        This is a 'list' of elements of the SE(3) group.
    """
    theta = 2 * np.pi * s
    hds = np.identity(4)  # initialize the homogeneous transformation matrix
    position = [
        r * np.cos(theta),
        r * np.sin(theta),
        b + d * r**2 * (np.cos(theta) ** 2 - np.sin(theta) ** 2),
    ]
    hds[:3, 3] = np.array(position)
    angle = np.pi / 6 * np.sin(2 * np.pi * s)
    # angle = theta
    orientation = np.array(
        [
            [1, 0, 0],
            [0, np.cos(angle), np.sin(angle)],
            [0, -np.sin(angle), np.cos(angle)],
        ]
    )
    # axis = np.array([1, 1, 1])
    # axis = axis / np.linalg.norm(axis)
    # skew_mat = SmapSO3(axis)
    # orientation = expSO3(theta * skew_mat)
    # orientation = np.eye(3)
    hds[:3, :3] = orientation
    return hds


def hd_derivative(s, r=1, b=1, d=0.2):
    theta = 2 * np.pi * s
    dhds = np.zeros((4, 4))
    dposition_ds = [
        -r * 2 * np.pi * np.sin(theta),
        r * 2 * np.pi * np.cos(theta),
        d
        * r**2
        * 2
        * (-2 * np.cos(theta) * np.sin(theta) - 2 * np.sin(theta) * np.cos(theta))
        * 2
        * np.pi,
    ]
    dhds[:3, 3] = np.array(dposition_ds)
    angle = np.pi / 6 * np.sin(2 * np.pi * s)
    # angle = theta
    orientation = np.array(
        [
            [1, 0, 0],
            [0, np.cos(angle), np.sin(angle)],
            [0, -np.sin(angle), np.cos(angle)],
        ]
    )
    chain = np.pi / 6 * 2 * np.pi * np.cos(2 * np.pi * s)
    # chain = 2 * np.pi
    dorientation_ds = chain * SmapSO3(np.array([1, 0, 0])) @ orientation
    # axis = np.array([1, 1, 1])
    # axis = axis / np.linalg.norm(axis)
    # dorientation_ds = 2 * np.pi * SmapSO3(axis * theta)
    dhds[:3, :3] = dorientation_ds

    # dhds[:3, :3] = 2 * np.pi * np.array(
    #     [
    #         [0, 0, 0],
    #         [0, -np.sin(theta), np.cos(theta)],
    #         [0, -np.cos(theta), -np.sin(theta)],
    #     ]
    # )
    return dhds


def precomputed_hd(curve_fun, n_points, *args, **kwargs):
    """Function that precomputes the curve for each parameter s.

    Parameters
    ----------
    curve_fun : function
        Function that computes the curve. It must be a function that takes as
        first argument the parameter s, and returns a homogeneous transformation
        matrix.
    n_points : int
        Number of points in the curve.
    *args : list
        Arguments of the curve function.
    **kwargs : dict
        Keyword arguments of the curve function.

    Returns
    -------
    precomputed : np.array
        Array with the precomputed curve. The shape is (n_points, 4, 4).
    """
    s = np.linspace(0, 1, num=n_points)
    precomputed = []
    for si in s:
        precomputed.append(curve_fun(si, *args, **kwargs))
    # precomputed = np.array(precomputed)
    return precomputed


# %%
# Load data
# LOCAL PC
path = "/home/fbartelt/Documents/Projetos/robotics-experiments/omniocta/data"
# SERVER
path = "/home/fbartelt/Projects/robotics-experiments/omniocta/data"

# Get all pickle files in the path
# files = [f for f in os.listdir(path) if f.endswith(".pkl")]
files_to_anim = [
    "pos_0.15789473684210525_ori_0.031578947368421054_seed",
    "pos_0.0_ori_0.0_seed",
    "pos_0.4473684210526315_ori_0.08947368421052632_seed",
    "pos_0.42105263157894735_ori_0.08421052631578947_seed",
]
get_now = 3
files = [
    f
    for f in sorted(os.listdir(path))
    # if f.startswith()
    if f.startswith(files_to_anim[get_now])
]
files = sorted(files, key=lambda x: int(re.findall(r"seed_(\d+)", x)[0]))
print(files)

# File 0 = highest noise full traversal
# File 1 = highest noise failed traversal
# File 2 = medium noise full traversal
seeds_data = {}
if get_now > 1:
    decimation = 8  # Decimate data above 0.42m noise
    final_time = 20 * 60  # seconds (20 minutes)
else:
    decimation = 5  # Decimate data
    final_time = 5 * 60  # seconds (5 minutes)

sim_dt = 1e-2
cutoff = int(final_time / sim_dt)
# cutoff = None
seeds_to_parse = 10
seed_values = []
print(
    f"Animating {seeds_to_parse} seeds with final time {final_time}s and decimation {decimation}..."
)

for file in files[1 : seeds_to_parse + 1]:
    # for file in files:
    seed = re.findall(r"seed_(\d+)", file)[0]
    seed_values.append(int(seed))
    with open(os.path.join(path, file), "rb") as f:
        data = pickle.load(f)
    seeds_data[seed] = {
        "p_hist": data["p_hist"][:cutoff][::decimation],
        "R_hist": data["R_hist"][:cutoff][::decimation],
    }

print(data.keys())

# p_hist = data["p_hist"]
# R_hist = data["R_hist"]

n_points = 5000
r, b, d = 2.5, 1, 0.2
curve = precomputed_hd(hd, n_points, r, b, d)
curve = np.array(curve)


# %%
VISUAL_DISPLACEMENT = np.array([0, 0, 1.5])


def create_uav(
    n_rotors,
    body_radius,
    rotor_radius,
    arm_length,
    htm=np.eye(4),
    body_color="black",
    rotor_color="black",
    arm_color="gray",
    opacity=1.0,
):
    """Create a UAV model with given parameters.

    Parameters
    ----------
    n_rotors : int
        Number of rotors.
    body_radius : float
        Radius of the body.
    rotor_radius : float
        Radius of the rotors.
    arm_length : float
        Length of the arms.

    Returns
    -------
    uav : uaibot.GroupObject
    """

    # Create body
    htm = np.array(htm)
    mesh1 = ub.MeshMaterial(
        color="#2b2b2b",  # dark charcoal gray (nearly black)
        metalness=0.6,  # moderately metallic (composite + metal feel)
        roughness=0.45,  # subtle sheen without mirror reflection
        clearcoat=0.2,  # thin glossy coating typical of polymers
        clearcoat_roughness=0.3,  # slightly diffused highlights
        reflectivity=0.5,  # standard reflectivity for matte composites
        emissive="#000000",  # no self-lighting
        opacity=opacity,  # fully opaque
        side="FrontSide",  # default shading
        flat_shading=False,  # smooth curved surfaces
        ior=1.4,  # typical for carbon/plastic composites
    )
    mesh2 = ub.MeshMaterial(
        color=body_color,  # bright silver-gray base
        metalness=0.7,  # semi-metallic look (not pure metal, avoids blackening)
        roughness=0.35,  # moderate sheen, soft reflections
        clearcoat=0.25,  # adds realistic highlight coating
        clearcoat_roughness=0.5,  # diffuses reflections for "brushed" feel
        reflectivity=0.7,  # brightens material under limited lighting
        emissive="#1a1a1a",  # subtle base emission to prevent over-darkening
        opacity=opacity,
        side="FrontSide",
        flat_shading=False,
        ior=1.35,  # typical for metal–oxide finish
    )
    # mesh1 = ub.MeshMaterial(
    #     color="#808080", metalness=1.0, roughness=0.35, clearcoat=0.15, reflectivity=0.9
    # )
    body = ub.Ball(
        htm=htm,
        radius=body_radius,
        color=body_color,
        opacity=opacity,
        mesh_material=mesh2,
    )
    body_frame = ub.Frame(htm=htm, size=0.4)

    # Create rotors
    rotors = []
    for i in range(n_rotors):
        angle = i * 2 * np.pi / n_rotors
        x = arm_length * np.cos(angle)
        y = arm_length * np.sin(angle)
        # z = htm[2, 3]
        z = 0.0
        htm_body2rotor = pose2htm([x, y, z], np.eye(3))
        htm_rotor = htm @ htm_body2rotor
        rotor = ub.Ball(
            htm=htm_rotor,
            radius=rotor_radius,
            color=rotor_color,
            opacity=opacity,
            mesh_material=mesh2,
        )
        rotors.append(rotor)
    # Create arms
    arms = []
    for i in range(n_rotors):
        angle = i * 2 * np.pi / n_rotors
        x1 = arm_length * np.cos(angle)
        y1 = arm_length * np.sin(angle)
        # z1 = htm[2, 3]
        z1 = 0
        x2 = 0
        y2 = 0
        z2 = 0
        # z2 = htm[2, 3]
        rotz = np.array(ub.Utils.rotz(angle))[:3, :3]
        rotx = np.array(ub.Utils.roty(np.pi / 2))[:3, :3]
        rot = rotz @ rotx
        htm_body2arm = pose2htm([(x1 + x2) / 2, (y1 + y2) / 2, (z1 + z2) / 2], rot)
        htm_arm = htm @ htm_body2arm
        arm = ub.Cylinder(
            htm=htm_arm,
            height=arm_length,
            radius=rotor_radius / 4,
            color=arm_color,
            opacity=opacity,
            mesh_material=mesh2,
        )
        arms.append(arm)

    uav = ub.Group([body, body_frame] + rotors + arms)
    return uav


def create_SE3_curve(
    curve, curve_color="red", curve_width=5, show_frame=True, n_frames=10
):
    curve_ = np.array(curve).copy()
    htm_displace = np.array(ub.Utils.trn(VISUAL_DISPLACEMENT))
    for i in range(curve_.shape[0]):
        curve_[i, :, :] = htm_displace @ curve_[i, :, :]
    n_points = curve_.shape[0]
    positions = curve_[:, :3, 3]
    # positions += VISUAL_DISPLACEMENT
    curve_trace = ub.PointCloud(points=positions.T, color=curve_color, size=curve_width)
    if show_frame:
        frame_indices = np.linspace(0, n_points - 1, n_frames).astype(int)
        frames = []
        for idx in frame_indices:
            htm = curve_[idx]
            # htm[:3, 3] += VISUAL_DISPLACEMENT
            frame = ub.Frame(htm=htm, size=0.3)
            frames.append(frame)
        frames_group = ub.Group(frames)
        return curve_trace, frames_group
    else:
        return curve_trace, None


# N = p_hist_dec.shape[0]

# Sum visual displace to p_hist (in world frame)
htm_displace = np.array(ub.Utils.trn(VISUAL_DISPLACEMENT))
uav_seeds = []
colorscale = px.colors.qualitative.Dark24

for idx, (seed, data) in enumerate(seeds_data.items()):
    print(f"Processing seed {seed}...")
    # Add displacement to each pose
    p_hist_dec, R_hist_dec = data["p_hist"], data["R_hist"]

    for i in range(p_hist_dec.shape[0]):
        htm_ = pose2htm(p_hist_dec[i], R_hist_dec[i])
        htm = htm_displace @ htm_
        p_hist_dec[i] = htm[:3, 3]
        R_hist_dec[i] = htm[:3, :3]

    data["p_hist"] = p_hist_dec
    data["R_hist"] = R_hist_dec
    htm0 = pose2htm(p_hist_dec[0, :].flatten(), R_hist_dec[0, :, :])
    octarotor = create_uav(
        # htm=htm0,
        n_rotors=8,
        body_radius=0.1,
        rotor_radius=0.05,
        arm_length=0.3,
        body_color=colorscale[idx % len(colorscale)],
        rotor_color="gray",
        arm_color="gray",
        opacity=0.9,
    )
    uav_seeds.append(octarotor)

curve_pc, curve_frames = create_SE3_curve(
    curve, curve_color="red", curve_width=0.03, show_frame=True, n_frames=20
)
sim = ub.Simulation.create_sim_lesson(
    uav_seeds + [curve_pc, curve_frames], light_intensity=2.0
)
num_file_samples = seeds_data[str(seed_values[0])]["p_hist"].shape[0]
# final = int(seeds_data[str(seed_values[0])]["p_hist"].shape[0] * 0.3)
final = int(num_file_samples)
print(f"Total samples in file: {num_file_samples}, animating {final} samples...")
dt = 1e-2
if get_now > 1:
    speedup = 5.0
else:
    speedup = 2
print(f"Animation speedup: {speedup}x")
final_anim_time = dt * (final_time / sim_dt) / decimation / speedup
log_msg = f"""Final time: {final_time}s, decimation: {decimation}, sim_dt: {sim_dt}s, anim_dt: {dt}s, speedup: {speedup}x
    Total animation time: {final_anim_time:.2f}s
    Total animation speedup: {final_time / final_anim_time:.1f}x"""

for i, (seed, data_i) in enumerate(seeds_data.items()):
    new_p_hist, new_R_hist = data_i["p_hist"], data_i["R_hist"]
    new_p_hist, new_R_hist = new_p_hist[:final], new_R_hist[:final]
    octarotor = uav_seeds[i]
    print(f"Animating seed {seed}...")
    for j in range(new_p_hist.shape[0]):
        p, R = new_p_hist[j], new_R_hist[j]
        progress_bar(j, final)
        htm = pose2htm(p, R)
        octarotor.add_ani_frame(time=j * dt / speedup, htm=htm)

# Set camera (x, y, z) looking at (xo, yo ,zo) with zoom f
cam0 = [-17, 0, 2.6, 0, 0, 0, 1]
sim.set_parameters(width=800, height=600, pixel_ratio=0.7, camera_start_pose=cam0)
sim.save("./", f"octasim_{get_now}")
print("Saved")
print(log_msg)

# print(dt * (final_time / sim_dt) / decimation / speedup)


# %%
# open browser with html file (neovim workaround)
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


open_in_browser("./octasim.html")
