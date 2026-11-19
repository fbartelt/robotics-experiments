import os
import uaibot as ub
from uaibot.simobjects.curve import CurveSE3
import pickle
import numpy as np
import plotly.graph_objects as go
from scipy.spatial.transform import Rotation as R


def _rotation_from_to(a, b, eps=1e-12):
    """Return the minimal rotation matrix Q such that Q @ a_hat = b_hat."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)

    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)

    v = np.cross(a, b)
    c = np.dot(a, b)
    nv = np.linalg.norm(v)

    if nv < eps:
        if c > 0.0:
            return np.eye(3)

        # a and b are antiparallel: choose any perpendicular axis.
        axis = np.array([1.0, 0.0, 0.0])
        if abs(np.dot(axis, a)) > 0.9:
            axis = np.array([0.0, 1.0, 0.0])
        axis = axis - np.dot(axis, a) * a
        axis /= np.linalg.norm(axis)
        return R.from_rotvec(np.pi * axis).as_matrix()

    axis = v / nv
    angle = np.arctan2(nv, c)
    return R.from_rotvec(axis * angle).as_matrix()


def min_single_rotation_to_gimbal_lock(transforms, eps=1e-12):
    """
    Compute one minimum rotation Q such that, after applying it on the left
    to all transforms,

        T_i_new = Q_h @ T_i

    at least one rotation part becomes gimbal-locked when converted with

        R.from_matrix(T_i_new[:3, :3]).as_euler("xyz").

    Parameters
    ----------
    transforms : (N, 4, 4) ndarray
        Homogeneous transformation matrices.

    Returns
    -------
    Q : (3, 3) ndarray
        Minimum rotation matrix.
    Q_h : (4, 4) ndarray
        Same rotation as a homogeneous matrix. Use T_new = Q_h @ transforms.
    locked_index : int
        Index of the pose that becomes gimbal-locked.
    angle : float
        Minimum rotation angle in radians.
    beta_after : float
        Middle "xyz" Euler angle of the locked pose after applying Q.
        It will be approximately +pi/2 or -pi/2.
    """
    T = np.asarray(transforms, dtype=float)

    if T.ndim != 3 or T.shape[1:] != (4, 4):
        raise ValueError("transforms must have shape (N, 4, 4)")
    if T.shape[0] == 0:
        raise ValueError("transforms must contain at least one matrix")

    rotations = T[:, :3, :3]

    # First column of each rotation matrix: u_i = R_i @ e_x
    u = rotations[:, :, 0]  # shape (N, 3)

    e_z = np.array([0.0, 0.0, 1.0])

    dots = u @ e_z

    # Choose +e_z or -e_z, whichever is closest to u_i.
    signs = np.where(dots < 0.0, -1.0, 1.0)

    # angle to closest of +/- e_z
    cos_angles = np.abs(dots)
    angles = np.arccos(np.clip(cos_angles, -1.0, 1.0))

    locked_index = int(np.argmin(angles))
    angle = float(angles[locked_index])

    target = signs[locked_index] * e_z
    Q = _rotation_from_to(u[locked_index], target, eps=eps)

    Q_h = np.eye(4)
    Q_h[:3, :3] = Q

    # Sanity check: this pose should now be at gimbal lock.
    locked_rotation = Q @ rotations[locked_index]
    beta_after = float(R.from_matrix(locked_rotation).as_euler("xyz")[1])

    return Q, Q_h, locked_index, angle, beta_after


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


# %%
# transforms: np.array of shape (N, 4, 4)
Q, Q_h, idx, angle, beta_after = min_single_rotation_to_gimbal_lock(curve)

# Apply the single rotation to every homogeneous transform.
transforms_rotated = Q_h @ curve

# Convert all to xyz Euler angles.
phi_curve = np.array(
    [
        R.from_matrix(transforms_rotated[i, :3, :3]).as_euler("xyz")
        for i in range(curve.shape[0])
    ]
)

print("locked index:", idx)
print("minimum rotation angle [deg]:", np.degrees(angle))
print("locked pose beta [deg]:", np.degrees(beta_after))


# %%
fig = go.Figure()
for i in range(3):
    fig.add_trace(go.Scatter(y=np.array(phi_curve[:, i])))

fig.show()


# %%
sim = ub.Simulation(width=1480, height=1080)
ball_idx = ub.Ball(htm=transforms_rotated[idx], radius=0.05)
curve_ub = CurveSE3(points=transforms_rotated, size=0.05, color='cyan')
curve_ub2 = CurveSE3(points=curve, size=0.05, color='magenta')
sim.add([curve_ub, curve_ub2, ball_idx])
sim.run_in_browser()
