# %%
import pickle
import time
import uaibot as ub
import numpy as np
import sys
from adaptive_cpp import AdaptiveController, ControlLoop
from uaibot_cpp_bind import expSO3, SmapSO3, SmapSE3, expSE3
from uaibot.utils import Utils
from scipy.linalg import block_diag
from vfutils import vector_field_plot

# Sytem
rho = 8050.0  # Density of steel in kg/m^3
r = 15e-2  # Radius of the cylinder in meters
h = 10e-2  # Height of the cylinder in meters
r = 0.25
h = 1.0
m = rho * np.pi * r**2 * h  # Mass of the cylinder in Kg

# Box
l, w, h = 0.5, 0.3, 0.2 # y, x, z
m = 10.0
print(m)


rng = np.random.default_rng(42)
# r_p = np.array([0.0, 0.0, h / 2.0])  # Measurement point
# N = 6  # Number of agents
N = 2
# Initial positions of the agents
# r_i = np.array([[0, 0, h / 2], [0, 0, -h / 2], [r, 0, 0], [-r, 0, 0], [0, r, 0], [0, -r, 0]])
# Radially distributed around the object
# angle_dist = np.linspace(0, 2 * np.pi, N, endpoint=False)
# r_i = np.array([[r * np.cos(angle), r * np.sin(angle), h / 2.0] for angle in angle_dist])
r_i = np.array(
    [
        [0.15, 0.25, -0.1],
        [-0.15, -0.25, 0.1],
    ]
)
r_p = r_i[0]
print(f"Agents distributed as: {r_i}")

# # Inertia tensor (Cilinder)
# I_cm = (1.0 / 12.0) * np.eye(3)
# I_cm[0, 0] *= m * (3 * r**2 + h**2)
# I_cm[1, 1] *= m * (3 * r**2 + h**2)
# I_cm[2, 2] *= 6 * m * r**2
# I_p = np.array(I_cm - m * Utils.S(r_p) @ Utils.S(r_p))

# Inertia tensor (Box)
I_cm = (1.0 / 12.0) * np.eye(3)
I_cm[0, 0] *= m * (w**2 + h**2)
I_cm[1, 1] *= m * (l**2 + h**2)
I_cm[2, 2] *= m * (l**2 + w**2)
I_p = np.array(I_cm - m * Utils.S(r_p) @ Utils.S(r_p))


mean_a, std_a = 0.0, 1.0
o_hat = [rng.normal(mean_a, std_a, (10,)) for _ in range(N)]
o_true = np.array(
    [
        m,
        0,
        0,
        0,  # m * r_p
        I_p[0, 0],
        I_p[0, 1],
        I_p[0, 2],
        I_p[1, 1],
        I_p[1, 2],
        I_p[2, 2],
    ]
)
o_true[1:4] = m * r_p
o_i = o_true / N

mean_r, std_r = mean_a, 2 * std_a
r_hat = [rng.normal(mean_r, std_r, (3,)) for _ in range(N)]

k1 = 35e-1 / 500
# k1 = 35e-3
K_adap = k1 * block_diag(20e3 / N * np.eye(3), 25e3 / N * np.eye(3))


abs_o_i = np.abs(o_i) + 1e-2 * np.ones(o_i.shape)
kgo, kgr = 3e1, 3e3
factor_ = 1.0
Gamma_o_inv = factor_ * kgo * np.linalg.inv(np.diag(abs_o_i.flatten()))  # 3e1
Gamma_r_inv = factor_ * kgr * np.eye(3)  # 3e3

print(type(Gamma_o_inv), Gamma_o_inv.shape)
print(type(Gamma_r_inv), Gamma_r_inv.shape)
print(type(K_adap), K_adap.shape)
print(type(o_i), o_i.shape)
print(type(r_i), len(r_i), len(r_i[0]))
print(type(I_p), I_p.shape)
print(type(m), type(r_hat))
print(type(r_hat), len(r_hat), r_hat[0].shape)
adaptiveSys = AdaptiveController(
    Gamma_o_inv,
    Gamma_r_inv,
    K_adap,
    o_i,
    r_i,
    I_p,
    m,
    N,
    r_p,
)


# %%
# Kinematic Controller
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
    # position = [
    #     r * (np.sin(theta) + 2 * np.sin(2 * theta)),
    #     r * (np.cos(theta) - 2 * np.cos(2 * theta)),
    #     b + r * (-np.sin(3 * theta)),
    # ]
    #
    hds[:3, 3] = np.array(position)
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

    angle = np.pi / 6 * np.sin(2 * np.pi * s)
    # Rotate along tangent vector
    orientation = expSO3(
        SmapSO3(np.array(dposition_ds) / (np.linalg.norm(dposition_ds) + 1e-6)) * angle
    )
    # angle = theta
    # orientation = np.array(
    #     [
    #         [1, 0, 0],
    #         [0, np.cos(angle), np.sin(angle)],
    #         [0, -np.sin(angle), np.cos(angle)],
    #     ]
    # )
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

    # dposition_ds = [
    #     r * 2 * np.pi * (np.cos(theta) + 2 * 2 * np.cos(2 * theta)),
    #     r * 2 * np.pi * (-np.sin(theta) + 2 * 2 * np.sin(2 * theta)),
    #     r * 2 * np.pi * (-3 * np.cos(3 * theta)),
    # ]
    #
    dhds[:3, 3] = np.array(dposition_ds)
    angle = np.pi / 6 * np.sin(2 * np.pi * s)
    # angle = theta
    # orientation = np.array(
    #     [
    #         [1, 0, 0],
    #         [0, np.cos(angle), np.sin(angle)],
    #         [0, -np.sin(angle), np.cos(angle)],
    #     ]
    # )
    chain = np.pi / 6 * 2 * np.pi * np.cos(2 * np.pi * s)
    # # chain = 2 * np.pi
    dorientation_ds = (
        SmapSO3(np.array(dposition_ds) / (np.linalg.norm(dposition_ds) + 1e-6))
        * angle
        * chain
    )
    # dorientation_ds = chain * SmapSO3(np.array([1, 0, 0])) @ orientation
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


def pose2htm(p, R):
    """Homogeneous transformation matrix from position and rotation."""
    p = np.array(p)
    htm = np.eye(4)
    htm[0:3, 0:3] = R
    htm[0:3, 3] = p.ravel()
    return htm


def progress_bar(i, imax):
    """Prints a progress bar in the terminal.

    Parameters
    ----------
    i : int
        Current iteration.
    imax : int
        Maximum number of iterations.
    """
    sys.stdout.write("\r")
    sys.stdout.write(
        "[%-20s] %d%%" % ("=" * round(20 * i / (imax - 1)), round(100 * i / (imax - 1)))
    )
    sys.stdout.flush()


# Initial conditions
n_points = 3000
n_points = 2000
r, b, d = 0.35, 0.3, 0.2
# r, b = 0.7, 0.4
curve = precomputed_hd(hd, n_points, r, b, d)
curve_derivative = precomputed_hd(hd_derivative, n_points, r, b, d)
p0 = np.array([-0.1, 0, 0.2]).reshape(-1, 1)
# p0 = curve[0, :3, 3].reshape(-1, 1)  # Start at beginning of curve
p = p0.copy()
# R0 = curve[0, :3, :3]  # Start with orientation of curve
R0 = np.eye(3)  # Start with no rotation
R = R0.copy()
htm0 = pose2htm(p0, R0)
htm = htm0.copy()

xi = np.zeros((6, 1))
v, omega = xi[0:3], xi[3:6]

kn1, kn2 = 1.0, 1.0
kt1, kt2, kt3 = 1.0, 1, 1.0

# %%
# Simulation
dt = 1e-3
T = 5.0#30.0
delta = 1e-3
ds = 1e-3
deadband = 0.01
n_steps = int(T / dt)

loop = ControlLoop(
    htm0,
    xi,
    xi.copy(),
    curve,
    curve_derivative,
    adaptiveSys,
    o_hat,
    r_hat,
    kt1,
    kt2,
    kt3,
    kn1,
    kn2,
    delta,
    ds,
    deadband,
    dt,
    T,
)
loop.simulate()

print(loop.min_distances[-10:])
print(loop.closest_indexes[-10:])
print(loop.closest_indexes[::1000])
with open("data_adaptive.pkl", "wb") as f:
    data = {
        "H_hist": loop.H_hist,
        "xi_hist": loop.xi_hist,
        "xi_dot_hist": loop.xi_dot_hist,
        "psi_hist": loop.psi_hist,
        "o_hat_hist": loop.o_hat_hist,
        "r_hat_hist": loop.r_hat_hist,
        "closest_indexes": loop.closest_indexes,
        "min_distances": loop.min_distances,
        "zeta_hist": loop.zeta_hist,
        "taui_hist": adaptiveSys.taui_hist,
        "aprox_hist": adaptiveSys.aprox_hist,
        "input_hist": adaptiveSys.input_hist,
        "r_i": adaptiveSys.r_i,
    }
    pickle.dump(data, f)
print("Data saved to data_adaptive.pkl")

# %%
# Animation
box = ub.Box(htm0, width=w, height=h, depth=l, color="cyan", opacity=0.5)
htm_base_1 = pose2htm([r_i[0, 0], r_i[0, 1], 0], np.eye(3))
agent1 = ub.Robot.create_kinova_gen3(htm=htm_base_1)
agent1._joint_limit = np.matrix(
    [
        [-10 * np.pi, 10 * np.pi],
        [-10 * np.pi, 10 * np.pi],
        [-10 * np.pi, 10 * np.pi],
        [-10 * np.pi, 10 * np.pi],
        [-10 * np.pi, 10 * np.pi],
        [-10 * np.pi, 10 * np.pi],
        [-10 * np.pi, 10 * np.pi],
    ]
)
grip1_htm = htm0 @ pose2htm(r_i[0], np.eye(3)) # in box frame
q0_1 = np.array(agent1.ikm(
    htm_tg=grip1_htm,
    q0=np.zeros((7, 1)),
    ignore_orientation=True,
    check_joint=False,
)).reshape(-1, 1)
agent1.set_ani_frame(q0_1)
htm_base_2 = pose2htm([r_i[1, 0], r_i[1, 1], 0], np.eye(3))
agent2 = ub.Robot.create_kinova_gen3(htm=htm_base_2)
agent2._joint_limit = agent1._joint_limit
grip2_htm = htm0 @ pose2htm(r_i[1], np.eye(3)) # in box frame
q0_2 = np.array(agent2.ikm(
    htm_tg=grip2_htm,
    q0=np.zeros((7, 1)),
    ignore_orientation=True,
    check_joint=False,
)).reshape(-1, 1)
agent2.set_ani_frame(q0_2)

grip1 = ub.Frame(htm=grip1_htm)
grip2 = ub.Frame(htm=grip2_htm)
curve_pos = np.array([c[:3, 3] for c in curve])
pointcloud_curve = ub.PointCloud(points=curve_pos.T, color="cyan", size=0.03)
sim = ub.Simulation.create_sim_grid([agent1, agent2, box, pointcloud_curve, grip1, grip2])

# Integrate manipulators + add box animation
q1, q1_dot, q1_ddot = q0_1.copy(), np.zeros((7, 1)), np.zeros((7, 1))
q2, q2_dot, q2_ddot = q0_2.copy(), np.zeros((7, 1)), np.zeros((7, 1))

wrench_hist = loop.controller.taui_hist

for i in range(n_steps):
    progress_bar(i, n_steps)
    M1, C1_, g1 = agent1.dynamic_model(q1, q1_dot)
    jac1, *_ = agent1.jac_geo(q=q1)
    M2, C2_, g2 = agent2.dynamic_model(q2, q2_dot)
    jac2, *_ = agent2.jac_geo(q=q2)
    ith_wrenches = wrench_hist[i]
    wrench1_world = ith_wrenches[0].reshape(-1, 1)
    wrench2_world = ith_wrenches[1].reshape(-1, 1)
    # Transform wrenches to local frame of the end-effector
    # print(wrench1_world.shape)
    # print(jac1.shape)
    wrench1 = jac1.T @ wrench1_world
    wrench2 = jac2.T @ wrench2_world
    q1_ddot = np.linalg.inv(M1) @ (wrench1 - C1_ - g1)
    q2_ddot = np.linalg.inv(M2) @ (wrench2 - C2_ - g2)
    q1_dot += q1_ddot * dt
    q1 += q1_dot * dt
    q2_dot += q2_ddot * dt
    q2 += q2_dot * dt
    print(q1_dot.ravel(), q2_dot.ravel())
    print(q1.ravel(), q2.ravel())
    # If any qdot or q is nan, break
    if np.any(np.isnan(q1)) or np.any(np.isnan(q1_dot)) or np.any(np.isnan(q2)) or np.any(
        np.isnan(q2_dot)
    ):
        print("NaN detected, stopping simulation")
        break
    # pause for 0.5s
    # time.sleep(0.5)
    agent1.add_ani_frame(time=i * dt, q=q1.ravel())
    agent2.add_ani_frame(time=i * dt, q=q2.ravel())
    box.add_ani_frame(time=i * dt, htm=loop.H_hist[i])

sim.save("./", "adaptive_anim")
