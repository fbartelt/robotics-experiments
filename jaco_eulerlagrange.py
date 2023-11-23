# %%
import sympy as sp
from IPython.display import display, clear_output, Latex
import numpy as np
from numpy import pi
from functools import reduce


def translation_htm(vector: sp.Matrix) -> sp.Matrix:
    """Given a translation vector `vector`, returns its respective homogeneous
    transformation matrix.

    Parameters
    ----------
    vector: sp.Matrix
        The translation vector
    """
    return sp.eye(4, 3).col_insert(4, vector.row_insert(4, sp.Matrix([[1]])))


def rotation_htm(rotation_matrx: sp.Matrix) -> sp.Matrix:
    """Given a rotation matrix `rotation_matrx`, returns its respective
    homogeneous transformation matrix.

    Parameters
    ----------
    rotation_matrx: sp.Matrix
        The rotation matrix
    """
    return rotation_matrx.row_insert(4, sp.zeros(1, 3)).col_insert(
        4, sp.Matrix([[0], [0], [0], [1]])
    )


def dht_htms(
    theta_list: list,
    d_list: list,
    alpha_list: list,
    a_list: list,
) -> list[sp.Matrix]:
    """Returns the Denavit-Hartenberg (DH) Homogeneous Transformation
    Matrices.

    The DHT parameters can be sp.Symbol or float. If they are symbolic,
    for rotative joints, `theta_list` element must be evaluated, i.e.
    theta_list=[theta(t),...] if t is the time symbol. Respectively for
    prismatic joints and elements of `d_list`.

    $H_{i}^{i-1} = Trn_z{d_i}Rot_z{\\theta_i}Trn_x{a_i}Rot_x{\\alpha_i}$.

    Parameters
    ----------
    theta_list: list
        Rotation around each z-axis
    d_list: list
        Translation in each z-axis
    alpha_list: list
        Rotation around each x-axis
    a_list: list
        Translation in each x-axis
    """
    htms = []
    for i, theta in enumerate(theta_list):
        rotz = sp.rot_ccw_axis3(theta)
        rotx = sp.rot_ccw_axis1(alpha_list[i])
        d = sp.Matrix([[0], [0], [d_list[i]]])
        a = sp.Matrix([[a_list[i]], [0], [0]])
        htm = translation_htm(d) @ rotation_htm(rotz)
        htm = htm @ translation_htm(a) @ rotation_htm(rotx)
        htms.append(htm)
    return htms


def fkm(
    htm_list: list[sp.Matrix],
    axis: str = "dh",
    com_coords: list | None = None,
    tolerance_nsimp: float = 1e-6,
) -> list[sp.Matrix] | sp.Matrix:
    """Compute the forward kinematics (FK) for an `axis` in world frame.

    If `axis`='dh', the function returns the FK for each DH frame.
    If `axis`='com', the function returns the FK for each center of
    mass (`com_coords` must be provided).
    If `axis`='eef', the function returns the FK for the last DH frame.

    Parameters
    ----------
    htm_list: list[sp.Matrix]
        An ordered list with each DH HTM.
    axis: str
        Reference axis for FK.
    com_coords: list
        A list of each center of mass position in respect to its
        respective DH frame.
    tolerance_nsimp: float
        Tolerance parameter for `sp.nsimplify` (used to reduce
        computational effort).
    """
    new_htm = []
    match axis:
        case "dh":
            new_htm = [
                sp.trigsimp(
                    reduce(
                        lambda x, y: fracsimp(x @ y, tolerance_nsimp),
                        htm_list[: i + 1],
                        sp.eye(4),
                    )
                )
                for i, _ in enumerate(htm_list)
            ]
        case "com":
            htm_old = sp.eye(4)
            if com_coords is None:
                com_coords = [sp.zeros(3, 1) for _ in range(len(htm_list))]
            for i, htm in enumerate(htm_list):
                htm_ = htm_list[i].as_mutable()
                displacement = (htm_[:3, :3] @ com_coords[i]).row_insert(
                    4, sp.Matrix([[0]])
                )
                htm_ = htm_ + sp.zeros(4, 3).col_insert(4, displacement)
                htm_ = fracsimp(sp.trigsimp(htm_old @ htm_), tolerance_nsimp)
                new_htm.append(htm_)
                htm_old = htm
        case "eef":
            new_htm = sp.trigsimp(reduce(lambda x, y: x @ y, htm_list, sp.eye(4)))
        case _:
            raise ValueError(f'{axis} is not a valid axis. Use "dh", "com" or "eef"')

    return new_htm


def get_W(
    q: list | sp.Matrix,
    htm_list: list,
    fkm_list: list | None = None,
) -> sp.Matrix:
    """Returns the orientation jaocbian.

    W is given as $\omega = W \dot{\omega}$, where $\omega$ is the
    angular velocity.

    Parameters
    ----------
    q: list | sp.Matrix
        The configuration vector.
    htm_list: list[sp.Matrix]
        An ordered list with each DH HTM.
    fkm_list: list | None
        The forward kinematics map. If `None`, then it is calculated
        by `fkm()`
    """
    W_list = []
    if fkm_list is None:
        htm_list = fkm(htm_list)
    else:
        htm_list = fkm_list
    for htm in htm_list:
        rot = htm[:3, :3]
        rx, ry, rz = rot[:, 0], rot[:, 1], rot[:, 2]
        dry, drz = ry.jacobian(q), rz.jacobian(q)
        W = sp.Matrix([[-ry.T @ drz], [rx.T @ drz], [-rx.T @ dry]])
        W_list.append(sp.trigsimp(W))
    return W_list


def steiner_theorem(
    inertia: sp.Matrix,
    displacement: list | sp.Matrix,
    mass: float,
    reverse: bool = False,
) -> sp.Matrix:
    """Applies Steiner Theorem to `inertia` and returns the result.

    Default behavior is to return the inertia tensor about an arbitrary
    axis given the inertia about the center of mass `inertia`, a
    displacement vector from one axis to another and the mass.

    If `reverse=True` it means that the provided `inertia` tensor is
    about an arbitrary axis and the expected return is Icm, the
    inertia about the center of mass.

    Parameters
    ----------
    inertia: sp.Matrix
        A constant inertia tensor.
    displacement: list | sp.Matrix
        The perpendicular distance between both rotation axes.
    mass: float
        The element mass.
    reverse: bool
        Indicates wheter the provided `inertia` was calculated about
        the center of mass or an arbitrary axis. If `True`, the Icm is
        returned, if `False`, an arbitrary I is returned.
    """
    displacement = sp.Matrix(displacement)
    Inew = inertia + (-1) ** (reverse) * mass * (
        displacement.dot(displacement) * sp.eye(3) - displacement @ displacement.T
    )
    return Inew


def christoffel(M: sp.Matrix, q: list | sp.Matrix) -> dict:
    """Returns the 1st kind Christoffel symbols Cijk as a dict.

    The dict keys are strings composed of the ijk indices (1-based).
    E.g. To access C123, do `cijk['123']`

    Parameters
    ----------
    q: list | sp.Matrix
        The configuration vector.
    M: sp.Matrix
        Mass (inertia) matrix
    """
    n = len(q)
    cijk = {}
    for k in range(n):
        for i in range(n):
            for j in range(0, i + 1):
                c = sp.simplify(
                    0.5 * (M[k, j].diff(q[i]) + M[k, i].diff(q[j]) - M[i, j].diff(q[k]))
                )
                cijk[f"{i+1}{j+1}{k+1}"] = c
                cijk[f"{j+1}{i+1}{k+1}"] = c
    return cijk


def fracsimp(M: sp.Matrix, tol=1e-6) -> sp.Matrix:
    return sp.nsimplify(M.n(), tolerance=tol)


def dynamic_model(
    q: list | sp.Matrix,
    htm_list: list[sp.Matrix],
    mass_list: list,
    inertia_list: list,
    com_coords: list,
    height_axis: int = 2,
    gravity: float | sp.Symbol = 9.81,
    tolerance_nsimp: float = 1e-6,
) -> list[sp.Matrix]:
    """Computes the system's dynamic model as the matrices of the
    canonical equation $M(q)\ddot{q}+C(q,\dot{q})\dot{q}+G(q)$.

    Parameters
    ----------
    q: list | sp.Matrix
        The configuration vector.
    htm_list: list[sp.Matrix]
        An ordered list with each DH HTM.
    mass_list: list
        A list of each link mass.
    inertia_list: list
        A list with each inertia tensor about the respective DH frame.
    com_coords: list
        A list of each center of mass position in respect to its
        respective DH frame.
    height_axis: int
        The axis number for gravity, defaults to 2 (z-axis).
    gravity: float
        The gravity acceleration.
    tolerance_nsimp: float
        Tolerance parameter for `sp.nsimplify` (used to reduce
        computational effort).

    Returns
    -------
    M: sp.Matrix
        Mass (inertia) matrix
    C: sp.Matrix
        Coriolis matrix
    G: sp.Matrix
        Gravity vector
    """
    n = len(q)
    M = sp.ZeroMatrix(n, n)
    G = sp.ZeroMatrix(n, 1)
    htm_list_ = htm_list
    htm_list = fkm(htm_list, axis="com", com_coords=com_coords)
    W_list = get_W(q, htm_list_, fkm_list=htm_list)

    # Get M and G
    for i, htm in enumerate(htm_list):
        p = sp.eye(3, 4) @ htm @ sp.Matrix([[0], [0], [0], [1]])
        Jp = p.jacobian(q)
        inertia = steiner_theorem(
            inertia_list[i], com_coords[i], mass_list[i], reverse=True
        )
        M += mass_list[i] * Jp.T @ Jp + W_list[i].T @ inertia @ W_list[i]
        M = fracsimp(M, tolerance_nsimp)
        grad = ((htm @ sp.zeros(3, 1).row_insert(4, sp.Matrix([[1]]))).jacobian(q))[
            height_axis, :
        ]
        G += sp.trigsimp(mass_list[i] * gravity * grad.T)
        G = fracsimp(G, tolerance_nsimp)

    M = fracsimp(sp.simplify(M), tolerance_nsimp)
    G = fracsimp(sp.simplify(G), tolerance_nsimp)

    # Get C
    cijk = christoffel(M, q)
    C = sp.zeros(n, n)
    for j in range(n):
        for k in range(n):
            gamma_kj = reduce(
                lambda x, y: x + y,
                [cijk[f"{i+1}{j+1}{k+1}"] * q[i].diff() for i in range(n)],
                0,
            )
            print(gamma_kj)
            C[k, j] = gamma_kj

    C = fracsimp(sp.simplify(C), tolerance_nsimp)
    
    return M, C, G


# Time
t = sp.Symbol("t", real=True)

# Gravity acceleration
g0 = sp.Symbol("g_0", real=True)

# Generalized force
tau = sp.Symbol("tau", real=True)

# Link masses
m1 = sp.Symbol("m_1", real=True)
m2 = sp.Symbol("m_2", real=True)
m3 = sp.Symbol("m_3", real=True)
m4 = sp.Symbol("m_4", real=True)
m5 = sp.Symbol("m_5", real=True)
m6 = sp.Symbol("m_6", real=True)

# Links angle
theta1 = sp.Function("theta_1", real=True)  # Shoulder
theta2 = sp.Function("theta_2", real=True)  # Upperarm
theta3 = sp.Function("theta_3", real=True)  # Forearm
theta4 = sp.Function("theta_4", real=True)  # Wrist1
theta5 = sp.Function("theta_5", real=True)  # Wrist2
theta6 = sp.Function("theta_6", real=True)  # Gripper

# %%
"""Jaco dynamic model"""
# Configuration variables vector
q = sp.Matrix(
    [[theta1(t)], [theta2(t)], [theta3(t)], [theta4(t)], [theta5(t)], [theta6(t)]]
)

# Links DHT
d1 = 0.2755
d2 = 0.41
d3 = 0.2073
d4 = 0.0741
d5 = 0.0741
d6 = 0.16
e2 = 0.0098
aa = 30 * pi / 180
sa = np.sin(aa)
s2a = np.sin(2 * aa)
d4b = d3 + (sa / s2a) * d4
d5b = (sa / s2a) * d4 + (sa / s2a) * d5
d6b = (sa / s2a) * d5 + d6
alpha1 = pi / 2
alpha2 = pi
alpha3 = pi / 2
alpha4 = 2 * aa
alpha5 = 2 * aa
alpha6 = pi

theta_list = q
d_list = [d1, 0, -e2, -d4b, -d5b, -d6b]
a_list = [0, d2, 0, 0, 0, 0]
alpha_list = [pi / 2, pi, pi / 2, 2 * aa, 2 * aa, pi]

com_coords = [
    sp.Matrix([[1.62075358e-05], [-8.68912100e-03], [-3.11506292e-03]]),
    sp.Matrix([[-6.19272071e-02], [-1.76325490e-05], [5.92762000e-03]]),
    sp.Matrix([[-0.00092673], [-0.00182087], [-0.17186498]]),
    sp.Matrix([[-8.09900000e-05], [1.15602217e-03], [-3.79405131e-02]]),
    sp.Matrix([[7.76000000e-05], [1.03514396e-03], [-3.78727272e-02]]),
    sp.Matrix([[-0.0064991], [-0.00185375], [-0.07984141]]),
]
mass_list = [
    0.740185285501933,
    0.8489361778912677,
    0.48326882919212083,
    0.43217459270218916,
    0.43217590992539645,
    0.6208313337899377,
]

# Inertia as Icm
inertia_list = [
    sp.Matrix(
        [
            [1.066e-03, 0.000e00, 3.800e-05],
            [0.000e00, 1.038e-03, -0.000e00],
            [3.800e-05, -0.000e00, 5.400e-04],
        ]
    ),
    sp.Matrix(
        [
            [0.014208, -0.000403, -0.000865],
            [-0.000403, 0.011765, -0.005182],
            [-0.000865, -0.005182, 0.003068],
        ]
    ),
    sp.Matrix([[0.002, -0.0, -0.0], [-0.0, 0.001, -0.001], [-0.0, -0.001, 0.002]]),
    sp.Matrix(
        [
            [3.03e-04, 0.00e00, -0.00e00],
            [0.00e00, 3.05e-04, -3.00e-06],
            [-0.00e00, -3.00e-06, 1.97e-04],
        ]
    ),
    sp.Matrix(
        [
            [3.03e-04, -0.00e00, -0.00e00],
            [-0.00e00, 2.64e-04, -5.20e-05],
            [-0.00e00, -5.20e-05, 2.39e-04],
        ]
    ),
    sp.Matrix(
        [
            [1.731e-03, -5.300e-05, -6.400e-05],
            [-5.300e-05, 1.822e-03, -3.500e-05],
            [-6.400e-05, -3.500e-05, 1.079e-03],
        ]
    ),
]

inertia_list = [
    steiner_theorem(i, c, m, reverse=False)
    for i, c, m in zip(inertia_list, com_coords, mass_list)
]
# (sp.nsimplify((sp.nsimplify(dht[0]@dht[1], tolerance=1e-6)).inv(), tolerance=1e-6) @ sp.Matrix([-0.00592762,  0.14709695,  0.5909634, 1])).subs(theta1(t), pi).subs(theta2(t), pi)
dht = dht_htms(theta_list, d_list, alpha_list, a_list)
M, C, G = dynamic_model(
    q,
    dht,
    mass_list,
    inertia_list,
    com_coords,
    height_axis=2,
    gravity=9.81,
    tolerance_nsimp=1e-6,
)
# M, C, G = dynamic_model(q[:3], dht[:3], mass_list[:3], inertia_list[:3], com_coords[:3], 2, 9.81)
# sp.nsimplify(sp.separatevars(M, symbols=q[:3]).n(6), tolerance=1e-6)
# %%
"""Spong(2020) Planar Elbow Manipulator (Cap 6.4, P. 186)"""
theta_list = [theta1(t), theta2(t)]
d_list = [0, 0, 0]
alpha_list = [0, 0, 0]
a1, a2 = sp.Symbol("l1", real=True), sp.Symbol("l2", real=True)  # link length
l1, l2 = sp.Symbol("l_{c1}", real=True), sp.Symbol(
    "l_{c2}", real=True
)  # distance to COM
a_list = [a1, a2]

htms = dht_htms(theta_list, d_list, alpha_list, a_list)
com_coords = [sp.Matrix([[l1 - a1], [0], [0]]), sp.Matrix([[l2 - a2], [0], [0]])]
mass_list = [m1, m2]

Ixx1 = sp.Symbol("I_{xx1}", real=True)
Iyy1 = sp.Symbol("I_{yy1}", real=True)
Izz1 = sp.Symbol("I_{zz1}", real=True)
Ixy1 = sp.Symbol("I_{xy1}", real=True)
Ixz1 = sp.Symbol("I_{xz1}", real=True)
Iyz1 = sp.Symbol("I_{yz1}", real=True)
Ixx2 = sp.Symbol("I_{xx2}", real=True)
Iyy2 = sp.Symbol("I_{yy2}", real=True)
Izz2 = sp.Symbol("I_{zz2}", real=True)
Ixy2 = sp.Symbol("I_{xy2}", real=True)
Ixz2 = sp.Symbol("I_{xz2}", real=True)
Iyz2 = sp.Symbol("I_{yz2}", real=True)

# Inertias in respect to COM
inertia1 = sp.Matrix([[Ixx1, -Ixy1, -Ixz1], [Ixy1, Iyy1, -Iyz1], [Ixz1, Iyz1, Izz1]])
inertia2 = sp.Matrix([[Ixx2, -Ixy2, -Ixz2], [Ixy2, Iyy2, -Iyz2], [Ixz2, Iyz2, Izz2]])

# Inertias in respect to DH
inertia1 = steiner_theorem(inertia1, com_coords[0], mass_list[0], reverse=False)
inertia2 = steiner_theorem(inertia2, com_coords[1], mass_list[1], reverse=False)

W_list = get_W(theta_list, htms)
inertia_list = [inertia1, inertia2]

M, C, G = dynamic_model(
    theta_list,
    htms,
    mass_list,
    inertia_list,
    com_coords,
    height_axis=1,
    gravity=sp.Symbol("g_0"),
)
display(Latex("Mass Matrix $M(q)$:"), M)
display(Latex("Coriolis Matrix $C(q, \dot{q}):$"), C)
display(Latex("Gravity vector $G(q)$:"), G)

# %%
