# %%
import uaibot as ub
from uaibot.utils import Utils
import numpy as np
import uaibot_cpp_bind as ub_cpp

print("module file:", ub_cpp.__file__)
print("class obj:", ub_cpp.CPP_Manipulator)  # prints the pybind11 class wrapper
print("dir(class):", [x for x in dir(ub_cpp.CPP_Manipulator) if "gravity" in x.lower()])
# show visible names on the class
print("class dir (sample):", dir(ub_cpp.CPP_Manipulator)[:80])

# Create instance and inspect
inst = ub_cpp.CPP_Manipulator(6)
print("hasattr:", hasattr(inst, "gravity_vector"))
print("dir(instance):", [x for x in dir(inst) if "gravity" in x.lower()])


# %%
# Jonatan origianl
def GetLinkMasses(robot):
    list_mass = [1.377, 1.1636, 1.1636, 0.930, 0.678, 0.678, 0.500]
    list_mass = [link.mass for link in robot.links]
    return list_mass


def GetCmPos(robot):
    # cmpos = [-0.5, 0 , 0]
    cmpos = [link.center_of_mass.flatten() for link in robot.links]
    # htms = robot.fkm(axis="dh")
    # for i in range(len(cmpos)):
    #     cmpos_aug = np.hstack((cmpos[i], 1)).reshape(4, 1)
    #     cmpos[i] = (np.array(Utils.inv_htm(htms[i])) @ cmpos_aug).flatten()[:3]
        # cmpos[i] = (htms[i][:3, :3].T @ cmpos[i].reshape(3, 1)).flatten()
    return cmpos


def create_inertia(Ixx, Ixy, Ixz, Iyy, Iyz, Izz):
    return np.array([[Ixx, Ixy, Ixz], [Ixy, Iyy, Iyz], [Ixz, Iyz, Izz]])


def GetInertia(robot):

    # list_inertia_mat_ = [
    #       create_inertia(0.004570, 0.000001, 0.000002, 0.004831, 0.000448, 0.001409),
    #       create_inertia(0.011088, 0.000005, 0.000000, 0.001072, -0.000691, 0.011255),
    #       create_inertia(0.010932, 0.000000, -0.000007, 0.011127, 0.000606, 0.001043),
    #       create_inertia(0.008147, -0.000001, 0.000000, 0.000631, -0.000500, 0.008316),
    #       create_inertia(0.001596, 0.000000, 0.000000, 0.001607, 0.000256, 0.000399),
    #       create_inertia(0.001641, 0.000000, 0.000000, 0.000410, -0.000278, 0.001641),
    #     #  create_inertia(0.000587, 0.000003, 0.000003, 0.000369, 0.000118, 0.000609),
    #   ]
    list_inertia_mat_ = [link.inertia_tensor for link in robot.links]
    return list_inertia_mat_


def GetJointType(robot):
    list = [0, 0, 0, 0, 0, 0, 0]
    return list


def Newton_Euler(q, qp, qpp, g0, robot):
    n = len(q)

    # Extract mass and inertia
    #    m = np.array([robot.links[i].m for i in range(n)])
    #    I = np.array([robot.links[i].I for i in range(n)])  # shape: (n, 3, 3)
    m = GetLinkMasses(robot)
    I = GetInertia(robot)
    coms_ = GetCmPos(robot)

    # Pre-allocate
    w_i = np.zeros((3, n))
    wp_i = np.zeros((3, n))
    a_i = np.zeros((3, n))
    r_i_im1_i = np.zeros((3, n))
    r_i_i_cmi = np.zeros((3, n))
    a_ci = np.zeros((3, n))

    # Initial values
    w_i0 = np.zeros((3, 1))
    wp_i0 = np.zeros((3, 1))
    a_i0 = np.zeros((3, 1))

    # Homgeneous transformation matrices
    T = robot.fkm(axis="dh")

    for i in range(n):
        if i == 0:
            R_im1_0 = np.eye(3)
            r_0_im1 = np.zeros((3, 1))
            z_im1_im1 = np.array([[0], [0], [1]])
        else:
            R_im1_0 = T[i - 1][:3, :3]
            r_0_im1 = T[i - 1][:3, 3].reshape(3, 1)
            z_im1_im1 = T[i - 1][:3, :3].T @ T[i - 1][:3, 2].reshape(3, 1)

        R_i_im1 = T[i][:3, :3].T @ R_im1_0

        w_i[:, i : i + 1] = R_i_im1 @ (w_i0 + qp[i] * z_im1_im1)
        wp_i[:, i : i + 1] = R_i_im1 @ (
            wp_i0
            + qpp[i] * z_im1_im1
            + np.cross(qp[i] * w_i0.flatten(), z_im1_im1.flatten()).reshape(3, 1)
        )

        r_i_im1_i[:, i] = (
            T[i][:3, :3].T @ (T[i][:3, 3].reshape(3, 1) - r_0_im1)
        ).flatten()

        a_i[:, i : i + 1] = (
            R_i_im1 @ a_i0
            + np.cross(wp_i[:, i], r_i_im1_i[:, i]).reshape(3, 1)
            + np.cross(w_i[:, i], np.cross(w_i[:, i], r_i_im1_i[:, i])).reshape(3, 1)
        )

        # r_i_i_cmi[:, i] = robot.links[i].r.flatten()
        r_i_i_cmi[:, i] = coms_[i]

        a_ci[:, i : i + 1] = (
            a_i[:, i : i + 1]
            + np.cross(wp_i[:, i], r_i_i_cmi[:, i]).reshape(3, 1)
            + np.cross(w_i[:, i], np.cross(w_i[:, i], r_i_i_cmi[:, i])).reshape(3, 1)
        )

        # Update values
        w_i0 = w_i[:, i : i + 1]
        wp_i0 = wp_i[:, i : i + 1]
        a_i0 = a_i[:, i : i + 1]

    # Backward recursion
    f_i = np.zeros((3, n + 1))
    tau_i = np.zeros((3, n + 1))
    _u = np.zeros(n)

    for i in reversed(range(n)):
        gi = T[i][:3, :3].T @ g0.reshape(3, 1)
        aci_i = a_ci[:, i].reshape(3, 1)

        if i == n - 1:
            f_i[:, i : i + 1] = m[i] * (aci_i - gi)
            tau_i[:, i : i + 1] = (
                -np.cross(f_i[:, i], r_i_im1_i[:, i] + r_i_i_cmi[:, i]).reshape(3, 1)
                + I[i] @ wp_i[:, i : i + 1]
                + np.cross(w_i[:, i], I[i] @ w_i[:, i]).reshape(3, 1)
            )
        else:
            R_i_ip1 = T[i][:3, :3].T @ T[i + 1][:3, :3]
            f_i[:, i : i + 1] = R_i_ip1 @ f_i[:, i + 1 : i + 2] + m[i] * (aci_i - gi)

            tau_init = (
                -np.cross(f_i[:, i], r_i_im1_i[:, i] + r_i_i_cmi[:, i]).reshape(3, 1)
                + I[i] @ wp_i[:, i : i + 1]
                + np.cross(w_i[:, i], I[i] @ w_i[:, i]).reshape(3, 1)
            )

            tau_i[:, i : i + 1] = (
                R_i_ip1 @ tau_i[:, i + 1 : i + 2]
                + np.cross(R_i_ip1 @ f_i[:, i + 1], r_i_i_cmi[:, i]).reshape(3, 1)
                + tau_init
            )

        if i == 0:
            z_im1_im1 = np.array([[0], [0], [1]])
            _u[i] = (tau_i[:, i].T @ T[i][:3, :3].T @ z_im1_im1).item()
        else:
            R_im1_0 = T[i - 1][:3, :3]
            z_im1_im1 = T[i - 1][:3, :3].T @ T[i - 1][:3, 2].reshape(3, 1)
            _u[i] = (tau_i[:, i].T @ T[i][:3, :3].T @ R_im1_0 @ z_im1_im1).item()

    return _u


def GetEulerLagrangeMatrices(_q, _qp, _g0, robot):
    n = len(_q)
    # Gravity Vector
    _G = Newton_Euler(_q, np.zeros(n), np.zeros(n), _g0, robot)
    # Coriolis vector
    _Cqp = Newton_Euler(_q, _qp, np.zeros(n), np.array([0, 0, 0]), robot)
    # Inertia Matrix
    _M = np.zeros([n, n])
    I = np.eye(n)
    for i in range(n):
        _M[:, i] = Newton_Euler(_q, np.zeros(n), I[:, i], np.array([0, 0, 0]), robot)

    return _M, _Cqp, _G


# %%
def _dyn_model(robot_, q, qdot):
    n = len(robot_.links)
    q = np.matrix(q).reshape((n, 1))
    qdot = np.matrix(qdot).reshape((n, 1))
    # Error handling
    if not Utils.is_a_vector(qdot, n):
        raise Exception(
            "The parameter 'qdot' should be a " + str(n) + " dimensional vector."
        )

    if not Utils.is_a_vector(q, n):
        raise Exception(
            "The parameter 'q' should be a " + str(n) + " dimensional vector."
        )
    # end error handling

    jj_geo, jac_geo, htm_com = robot_.jac_jac_geo(q=q, axis="com")

    i_mat_rot = []

    for i in range(n):
        i_mat = robot_.links[i].inertia_matrix + robot_.links[i].mass * Utils.S(
            robot_.links[i].center_of_mass
        ) * Utils.S(robot_.links[i].center_of_mass)
        i_mat_rot.append(htm_com[i][0:3, 0:3] * i_mat * htm_com[i][0:3, 0:3].T)

    list_pjac_i_pk = []

    for i in range(n):
        list = []
        for k in range(n):
            pjac_i_pk = np.matrix(np.zeros((6, n)))
            for j in range(i + 1):
                pjac_i_pk[:, j] = jj_geo[i][j][:, k]

            list.append(pjac_i_pk)

        list_pjac_i_pk.append(list)

    list_pm_pk = []

    for k in range(n):
        pm_pk = np.zeros((n, n))
        for i in range(k, n):
            pm_pk += (
                robot_.links[i].mass
                * list_pjac_i_pk[i][k][0:3, :].T
                * jac_geo[i][0:3, :]
            )
            pm_pk += (
                (
                    list_pjac_i_pk[i][k][3:6, :].T
                    + jac_geo[i][3:6, :].T * Utils.S(jac_geo[i][3:6, k])
                )
                * i_mat_rot[i]
                * jac_geo[i][3:6, :]
            )

        list_pm_pk.append(pm_pk + (pm_pk).T)

    # Compute the M matrix

    dyn_m = np.zeros((n, n))

    for i in range(n):
        dyn_m += robot_.links[i].mass * jac_geo[i][0:3, :].T * jac_geo[i][0:3, :]
        dyn_m += jac_geo[i][3:6, :].T * i_mat_rot[i] * jac_geo[i][3:6, :]

    # Compute the C matrix

    dyn_c1 = np.matrix(np.zeros((n, 1)))
    dyn_c2 = np.matrix(np.zeros((n, 1)))
    qdot_v = np.matrix(qdot).reshape((n, 1))

    for j in range(n):
        dyn_c1 = dyn_c1 + (qdot_v[j, 0] * list_pm_pk[j]) * qdot_v

    for i in range(n):
        dyn_c2[i] = 0.5 * qdot_v.T * list_pm_pk[i] * qdot_v

    dyn_c = (dyn_c1 - dyn_c2).reshape((n, 1))

    # Compute the G matrix

    dyn_g = np.zeros((n, 1))
    GRAVITY_ACC = 9.81
    for i in range(n):
        dyn_g += GRAVITY_ACC * robot_.links[i].mass * jac_geo[i][2, :].reshape((n, 1))

    return dyn_m, dyn_c, dyn_g


# %%
robot = ub.Robot.create_kinova_gen3()
# robot = ub.Robot.create_kuka_kr5()
print("a")
print(robot)
# robot.gravity_vector = np.zeros((3,))
robot.gravity_vector = np.array([0, 0, 9.81])
print(robot.gravity_vector)
for link in robot.links:
    print(link.mass, link.inertia_tensor, link.center_of_mass)

M, C_, g = robot.dynamic_model(np.zeros(len(robot.links)), np.zeros(len(robot.links)))
print(f"M: {M}\nC_: {C_.ravel()}\ng: {g.ravel()}")
M2, C2, g2 = _dyn_model(robot, np.zeros(len(robot.links)), np.zeros(len(robot.links)))
print(f"M2: {M2}\nC2: {C2.ravel()}\ng2: {g2.ravel()}")
M3, C3, g3 = GetEulerLagrangeMatrices(
    np.zeros(len(robot.links)), np.zeros(len(robot.links)), robot.gravity_vector, robot
)
print(f"M3: {M3}\nC3: {C3.ravel()}\ng3: {g3.ravel()}")

M, C_, g = robot.dynamic_model(
    np.array([0.1] * len(robot.links)), np.array([0.1] * len(robot.links))
)
print(f"M: {M}\nC_: {C_.ravel()}\ng: {g.ravel()}")
M2, C2, g2 = _dyn_model(
    robot, np.array([0.1] * len(robot.links)), np.array([0.1] * len(robot.links))
)
print(f"M2: {M2}\nC2: {C2.ravel()}\ng2: {g2.ravel()}")

print(np.linalg.pinv(M) @ (g - C_))
print(np.linalg.pinv(M2) @ (g2 - C2))

# %%
# gravity comparison


def check_gravity_consistency(robot, q, g0=np.array([0, 0, -9.81]), tol=1e-5):
    """
    Compare the analytical gravity torque vector from the robot model
    with the numerical gradient of the potential energy.

    Parameters
    ----------
    robot : Robot
        Instance providing:
            - robot.gravity(q) or robot.rne(q, 0*q, 0*q, g0) → τ_g
            - robot.masses, robot.com_positions
            - robot.fkm(axis='dh') → list of HTMs (base→i)
    q : array_like, shape (n,)
        Joint configuration.
    g0 : array_like, shape (3,), optional
        Gravity vector expressed in base frame.
    tol : float
        Tolerance for comparison (L2 norm threshold).

    """
    q = np.asarray(q).flatten()
    n = len(q)

    # Compute potential energy U(q)
    def potential_energy(q_):
        T_list = robot.fkm(q_, axis="dh")
        U = 0.0
        for i, T in enumerate(T_list):
            m_i = robot.links[i].mass
            r_cmi = robot.links[i].center_of_mass.copy().reshape(-1, 1)  # expressed in DH frame
            # position of COM in base frame
            p_cmi_base = T[:3, :3] @ r_cmi + T[:3, 3]
            U += -m_i * g0.dot(p_cmi_base)  # minus because gravity potential is -m g·r
        return U

    # Finite difference approximation of ∂U/∂q
    eps = 1e-5
    tau_g_fd = np.zeros(n)
    for i in range(n):
        dq = np.zeros(n)
        dq[i] = eps
        tau_g_fd[i] = (potential_energy(q + dq) - potential_energy(q - dq)) / (2 * eps)

    # Analytical gravity torque vector
    _, _, g1 = robot.dynamic_model(q, np.zeros(n))
    _, _, g2 = _dyn_model(robot, q, np.zeros(n))
    _, _, g3 = GetEulerLagrangeMatrices(q, np.zeros(n), g0, robot)
     
    # Compare
    err1 = np.linalg.norm(g1.flatten() - tau_g_fd.flatten())
    err2 = np.linalg.norm(g2.flatten() - tau_g_fd.flatten())
    err3 = np.linalg.norm(g3.flatten() - tau_g_fd.flatten())
    print(f"Cpp NE error: {err1}, Python NE error: {err3}, Python EL error: {err2}")
    print(f" Numm g: {tau_g_fd.ravel()}\n Analytical g1: {g1.ravel()}\n Analytical g2: {g2.ravel()}\n Analytical g3: {g3.ravel()}")


check_gravity_consistency(robot, np.ones(len(robot.links)))
check_gravity_consistency(robot, np.random.rand(len(robot.links)) * 2 - 1)

#%%

