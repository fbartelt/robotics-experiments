# %%
import numpy as np
from uaibot import Utils, Simulation
from uaibot.simobjects import Box
from uaibot.gpu.distance import holder_distance_objects_with_grad
from uaibot.gpu.lie import se3_exp
from uaibot.gpu.functional import holder_min, shaping_function
from uaibot_cpp_bind import expSE3, SmapSE3
from tqdm import tqdm

Z_DOWN = np.array([0.0, 0.0, 1.0]).reshape(-1, 1)


def to_homogeneous(a):
    a_hom = np.ones((4,))
    a_hom[:3] = np.asarray(a).ravel()
    return a_hom


def transform_inertia(inertia):
    """Transforms the input into a 3x3 inertia matrix. If inertia is a 3
    element list, it is assumed the elements are Ixx, Iyy, Izz and a
    diagonal matrix is returned. Else, a 9 element list is expected.

    Parameters
    ----------
    inertia: np.array
        The elements of the inertia matrix in order xx, xy, xz, yx, yy,
        yz, zx, zy, zz.

    Returns
    -------
    inertia_tensor: np.array
        The 3x3 inertia tensor
    """
    if len(inertia) == 9:
        return np.array(inertia).reshape(3, 3)
    elif len(inertia) == 3:
        return np.diag(np.asarray(inertia).ravel())
    else:
        raise ValueError(
            f"Inertia must be either of lenght 3 or 9, received {len(inertia)}"
        )


def adjoint_matrix_se3(xi):
    xi = np.asarray(xi, dtype=float).reshape(6)
    v = xi[:3]
    omega = xi[3:]

    Sv = np.asarray(Utils.S(v))
    Somega = np.asarray(Utils.S(omega))
    A = np.zeros((6, 6))
    A[:3, :3] = Somega
    A[:3, 3:] = Sv
    A[3:, 3:] = Somega
    return A


def coadjoint_SE3(v, w):
    """Computes the coadjoint ad*_v(w) for the SE(3) canonical basis

    Parameters
    ----------
    v: np.array
        Element of the tangent space (lie algebra).
    w: np.array
        Element of the cotangent space.

    Returns
    -------
    x: np.array
        Element of the cotangent space.
    """

    ad_mat = adjoint_matrix_se3(v)
    coad_vec = -np.asarray(w).reshape(1, -1) @ ad_mat
    return coad_vec


# Precompute once at module load — these are constant
_E_ADJ = [adjoint_matrix_se3(np.eye(6)[j]) for j in range(6)]


def aclecian_skew_part(Lf):
    """Skew-symmetric part of the aclecian from the first-order Lie derivative.

    Parameters
    ----------
    Lf : array-like, shape (6,)
        L[f] in the convention [v_x, v_y, v_z, w_x, w_y, w_z].

    Returns
    -------
    A_skew : np.ndarray, shape (6, 6)
    """
    Lf = np.asarray(Lf, dtype=float).reshape(6)
    A = np.zeros((6, 6))
    for j in range(6):
        A[:, j] = _E_ADJ[j].T @ Lf
    return A


class UAV:
    def __init__(
        self,
        inertia,
        mass,
        htm=np.eye(4),
        gravity_dir=Z_DOWN,
        gravity_acceleration=9.81,
        height=0.2,
        width=0.4,
        depth=0.4,
    ):
        self.inertia_tensor = transform_inertia(inertia)
        self.mass = mass
        self.gravity_dir = gravity_dir
        self.gravity_acceleration = gravity_acceleration
        self.collision_model = Box(width=width, height=height, depth=depth)
        self.htm = htm
        self.xi = np.zeros((6, 1))
        self.xi_dot = np.zeros((6, 1))

    def get_mass_matrix(self, H=None):
        if H is None:
            H = self.htm
        R = np.asarray(H)[:3, :3]
        linear_part = np.eye(3) * self.mass
        angular_part = np.asarray(R) @ self.inertia_tensor @ np.asarray(R).T
        mass_matrix = np.eye(6)
        mass_matrix[:3, :3] = linear_part
        mass_matrix[3:, 3:] = angular_part

        return mass_matrix

    def get_mass_matrix_derivative(self, H=None, xi=None):
        if H is None:
            H = self.htm
        if xi is None:
            xi = self.xi
        R = np.asarray(H)[:3, :3]
        omega = np.asarray(xi).ravel()[3:]
        linear_part = np.zeros((3, 3))
        S_omega = np.asarray(Utils.S(omega))
        R_ = np.asarray(R)
        R_T = R_.T
        angular_part = (
            S_omega @ R_ @ self.inertia_tensor @ R_T
            - R_ @ self.inertia_tensor @ R_T @ S_omega
        )
        mass_matrix_derivative = np.eye(6)
        mass_matrix_derivative[:3, :3] = linear_part
        mass_matrix_derivative[3:, 3:] = angular_part

        return mass_matrix_derivative

    def get_coadjoint_term(self, H=None, xi=None):
        if H is None:
            H = self.htm
        if xi is None:
            xi = self.xi
        R = np.asarray(H)[:3, :3]
        coad = coadjoint_SE3(xi, xi)
        M = self.get_mass_matrix(R)
        return M @ coad.T

    def get_gravity_vector(self, H=None):
        if H is None:
            H = self.htm
        R = np.asarray(H)[:3, :3]
        # world-frame gravity force
        f_world = (self.mass * self.gravity_dir * self.gravity_acceleration).reshape(
            3, 1
        )
        # rotate to body frame
        f_body = R.T @ f_world
        # embed in se(3): linear part = force, angular part = 0 (gravity acts at CoM)
        g = np.zeros((6, 1))
        g[:3, :] = f_body
        return g

    # def get_gravity_vector(self, H=None):
    #     if H is None:
    #         H = self.htm
    #     gravity = H @ to_homogeneous(
    #         self.mass * self.gravity_dir * self.gravity_acceleration
    #     ).reshape(-1, 1)
    #     return gravity

    def dynamics(self, u, H=None, xi=None):
        if H is None:
            H = self.htm
        if xi is None:
            xi = self.xi
        u = np.asarray(u).reshape(6, 1)
        M = self.get_mass_matrix(H)
        M_dot = self.get_mass_matrix_derivative(H, xi)
        coad = self.get_coadjoint_term(H, xi)
        g = self.get_gravity_vector(H)
        # u = M @ xi_dot + M_dot @ xi - coad + g
        xi_dot = np.linalg.pinv(M) @ (u - M_dot @ xi + coad - g)
        return xi_dot

    def inv_dynamics(self, H=None, xi=None, xi_dot=None):
        if H is None:
            H = self.htm
        if xi is None:
            xi = self.xi
        if xi_dot is None:
            xi_dot = self.xi_dot
        M = self.get_mass_matrix(H)
        M_dot = self.get_mass_matrix_derivative(H, xi)
        coad = self.get_coadjoint_term(H, xi)
        g = self.get_gravity_vector(H)
        u = M @ xi_dot + M_dot @ xi - coad + g
        return u

    def step(self, t, dt, u):
        Omega = SmapSE3(self.xi)
        self.htm = expSE3(Omega * dt) @ self.htm
        self.xi += self.xi_dot * dt
        self.xi_dot = self.dynamics(u)
        self.collision_model.add_ani_frame(time=float(t), htm=self.htm)


def get_aclecian(
    uav_model, obstacles, delta=1e-4, gamma=2.0, eps=1e-3, device=None, k=2
):
    H0 = np.asarray(uav_model.htm, dtype=float)

    reduce_fn = lambda d: shaping_function(holder_min(d, gamma, dim=1), k).sum()

    def h_grad(H):
        dists, grads = holder_distance_objects_with_grad(
            [uav_model.collision_model],
            obstacles,
            htms_a=[H],
            gamma=gamma,
            eps=eps,
            device=device,
            reduce_fn=reduce_fn,
        )
        return (
            dists.detach().cpu(),  # tensor (1, N_B)
            grads["grad_HA"][0].detach().cpu().numpy().astype(np.float64),
        )

    d0, Lf = h_grad(H0)

    A_fd = np.zeros((6, 6))
    for i in range(6):
        ei = np.zeros(6)
        ei[i] = 1.0
        Ei = SmapSE3(ei)
        _, gp = h_grad(expSE3(Ei * delta) @ H0)
        _, gm = h_grad(expSE3(-Ei * delta) @ H0)
        A_fd[i, :] = (gp - gm) / (2.0 * delta)

    Sym_fd = 0.5 * (A_fd + A_fd.T)
    Skew_exact = aclecian_skew_part(Lf)
    A = Sym_fd + Skew_exact
    # NO final symmetrization — the aclecian is not symmetric.

    # Consistency check (debug)
    skew_fd = 0.5 * (A_fd - A_fd.T)
    err = np.linalg.norm(skew_fd - Skew_exact)
    # For central differences you should see err ~ O(delta^2).
    # If err ~ O(delta), something is wrong (transpose, twist order, sign).
    # beta = the SAME scalar the reduce_fn minimized
    beta = shaping_function(holder_min(d0, gamma=gamma, dim=1), k).item()

    return A, Lf, beta


def control(
    uav_model,
    obstacles,
    k1=2.0,
    k2=0.5,
    delta_acl=1e-3,
    eta=0.1,
    gamma=2.0,
    eps=1e-3,
    k_shape=2,
):
    H = uav_model.htm
    xi = uav_model.xi  # (6, 1)
    xid = uav_model.xi_dot  # (6, 1)

    aclecian, L_beta, beta = get_aclecian(
        uav_model,
        obstacles,
        delta=delta_acl,
        gamma=gamma,
        eps=eps,
        k=k_shape,
    )

    L_beta = np.asarray(L_beta, dtype=float).reshape(1, 6)  # row vector
    beta = float(beta)  # scalar
    print(beta, L_beta.ravel())

    quad = (xi.T @ aclecian @ xi).item()  # xi^T A xi
    lin = (L_beta @ xi).item()  # L_beta @ xi

    # Psi_2 constraint
    # xi.T @ aclecian @ xi + L_beta * (k1 * xi + u) + k2 * (L_beta @ xi + k1 * beta)

    # CBF drift d = xi^T A xi + (k1+k2) L_beta xi + k1 k2 beta
    d_cbf = quad + (k1 + k2) * lin + k1 * k2 * beta

    A_qp = L_beta
    b_qp = -eta * np.array([d_cbf])

    # Nominal: input realizing the desired xi_dot (hover if xid = 0)
    u_d = uav_model.inv_dynamics(H, xi, xid).reshape(
        6,
    )

    H_qp = np.eye(6)
    f_qp = -u_d
    # QP: min 0.5 ||u - u_d||^2 s.t. A_qp u >= b_qp
    u = np.asarray(Utils.solve_qp(H_qp, f_qp, A_qp, b_qp)).reshape(6, 1)

    # Sanity: verify the CBF constraint is honoured
    psi2 = (
        xi.T @ aclecian @ xi + L_beta @ (k1 * xi + u) + k2 * (L_beta @ xi + k1 * beta)
    ).item()
    assert psi2 >= -1e-6, f"CBF violated: psi2={psi2:.3e}, beta={beta:.3e}"

    return u


# def control(
#     uav_model,
#     obstacles,
#     k1=2.0,
#     k2=0.5,
#     delta_acl=1e-3,
#     gamma=2.0,
#     eps=1e-3,
#     k_shape=2,
# ):
#     H = uav_model.htm
#     xi = uav_model.xi
#     xi_dot = uav_model.xi_dot
#     # Psi_2 constraint
#     # xi.T @ aclecian @ xi + L_beta * (k1 * xi + u) + k2 * (L_beta @ xi + k1 * beta)
#
#     aclecian, L_beta, beta = get_aclecian(
#         uav_model, obstacles, delta=delta_acl, gamma=gamma, eps=eps, k=k_shape
#     )
#     L_beta = np.asarray(L_beta, dtype=float).reshape(1, 6)
#     beta = float(beta)  # scalar
#
#     # CBF constraint: L_beta @ u >= -(xi^T A xi + (k1+k2) L_beta @ xi + k1 k2 beta)
#     b_qp = float(xi @ aclecian @ xi + (k1 + k2) * (L_beta @ xi) + k1 * k2 * beta)
#     A_qp = L_beta.reshape(1, 6)
#     b_qp = -np.array([b_qp])
#
#     # Nominal control and QP-based safety filter
#     u_d = uav_model.dynamics(H, xi, xi_dot)  # your nominal, shape (6,)
#     # QP: min 0.5 ||u - u_d||^2 s.t. A_qp u >= b_qp
#     H_qp = np.eye(6)
#     f_qp = -u_d
#     u = np.asarray(Utils.solve_qp(H_qp, f_qp, A_qp, b_qp)).reshape(6, 1)
#
#     # Optional debug: verify psi_2
#     psi2 = float(
#         xi @ aclecian @ xi + L_beta @ (k1 * xi + u) + k2 * (L_beta @ xi + k1 * beta)
#     )
#     assert psi2 >= -1e-6, f"CBF violated: psi2={psi2:.3e}"
#
#     return u


# %%
drone = UAV(inertia=np.ones(3), mass=20)
obstacle = Box(htm=Utils.trn([1, 0.1, 0.5]), width=1, height=1, depth=1)

sim = Simulation.create_sim_grid([drone.collision_model, obstacle])

T, dt = 5.0, 1e-2
for t in tqdm(np.arange(0, T, dt)):
    u = control(drone, [obstacle])
    drone.step(t, dt, u)

sim.run_in_browser()
print(drone.get_gravity_vector().ravel())
