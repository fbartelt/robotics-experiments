# %%
import numpy as np
from uaibot import Utils
from uaibot.gpu.distance import holder_distance_objects_with_grad
from uaibot.gpu.lie import se3_exp
from uaibot.gpu.functional import holder_min

Z_DOWN = np.array([0.0, 0.0, 1.0]).reshape(-1, 1)

Z_DOWN.ndim


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
    coad_vec = -np.asarray(w).reshape(-1, 1) @ ad_mat
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
        gravity_dir=Z_DOWN,
        gravity_acceleration=9.81,
    ):
        self.inertia_tensor = self.transform_inertia(inertia)
        self.mass = mass
        self.gravity_dir = gravity_dir
        self.gravity_acceleration = gravity_acceleration

    def get_mass_matrix(self, H):
        R = np.asarray(H)[:3, :3]
        linear_part = np.eye(3) * self.mass
        angular_part = np.asarray(R) @ self.inertia_tensor @ np.asarray(R).T
        mass_matrix = np.eye(6)
        mass_matrix[:3, :3] = linear_part
        mass_matrix[3:, 3:] = angular_part

        return mass_matrix

    def get_mass_matrix_derivative(self, H, xi):
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

    def get_coadjoint_term(self, H, xi):
        R = np.asarray(H)[:3, :3]
        coad = coadjoint_SE3(xi, xi)
        M = self.get_mass_matrix(R)
        return M @ coad

    def get_gravity_vector(self, H):
        gravity = H @ to_homogeneous(
            self.mass * self.gravity_dir * self.gravity_acceleration
        ).reshape(-1, 1)
        return gravity


def get_aclecian(uav_model, obstacles, delta=1e-4, gamma=2.0, eps=1e-3, device=None):
    H0 = np.asarray(uav_model.htm, dtype=float)

    def grad_at(H):
        _, g = holder_distance_objects_with_grad(
            [uav_model],
            obstacles,
            htms_a=[H],
            gamma=gamma,
            eps=eps,
            device=device,
        )
        return g["grad_HA"][0].detach().cpu().numpy().astype(np.float64)

    # Base call: need both the distances and the base gradient.
    dists, grads = holder_distance_objects_with_grad(
        [uav_model],
        obstacles,
        htms_a=[H0],
        gamma=gamma,
        eps=eps,
        device=device,
    )
    dists = dists.detach().cpu().numpy()  # (1, N_B)
    grad0 = grads["grad_HA"][0].detach().cpu().numpy().astype(np.float64)
    A = np.zeros((6, 6))
    for i in range(6):
        ei = np.zeros(6)
        ei[i] = 1.0
        Hp = se3_exp(ei * delta) @ H0
        Hm = se3_exp(-ei * delta) @ H0
        # A[i, j] = d/dvarpi |_0 [L[f]]_j( exp(varpi S(e_i)) X )
        A[i, :] = (grad_at(Hp) - grad_at(Hm)) / (2.0 * delta)

    # NOTE: no 0.5*(A+A.T) — the aclecian is not symmetric.

    # Optional sanity check: skew part vs -L[f]([lambda,omega])
    # ...

    return A, grad0, dists.ravel()


def control(
    uav_model,
    obstacles,
    k1=2.0,
    k2=0.5,
    delta_acl=1e-3,
):
    # alpha_1(x) = k1 * x
    # alpha_2(x) = k2 * x
    aclecian, L_beta = get_aclecian(uav_model, obstacles, delta=delta_acl)
    constraint = (
        xi.T @ aclecian @ xi + L_beta * (k1 * xi + u) + k2 * (L_beta @ xi + k1 * beta)
    )
