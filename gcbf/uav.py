# %%
import torch
import numpy as np
from uaibot import Utils, Simulation, Robot
from uaibot.simobjects import Box
from uaibot.simobjects.curve import CurveSE3
from uaibot.gpu.distance import (
    holder_distance_objects_with_grad,
    holder_distance_objects,
)
from uaibot.gpu.lie import se3_exp
from uaibot.gpu.functional import holder_min, shaping_function
from uaibot_cpp_bind import expSE3, SmapSE3
from tqdm import tqdm
from gcbf.curves import make_circle
from uaibot.gpu.vectorfield import VectorFieldSE3
from uaibot.gpu.lie import exp_se3, smap_se3, inv_smap_se3

Z_DOWN = np.array([0.0, 0.0, 1.0]).reshape(-1, 1)


def to_homogeneous(a):
    a_hom = np.ones((4,))
    a_hom[:3] = np.asarray(a).ravel()
    return a_hom


def _to_torch(x, device, dtype=torch.float64):
    if isinstance(x, torch.Tensor):
        return x.to(device=device, dtype=dtype)
    return torch.as_tensor(np.asarray(x, dtype=float), dtype=dtype, device=device)


class TorchVFContext:
    """Cache curve / derivative / VectorFieldSE3 across the sim loop."""

    def __init__(
        self,
        curve,
        curve_derivative,
        device=None,
        dtype=torch.float64,
        scale_pos=1.0,
        vf_delta=1e-3,
        vf_ds=1e-3,
    ):
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.dtype = dtype
        self.curve = _to_torch(curve, self.device, dtype)
        if curve_derivative is None or (
            isinstance(curve_derivative, (list, tuple)) and len(curve_derivative) == 0
        ):
            self.curve_derivative = None
        else:
            self.curve_derivative = _to_torch(curve_derivative, self.device, dtype)
        self.vf = VectorFieldSE3(delta=vf_delta, ds=vf_ds, scale_pos=scale_pos)

    def eval_vf(self, H_batch, kt1, kt2, kt3, kn1, kn2):
        return self.vf.vectorfield_se3(
            H_batch,
            self.curve,
            kt1=kt1,
            kt2=kt2,
            kt3=kt3,
            kn1=kn1,
            kn2=kn2,
            curve_derivative=self.curve_derivative,
        )


class UAV:
    def __init__(
        self,
        inertia,
        mass,
        htm=None,
        gravity_dir=Z_DOWN,
        gravity_acceleration=9.81,
        height=0.2,
        width=0.4,
        depth=0.4,
        model_color="red",
        model_opacity=0.5,
        device="cpu",
        dtype=torch.float64,
    ):
        self.device = torch.device(device)
        self.dtype = dtype

        # Physical parameters
        self.mass = float(mass)
        self.gravity_acceleration = float(gravity_acceleration)

        # Convert inertia to numpy before calling transform_inertia,
        # since that function may still be numpy-based.
        # if torch.is_tensor(inertia):
        #     inertia_np = inertia.detach().cpu().numpy()
        # else:
        #     inertia_np = np.asarray(inertia)

        # self.inertia_tensor = self._to_tensor(transform_inertia(inertia_np))
        self.inertia_tensor = transform_inertia(
            inertia,
            device=self.device,
            dtype=self.dtype,
        )

        self.gravity_dir = self._to_tensor(gravity_dir).reshape(3)

        # State
        if htm is None:
            htm = torch.eye(4, dtype=dtype, device=self.device)

        self.htm = self._to_tensor(htm).reshape(4, 4)
        self.xi = torch.zeros((6, 1), dtype=dtype, device=self.device)
        self.xi_dot = torch.zeros((6, 1), dtype=dtype, device=self.device)

        # Visualization remains unchanged
        self.collision_model = Box(
            width=width,
            height=height,
            depth=depth,
            color=model_color,
            opacity=model_opacity,
        )

    # ----------------------------------------------------------
    # Tensor utilities
    # ----------------------------------------------------------

    def _to_tensor(self, x):
        if torch.is_tensor(x):
            return x.to(device=self.device, dtype=self.dtype)

        return torch.as_tensor(
            x,
            device=self.device,
            dtype=self.dtype,
        )

    @staticmethod
    def _skew(v):
        """3x3 skew-symmetric matrix."""
        x, y, z = v.reshape(3)

        zero = torch.zeros_like(x)

        return torch.stack(
            [
                torch.stack([zero, -z, y]),
                torch.stack([z, zero, -x]),
                torch.stack([-y, x, zero]),
            ]
        )

    @staticmethod
    def _smap_se3(xi):
        """
        se(3) matrix representation.

        Twist convention:
            xi = [v, omega]
        """
        xi = xi.reshape(6)
        v = xi[:3]
        omega = xi[3:]

        Omega = torch.zeros(
            (4, 4),
            dtype=xi.dtype,
            device=xi.device,
        )

        Omega[:3, :3] = UAV._skew(omega)
        Omega[:3, 3] = v

        return Omega

    @staticmethod
    def _ad_se3(xi):
        """
        Adjoint representation of the Lie algebra se(3).

        Twist convention:
            xi = [v, omega]

        ad_xi = [[skew(omega), skew(v)],
                 [0,           skew(omega)]]
        """
        xi = xi.reshape(6)
        v = xi[:3]
        omega = xi[3:]

        zero = torch.zeros(
            (3, 3),
            dtype=xi.dtype,
            device=xi.device,
        )

        S_v = UAV._skew(v)
        S_omega = UAV._skew(omega)

        return torch.cat(
            [
                torch.cat([S_omega, S_v], dim=1),
                torch.cat([zero, S_omega], dim=1),
            ],
            dim=0,
        )

    @staticmethod
    def _rotation(H):
        """Extract the rotation matrix from H or accept a 3x3 R."""
        if H.shape == (3, 3):
            return H
        return H[:3, :3]

    # ----------------------------------------------------------
    # Mass matrix
    # ----------------------------------------------------------

    def get_mass_matrix(self, H=None):
        if H is None:
            H = self.htm
        else:
            H = self._to_tensor(H)

        R = self._rotation(H)

        linear_part = self.mass * torch.eye(
            3,
            dtype=self.dtype,
            device=self.device,
        )

        angular_part = R @ self.inertia_tensor @ R.T

        return torch.block_diag(linear_part, angular_part)

    # ----------------------------------------------------------
    # Mass matrix derivative
    # ----------------------------------------------------------

    def get_mass_matrix_derivative(self, H=None, xi=None):
        if H is None:
            H = self.htm
        else:
            H = self._to_tensor(H)

        if xi is None:
            xi = self.xi
        else:
            xi = self._to_tensor(xi)

        R = self._rotation(H)
        omega = xi.reshape(6)[3:]

        S_omega = self._skew(omega)

        linear_part = torch.zeros(
            (3, 3),
            dtype=self.dtype,
            device=self.device,
        )

        angular_inertia = R @ self.inertia_tensor @ R.T

        angular_part = S_omega @ angular_inertia - angular_inertia @ S_omega

        return torch.block_diag(linear_part, angular_part)

    # ----------------------------------------------------------
    # Coadjoint term
    # ----------------------------------------------------------

    def get_coadjoint_term(self, H=None, xi=None):
        if H is None:
            H = self.htm
        else:
            H = self._to_tensor(H)

        if xi is None:
            xi = self.xi
        else:
            xi = self._to_tensor(xi)

        xi = xi.reshape(6, 1)

        M = self.get_mass_matrix(H)
        ad_mat = self._ad_se3(xi)

        # Equivalent to:
        # coad = coadjoint_SE3(xi, xi)
        # return M @ coad.T

        coad = -(xi.T @ ad_mat).T  # (6,1)

        return M @ coad  # (6,1)

    # ----------------------------------------------------------
    # Gravity
    # ----------------------------------------------------------

    def get_gravity_vector(self, H=None):
        if H is None:
            H = self.htm
        else:
            H = self._to_tensor(H)

        R = self._rotation(H)

        # Gravity force in the world frame
        f_world = self.mass * self.gravity_dir.reshape(3, 1) * self.gravity_acceleration

        # Convert to the body frame
        f_body = R.T @ f_world

        g = torch.zeros(
            (6, 1),
            dtype=self.dtype,
            device=self.device,
        )

        g[:3, :] = f_body

        return g

    # ----------------------------------------------------------
    # Forward dynamics
    # ----------------------------------------------------------

    def dynamics(self, u, H=None, xi=None):
        if H is None:
            H = self.htm
        else:
            H = self._to_tensor(H)

        if xi is None:
            xi = self.xi
        else:
            xi = self._to_tensor(xi)

        u = self._to_tensor(u).reshape(6, 1)

        M = self.get_mass_matrix(H)
        M_dot = self.get_mass_matrix_derivative(H, xi)
        coad = self.get_coadjoint_term(H, xi)
        g = self.get_gravity_vector(H)

        rhs = u - M_dot @ xi + coad - g

        # xi_dot = pinv(M) @ rhs
        xi_dot = torch.linalg.pinv(M) @ rhs

        return xi_dot

    # ----------------------------------------------------------
    # Inverse dynamics
    # ----------------------------------------------------------

    def inv_dynamics(self, H=None, xi=None, xi_dot=None):
        if H is None:
            H = self.htm
        else:
            H = self._to_tensor(H)

        if xi is None:
            xi = self.xi
        else:
            xi = self._to_tensor(xi)

        if xi_dot is None:
            xi_dot = self.xi_dot
        else:
            xi_dot = self._to_tensor(xi_dot)

        H = H.reshape(4, 4)
        xi = xi.reshape(6, 1)
        xi_dot = xi_dot.reshape(6, 1)

        M = self.get_mass_matrix(H)
        M_dot = self.get_mass_matrix_derivative(H, xi)
        coad = self.get_coadjoint_term(H, xi)
        g = self.get_gravity_vector(H)

        u = M @ xi_dot + M_dot @ xi - coad + g

        return u

    # ----------------------------------------------------------
    # Integration step
    # ----------------------------------------------------------

    def step(self, t, dt, u, type="dynamic"):
        type_ = type.lower()

        u = self._to_tensor(u).reshape(6, 1)

        # Integrate the pose using the current body velocity.
        Omega = self._smap_se3(self.xi)

        self.htm = torch.matrix_exp(Omega * dt) @ self.htm

        if "kin" in type_:
            self.xi = u
            self.xi_dot = torch.zeros_like(self.xi)

        else:
            # Preserve the original integration order.
            self.xi = self.xi + self.xi_dot * dt

            self.xi_dot = self.dynamics(u)

            if not torch.isfinite(self.xi).all():
                print(f"[step t={t}] xi has NaN")

            if not torch.isfinite(self.xi_dot).all():
                print(f"[step t={t}] xi_dot has NaN")

            if self.xi_dot.abs().max().item() > 1e6:
                print(
                    f"[step t={t}] xi_dot HUGE "
                    f"(max={self.xi_dot.abs().max().item():.3e})"
                )

            if u.abs().max().item() > 1e6:
                print(f"[step t={t}] u is HUGE " f"(max={u.abs().max().item():.3e})")

        # The visualization library may still expect numpy arrays.
        self.collision_model.add_ani_frame(
            time=float(t),
            htm=self.htm.detach().cpu().numpy(),
        )



def transform_inertia(inertia, device=None, dtype=None):
    """Converts inertia into a 3x3 PyTorch inertia tensor.

    Accepted inputs:
        - 3 elements: Ixx, Iyy, Izz (diagonal inertia)
        - 9 elements: full inertia matrix
        - 3x3 tensor: full inertia matrix

    Device and dtype are preserved for tensor inputs unless
    explicitly overridden.
    """

    if torch.is_tensor(inertia):
        if device is None:
            device = inertia.device
        if dtype is None:
            dtype = inertia.dtype
    else:
        if device is None:
            device = torch.device("cpu")
        if dtype is None:
            dtype = torch.get_default_dtype()

    inertia_t = torch.as_tensor(
        inertia,
        device=device,
        dtype=dtype,
    )

    if inertia_t.shape == (3, 3):
        return inertia_t

    if inertia_t.numel() == 3:
        return torch.diag(inertia_t.reshape(3))

    if inertia_t.numel() == 9:
        return inertia_t.reshape(3, 3)

    raise ValueError(
        "Inertia must contain either 3 or 9 elements, "
        f"or have shape (3,3). Received shape {tuple(inertia_t.shape)}."
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


def _value_grad_hess(fn, x0):
    """Exact (value, grad, Hessian) of scalar fn at x0, pure autograd."""
    x = x0.detach().clone().requires_grad_(True)
    s = fn(x)
    (g,) = torch.autograd.grad(s, x, create_graph=True)
    n = x.numel()
    Hm = torch.empty(n, n, dtype=x.dtype, device=x.device)
    for i in range(n):
        (row,) = torch.autograd.grad(g[i], x, retain_graph=(i < n - 1))
        Hm[i] = row.detach()
    return s.detach(), g.detach(), Hm


def get_aclecian(
    uav_model,
    obstacles,
    delta=1e-4,
    gamma=2.0,
    eps=1e-3,
    device=None,
    k=2,
    dtype=torch.float64,
    debug=False,
):
    # ----------------------------------------------------------
    # 1. Device and dtype
    # ----------------------------------------------------------
    if device is None:
        if torch.is_tensor(uav_model.htm):
            device = uav_model.htm.device
        else:
            device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            )

    device = torch.device(device)

    # Preserve float64 for numerical differentiation by default.
    H0 = torch.as_tensor(
        uav_model.htm,
        device=device,
        dtype=dtype,
    ).reshape(4, 4)

    delta_t = torch.as_tensor(
        delta,
        device=device,
        dtype=dtype,
    )

    if delta <= 0:
        raise ValueError("delta must be positive.")

    # ----------------------------------------------------------
    # 2. Generate all 12 perturbations simultaneously
    # ----------------------------------------------------------

    I6 = torch.eye(
        6,
        device=device,
        dtype=dtype,
    )

    # Six canonical basis elements of se(3).
    E_hat = smap_se3(I6)  # (6, 4, 4)

    # Forward perturbations:
    # exp(delta * E_i) @ H0
    H_forward = (
        exp_se3(delta_t * E_hat)
        @ H0.unsqueeze(0)
    )  # (6, 4, 4)

    # Backward perturbations:
    # exp(-delta * E_i) @ H0
    H_backward = (
        exp_se3(-delta_t * E_hat)
        @ H0.unsqueeze(0)
    )  # (6, 4, 4)

    # Batch ordering:
    # 0       : nominal pose
    # 1:7     : forward perturbations
    # 7:13    : backward perturbations
    H_batch = torch.cat(
        [
            H0.unsqueeze(0),
            H_forward,
            H_backward,
        ],
        dim=0,
    )  # (13, 4, 4)

    # ----------------------------------------------------------
    # 3. Shaping function
    # ----------------------------------------------------------

    reduce_fn = lambda d: shaping_function(
        holder_min(d, gamma, dim=1),
        k,
        eps=eps,
    ).sum()

    # ----------------------------------------------------------
    # 4. Evaluate all poses in ONE geometry call
    # ----------------------------------------------------------

    # Each entry represents the same physical UAV at a different
    # perturbed pose.
    objects_a = [uav_model.collision_model] * 13

    d_all, grads = holder_distance_objects_with_grad(
        objects_a,
        obstacles,
        htms_a=H_batch,
        gamma=gamma,
        eps=eps,
        device=device,
        reduce_fn=reduce_fn,
        local_cache={},
    )

    grad_all = grads["grad_HA"]  # (13, 6)

    # ----------------------------------------------------------
    # 5. Nominal gradient L_beta
    # ----------------------------------------------------------

    Lf = grad_all[0]  # (6,)

    # ----------------------------------------------------------
    # 6. Batched central finite differences
    # ----------------------------------------------------------

    grad_forward = grad_all[1:7]   # (6, 6)
    grad_backward = grad_all[7:13] # (6, 6)

    # Row i corresponds to differentiation along basis E_i.
    A_fd = (
        grad_forward - grad_backward
    ) / (2.0 * delta_t)  # (6, 6)

    # Symmetric part.
    Sym_fd = 0.5 * (A_fd + A_fd.T)

    # Numerical skew part, only for diagnostics.
    Skew_fd = 0.5 * (A_fd - A_fd.T)

    # ----------------------------------------------------------
    # 7. Exact skew-symmetric component using Lie brackets
    # ----------------------------------------------------------

    # [E_i, E_j] = E_i E_j - E_j E_i
    #
    # E_hat has shape (6, 4, 4).
    # The result has shape (6, 6, 4, 4).

    commutators = (
        E_hat[:, None] @ E_hat[None, :]
        - E_hat[None, :] @ E_hat[:, None]
    )

    # Convert the Lie brackets back to se(3) coordinates.
    # Shape: (6, 6, 6)
    brackets = inv_smap_se3(commutators)

    # S_exact[i,j] = -0.5 * L_beta @ [E_i, E_j]
    Skew_exact = -0.5 * torch.einsum(
        "k,ijk->ij",
        Lf,
        brackets,
    )

    # ----------------------------------------------------------
    # 8. Original aclecian construction
    # ----------------------------------------------------------

    A = Sym_fd + 0.5 * Skew_exact

    # Do not symmetrize A.
    # It is intentionally nonsymmetric.

    # ----------------------------------------------------------
    # 9. beta
    # ----------------------------------------------------------

    beta = shaping_function(
        holder_min(
            d_all[0:1],
            gamma=gamma,
            dim=1,
        ),
        k,
        eps=eps,
    ).reshape(())

    # ----------------------------------------------------------
    # 10. Numerical validation
    # ----------------------------------------------------------

    if not torch.isfinite(A).all():
        raise FloatingPointError("NaN/Inf in aclecian.")

    if not torch.isfinite(Lf).all():
        raise FloatingPointError("NaN/Inf in L_beta.")

    if not torch.isfinite(beta):
        raise FloatingPointError("NaN/Inf in beta.")

    if not torch.isfinite(d_all).all():
        raise FloatingPointError("NaN/Inf in obstacle distances.")

    # ----------------------------------------------------------
    # 11. Debug
    # ----------------------------------------------------------

    if debug:
        skew_error = torch.linalg.norm(
            Skew_fd - Skew_exact
        )

        skew_relative_error = (
            skew_error
            / torch.linalg.norm(Skew_exact).clamp_min(1e-12)
        )

        print(f"[aclecian] delta: {delta:.2e}")
        print(f"[aclecian] beta: {beta.item():.8e}")
        print(
            f"[aclecian] ||A_fd||: "
            f"{torch.linalg.norm(A_fd).item():.6e}"
        )
        print(
            f"[aclecian] ||Sym_fd||: "
            f"{torch.linalg.norm(Sym_fd).item():.6e}"
        )
        print(
            f"[aclecian] ||Skew_fd - Skew_exact||: "
            f"{skew_error.item():.6e}"
        )
        print(
            f"[aclecian] relative skew error: "
            f"{skew_relative_error.item():.6e}"
        )
        print(
            f"[aclecian] ||A||: "
            f"{torch.linalg.norm(A).item():.6e}"
        )

    return A, Lf, beta

def get_beta_grad(uav_model, obstacles, gamma=2.0, eps=1e-3, k=2, device=None):
    H0 = np.asarray(uav_model.htm, dtype=float)

    reduce_fn = lambda d: shaping_function(
        holder_min(d, gamma, dim=1), k, eps=eps
    ).sum()

    dists, grads = holder_distance_objects_with_grad(
        [uav_model.collision_model],
        obstacles,
        htms_a=[H0],
        gamma=gamma,
        eps=eps,
        device=device,
        reduce_fn=reduce_fn,
    )
    d0_t = dists.detach().cpu()
    Lf = grads["grad_HA"][0].detach().cpu().numpy().astype(np.float64)
    beta = shaping_function(holder_min(d0_t, gamma=gamma, dim=1), k, eps=eps).item()
    return d0_t, Lf, beta


def get_beta_grad_torch(
    uav_model,
    obstacles,
    gamma=2.0,
    eps=1e-3,
    k=2,
    device=None,
    dtype=None,
):
    if device is None:
        device = uav_model.htm.device

    if dtype is None:
        dtype = uav_model.htm.dtype

    device = torch.device(device)

    H0 = _to_torch(
        uav_model.htm,
        device,
        dtype,
    ).reshape(4, 4)

    reduce_fn = lambda d: shaping_function(
        holder_min(d, gamma, dim=1),
        k,
        eps=eps,
    ).sum()

    # Obstacle distances and gradient with respect to UAV pose
    d0, grads = holder_distance_objects_with_grad(
        [uav_model.collision_model],
        obstacles,
        htms_a=[H0],
        gamma=gamma,
        eps=eps,
        device=device,
        reduce_fn=reduce_fn,
    )

    # Gradient of beta with respect to the UAV twist
    L_beta = grads["grad_HA"][0].reshape(1, 6)

    # Same shaping function as the original implementation
    beta_t = shaping_function(
        holder_min(d0, gamma=gamma, dim=1),
        k,
        eps=eps,
    )

    beta = float(beta_t.detach().item())

    if not torch.isfinite(L_beta).all():
        raise FloatingPointError(
            "NaN/Inf in L_beta"
        )

    if not np.isfinite(beta):
        raise FloatingPointError(
            "NaN/Inf in beta"
        )

    return d0, L_beta, beta


def tracking_xi_dot_torch(
    ctx,
    H,
    xi_actual,
    dt_fd=1e-4,
    kt1=1.0,
    kt2=1.0,
    kt3=1.0,
    kn1=1.0,
    kn2=1.0,
    Kp_xi=5.0,
):
    """
    Compute desired acceleration using the Torch vector field.

    1. Evaluate the vector field at the current pose.
    2. Perturb the pose forward and backward along the actual
       UAV velocity.
    3. Central-difference the vector field.
    4. Add proportional velocity feedback.

    All operations remain on the selected Torch device.

    Returns:
        xi_dot_des: (6,)
        xi_vf:      (6,)
        dist_vf:    scalar tensor
        idx_vf:     scalar tensor
    """

    device = ctx.device
    dtype = ctx.dtype

    # ----------------------------------------------------------
    # 1. Input conversion
    # ----------------------------------------------------------
    H = torch.as_tensor(
        H,
        device=device,
        dtype=dtype,
    ).reshape(4, 4)

    xi_actual = torch.as_tensor(
        xi_actual,
        device=device,
        dtype=dtype,
    ).reshape(6)

    # ----------------------------------------------------------
    # 2. Vector field evaluation helper
    # ----------------------------------------------------------
    def vf(H_batch):
        twist, dist, idx = ctx.eval_vf(
            H_batch,
            kt1,
            kt2,
            kt3,
            kn1,
            kn2,
        )

        return (
            twist.reshape(H_batch.shape[0], 6),
            dist.reshape(H_batch.shape[0]),
            idx.reshape(H_batch.shape[0]),
        )

    # ----------------------------------------------------------
    # 3. Nominal vector field
    # ----------------------------------------------------------
    xi_vf_batch, dist_batch, idx_batch = vf(
        H.unsqueeze(0)
    )

    xi_vf = xi_vf_batch[0]
    dist_vf = dist_batch[0]
    idx_vf = idx_batch[0]

    # ----------------------------------------------------------
    # 4. Forward and backward pose perturbations
    # ----------------------------------------------------------
    xi_pert = torch.stack(
        [
            xi_actual * dt_fd,
            -xi_actual * dt_fd,
        ],
        dim=0,
    )  # (2, 6)

    Omega_pert = smap_se3(xi_pert)  # (2, 4, 4)

    H_pert = (
        exp_se3(Omega_pert)
        @ H.unsqueeze(0)
    )  # (2, 4, 4)

    # ----------------------------------------------------------
    # 5. Batched vector field evaluation
    # ----------------------------------------------------------
    xi_pert_vf, _, _ = vf(H_pert)

    xi_fwd = xi_pert_vf[0]
    xi_bwd = xi_pert_vf[1]

    # ----------------------------------------------------------
    # 6. Feedforward acceleration
    # ----------------------------------------------------------
    xi_dot_ff = (
        xi_fwd - xi_bwd
    ) / (2.0 * dt_fd)

    # ----------------------------------------------------------
    # 7. Velocity feedback
    # ----------------------------------------------------------
    xi_dot_des = (
        xi_dot_ff
        + Kp_xi * (xi_vf - xi_actual)
    )

    # ----------------------------------------------------------
    # 8. Numerical validation
    # ----------------------------------------------------------
    for name, value in (
        ("xi_vf", xi_vf),
        ("xi_dot_ff", xi_dot_ff),
        ("xi_dot_des", xi_dot_des),
        ("dist_vf", dist_vf),
    ):
        if not torch.isfinite(value).all():
            raise FloatingPointError(
                f"NaN/Inf in tracking_xi_dot_torch: {name}"
            )

    return xi_dot_des, xi_vf, dist_vf, idx_vf


def control_1st(
    uav_model,
    obstacles,
    t,
    curve,
    curve_derivative=[],
    k1=2.0,
    k2=0.5,
    kt1=1.0,
    kt2=1.0,
    kt3=1.0,
    kn1=1.0,
    kn2=1.0,
    delta_acl=1e-4,
    gamma=2.0,
    eps=1e-3,
    k_shape=2,
    eta=10,
    dt_fd=1e-4,
    Kp_xi=5.0,
    beta_safe=0.1,
):
    H = np.asarray(uav_model.htm, dtype=float)
    xi = np.asarray(uav_model.xi).reshape(6, 1)

    # --- gradient of beta (no aclecian needed) ---
    d0_t, L_beta, beta = get_beta_grad(
        uav_model, obstacles, gamma=gamma, eps=eps, k=k_shape
    )
    L_beta = np.asarray(L_beta, dtype=float).reshape(1, 6)
    beta = float(beta)

    # --- nominal: vector field twist (with velocity feedback) ---
    xi_vf, dist, idx = Robot.vector_field_SE3(
        H,
        curve,
        kt1=kt1,
        kt2=kt2,
        kt3=kt3,
        kn1=kn1,
        kn2=kn2,
        curve_derivative=curve_derivative,
        delta=1e-3,
        ds=1e-3,
        mode="c++",
    )
    # first-order: nominal is xi_vf itself (already includes velocity P)
    u_d = np.asarray(xi_vf).reshape(6, 1)

    # --- first-order CBF QP ---
    # constraint:  L_beta @ u >= -eta * k1 * beta
    A_qp = L_beta
    b_qp = -eta * (beta - beta_safe) * np.ones(1)

    H_qp = np.eye(6) + 1e-4 * np.eye(6)
    f_qp = -u_d.ravel()
    u = np.asarray(Utils.solve_qp(H_qp, f_qp, A_qp, b_qp)).reshape(6, 1)

    # --- verification ---
    psi1 = (L_beta @ u + eta * (beta - beta_safe)).item()
    # assert psi1 >= -1e-6, f"CBF violated: psi1={psi1:.3e}, beta={beta:.3e}"
    logs = (dist, beta, np.asarray(xi_vf).reshape(-1, 1), L_beta, psi1)

    return u, logs



# def control_1st_torch(
#     uav_model,
#     obstacles,
#     t,
#     curve,
#     curve_derivative,
#     ctx,
#     k1=2.0,
#     k2=0.5,
#     kt1=1.0,
#     kt2=1.0,
#     kt3=1.0,
#     kn1=1.0,
#     kn2=1.0,
#     delta_acl=1e-4,
#     gamma=2.0,
#     eps=1e-3,
#     k_shape=2,
#     eta=10,
#     dt_fd=1e-4,
#     Kp_xi=5.0,
#     beta_safe=0.1,
# ):
#     device = ctx.device
#     dtype = ctx.dtype
#
#     # ----------------------------------------------------------
#     # 1. UAV state
#     # ----------------------------------------------------------
#     H_t = _to_torch(
#         uav_model.htm,
#         device,
#         dtype,
#     ).reshape(4, 4)
#
#     xi_t = _to_torch(
#         uav_model.xi,
#         device,
#         dtype,
#     ).reshape(6, 1)
#
#     # ----------------------------------------------------------
#     # 2. Obstacle barrier gradient
#     # ----------------------------------------------------------
#     d0_t, L_beta_t, beta = get_beta_grad_torch(
#         uav_model,
#         obstacles,
#         gamma=gamma,
#         eps=eps,
#         k=k_shape,
#         device=device,
#         dtype=dtype,
#     )
#
#     L_beta_t = L_beta_t.reshape(1, 6)
#
#     # ----------------------------------------------------------
#     # 3. Nominal vector field
#     # ----------------------------------------------------------
#     _, xi_vf, dist_vf, idx_vf = tracking_xi_dot_torch(
#         ctx,
#         H_t,
#         xi_t.squeeze(-1),
#         dt_fd=dt_fd,
#         kt1=kt1,
#         kt2=kt2,
#         kt3=kt3,
#         kn1=kn1,
#         kn2=kn2,
#         Kp_xi=Kp_xi,
#     )
#
#     # First-order nominal command is the vector-field twist.
#     u_d = xi_vf.reshape(6)
#
#     # ----------------------------------------------------------
#     # 4. First-order CBF constraint
#     # ----------------------------------------------------------
#     # L_beta @ u >= -eta * (beta - beta_safe)
#
#     b_cbf = -eta * (beta - beta_safe)
#
#     A_qp = L_beta_t
#
#     b_qp = torch.tensor(
#         [b_cbf],
#         dtype=dtype,
#         device=device,
#     )
#
#     # ----------------------------------------------------------
#     # 5. QP
#     # ----------------------------------------------------------
#     H_qp = (
#         1.0 + 1e-4
#     ) * torch.eye(
#         6,
#         dtype=dtype,
#         device=device,
#     )
#
#     f_qp = -u_d
#
#     u = solve_qp_single_torch(
#         H_qp,
#         f_qp,
#         A_qp,
#         b_qp,
#     ).reshape(6, 1)
#
#     # ----------------------------------------------------------
#     # 6. Verification
#     # ----------------------------------------------------------
#     psi1_t = (
#         L_beta_t @ u
#         + eta * (beta - beta_safe)
#     )
#
#     # ----------------------------------------------------------
#     # 7. Numerical checks
#     # ----------------------------------------------------------
#     checks = {
#         "H": H_t,
#         "xi": xi_t,
#         "L_beta": L_beta_t,
#         "xi_vf": xi_vf,
#         "dist_vf": dist_vf,
#         "u": u,
#         "psi1": psi1_t,
#         "d0": d0_t,
#     }
#
#     for name, value in checks.items():
#         if not torch.isfinite(value).all():
#             raise FloatingPointError(
#                 f"[t={t:.4f}] NaN/Inf in {name}"
#             )
#
#     if u.abs().max().item() > 1e2:
#         print(
#             f"[t={t}] u HUGE: "
#             f"{u.abs().max().item():.3e}"
#         )
#
#     # ----------------------------------------------------------
#     # 8. Logging
#     # ----------------------------------------------------------
#     logs = (
#         dist_vf.detach().cpu().numpy(),
#         beta,
#         xi_vf.detach().cpu().numpy().reshape(-1, 1),
#         L_beta_t.detach().cpu().numpy(),
#         psi1_t.item(),
#     )
#
#     return u, logs


def control_1st_torch(
    uav_model,
    obstacles,
    t,
    curve,
    curve_derivative,
    ctx,
    k1=2.0,
    k2=0.5,
    kt1=1.0,
    kt2=1.0,
    kt3=1.0,
    kn1=1.0,
    kn2=1.0,
    delta_acl=1e-4,
    gamma=2.0,
    eps=1e-3,
    k_shape=2,
    eta=10,
    dt_fd=1e-4,
    Kp_xi=5.0,
    beta_safe=0.1,
):
    device = ctx.device
    dtype = ctx.dtype

    # ----------------------------------------------------------
    # 1. UAV state
    # ----------------------------------------------------------
    H_t = _to_torch(
        uav_model.htm,
        device,
        dtype,
    ).reshape(4, 4)

    xi_t = _to_torch(
        uav_model.xi,
        device,
        dtype,
    ).reshape(6, 1)

    # ----------------------------------------------------------
    # 2. Obstacle barrier gradient
    # ----------------------------------------------------------
    d0_t, L_beta_t, beta = get_beta_grad_torch(
        uav_model,
        obstacles,
        gamma=gamma,
        eps=eps,
        k=k_shape,
        device=device,
        dtype=dtype,
    )

    L_beta_t = L_beta_t.reshape(1, 6)

    # ----------------------------------------------------------
    # 3. Nominal vector field
    # ----------------------------------------------------------
    _, xi_vf, dist_vf, idx_vf = tracking_xi_dot_torch(
        ctx,
        H_t,
        xi_t.squeeze(-1),
        dt_fd=dt_fd,
        kt1=kt1,
        kt2=kt2,
        kt3=kt3,
        kn1=kn1,
        kn2=kn2,
        Kp_xi=Kp_xi,
    )

    # First-order nominal command is the vector-field twist.
    u_d = xi_vf.reshape(6)

    # ----------------------------------------------------------
    # 4. First-order CBF constraint
    # ----------------------------------------------------------
    # L_beta @ u >= -eta * (beta - beta_safe)

    b_cbf = -eta * (beta - beta_safe)

    A_qp = L_beta_t

    b_qp = torch.tensor(
        [b_cbf],
        dtype=dtype,
        device=device,
    )

    # ----------------------------------------------------------
    # 5. QP
    # ----------------------------------------------------------
    H_qp = (
        1.0 + 1e-4
    ) * torch.eye(
        6,
        dtype=dtype,
        device=device,
    )

    f_qp = -u_d

    u = solve_qp_single_torch(
        H_qp,
        f_qp,
        A_qp,
        b_qp,
    ).reshape(6, 1)

    # ----------------------------------------------------------
    # 6. Verification
    # ----------------------------------------------------------
    psi1_t = (
        L_beta_t @ u
        + eta * (beta - beta_safe)
    )

    # ----------------------------------------------------------
    # 7. Numerical checks
    # ----------------------------------------------------------
    checks = {
        "H": H_t,
        "xi": xi_t,
        "L_beta": L_beta_t,
        "xi_vf": xi_vf,
        "dist_vf": dist_vf,
        "u": u,
        "psi1": psi1_t,
        "d0": d0_t,
    }

    for name, value in checks.items():
        if not torch.isfinite(value).all():
            raise FloatingPointError(
                f"[t={t:.4f}] NaN/Inf in {name}"
            )

    if u.abs().max().item() > 1e2:
        print(
            f"[t={t}] u HUGE: "
            f"{u.abs().max().item():.3e}"
        )

    # ----------------------------------------------------------
    # 8. Logging
    # ----------------------------------------------------------
    logs = (
        dist_vf.detach().cpu().numpy(),
        beta,
        xi_vf.detach().cpu().numpy().reshape(-1, 1),
        L_beta_t.detach().cpu().numpy(),
        psi1_t.item(),
        {}
    )

    return u, logs


def control_2nd_torch(
    uav_model,
    obstacles,
    t,
    curve,
    curve_derivative,
    ctx,
    k1=2.0,
    k2=0.5,
    kt1=1.0,
    kt2=1.0,
    kt3=1.0,
    kn1=1.0,
    kn2=1.0,
    delta_acl=1e-4,
    gamma=2.0,
    eps=1e-3,
    k_shape=2,
    eta=10,
    dt_fd=1e-4,
    Kp_xi=5.0,
    beta_safe=0.1,
    debug=False,
):
    device = ctx.device
    dtype = ctx.dtype

    def check_finite(name, x):
        if torch.is_tensor(x):
            valid = torch.isfinite(x).all().item()
        else:
            valid = np.isfinite(x).all()

        if not valid:
            raise FloatingPointError(
                f"[t={t:.4f}] Non-finite values in {name}"
            )

    # ----------------------------------------------------------
    # 1. UAV state
    # ----------------------------------------------------------
    H_t = _to_torch(
        uav_model.htm, device, dtype
    ).reshape(4, 4)

    xi_t = _to_torch(
        uav_model.xi, device, dtype
    ).reshape(6, 1)

    check_finite("H", H_t)
    check_finite("xi", xi_t)

    # ----------------------------------------------------------
    # 2. Aclecian
    # ----------------------------------------------------------
    A_acl_t, Lf_t, beta = get_aclecian(
        uav_model,
        obstacles,
        delta=delta_acl,
        gamma=gamma,
        eps=eps,
        k=k_shape,
        device=device,
        dtype=torch.float64,
    )

    A_acl_t = A_acl_t.to(
        device=device,
        dtype=dtype,
    )

    Lf_t = Lf_t.reshape(1, 6).to(
        device=device,
        dtype=dtype,
    )

    beta = beta.to(
        device=device,
        dtype=dtype,
    )
    check_finite("Aclecian", A_acl_t)
    check_finite("Lf", Lf_t)

    # ----------------------------------------------------------
    # 3. Second-order CBF
    # ----------------------------------------------------------
    quad_t = (xi_t.T @ A_acl_t @ xi_t).reshape(())

    lin_t = (Lf_t @ xi_t).reshape(())

    b_term = eta * k2 * (beta - beta_safe)

    d_cbf_t = (
        quad_t
        + (eta + k2) * lin_t
        + b_term
    )

    check_finite("quad", quad_t)
    check_finite("lin", lin_t)
    check_finite("d_cbf", d_cbf_t)

    # ----------------------------------------------------------
    # 4. Nominal acceleration
    # ----------------------------------------------------------
    xi_dot_des, xi_vf, dist_vf, idx_vf = tracking_xi_dot_torch(
        ctx,
        H_t,
        xi_t.squeeze(-1),
        dt_fd=dt_fd,
        kt1=kt1,
        kt2=kt2,
        kt3=kt3,
        kn1=kn1,
        kn2=kn2,
        Kp_xi=Kp_xi,
    )

    xi_dot_des = xi_dot_des.reshape(6)

    check_finite("xi_dot_des", xi_dot_des)
    check_finite("xi_vf", xi_vf)
    check_finite("dist_vf", dist_vf)

    # ----------------------------------------------------------
    # 5. QP
    # ----------------------------------------------------------
    H_qp = (
        1.0 + 1e-4
    ) * torch.eye(
        6,
        dtype=dtype,
        device=device,
    )

    f_qp = -xi_dot_des

    # L_beta @ xi_dot_safe >= -d_cbf
    A_qp = Lf_t
    b_qp = -d_cbf_t.reshape(1)

    xi_dot_safe = solve_qp_single_torch(
        H_qp,
        f_qp,
        A_qp,
        b_qp,
    ).reshape(6, 1)

    check_finite("xi_dot_safe", xi_dot_safe)

    # ----------------------------------------------------------
    # 6. CBF verification
    # ----------------------------------------------------------
    # d_cbf already includes:
    # quad + (eta + k2)*lin + eta*k2*(beta-beta_safe)
    psi2_t = (
        Lf_t @ xi_dot_safe
        + d_cbf_t.reshape(1, 1)
    )

    check_finite("psi2", psi2_t)

    # ----------------------------------------------------------
    # 7. Inverse dynamics
    # ----------------------------------------------------------
    u = uav_model.inv_dynamics(
        H_t,
        xi_t,
        xi_dot_safe,
    )

    check_finite("u", u)

    # ----------------------------------------------------------
    # 8. Forward/inverse dynamics consistency
    # ----------------------------------------------------------
    # Applying u through forward dynamics should reproduce
    # xi_dot_safe.
    xi_dot_recovered = uav_model.dynamics(
        u,
        H_t,
        xi_t,
    )

    check_finite("xi_dot_recovered", xi_dot_recovered)

    dyn_error = torch.linalg.norm(
        xi_dot_recovered - xi_dot_safe
    )

    # ----------------------------------------------------------
    # 9. Debug
    # ----------------------------------------------------------
    if debug:
        print(f"\n========== CONTROL DEBUG t={t:.4f} ==========")

        print(f"beta             = {beta:.8e}")
        print(f"quad             = {quad_t.item():.8e}")
        print(f"lin              = {lin_t.item():.8e}")
        print(f"b_term           = {b_term:.8e}")
        print(f"d_cbf            = {d_cbf_t.item():.8e}")
        print(f"psi2             = {psi2_t.item():.8e}")

        print(
            f"max|xi_dot_des|  = "
            f"{xi_dot_des.abs().max().item():.8e}"
        )
        print(
            f"max|xi_dot_safe| = "
            f"{xi_dot_safe.abs().max().item():.8e}"
        )
        print(
            f"max|u|           = "
            f"{u.abs().max().item():.8e}"
        )

        print(f"dynamics error   = {dyn_error.item():.8e}")

        print("============================================\n")

    if psi2_t.item() < -1e-6:
        print(
            f"[WARNING t={t:.4f}] CBF constraint violated: "
            f"psi2={psi2_t.item():.6e}"
        )

    if dyn_error.item() > 1e-6:
        print(
            f"[WARNING t={t:.4f}] "
            f"Inverse/forward dynamics mismatch: "
            f"{dyn_error.item():.6e}"
        )

    # ----------------------------------------------------------
    # 10. Logs
    # ----------------------------------------------------------
    other = (
        quad_t.item(),
        lin_t.item(),
        b_term,
        d_cbf_t.item(),
    )

    logs = (
        dist_vf.detach().cpu().numpy(),
        beta,
        xi_vf.detach().cpu().numpy().reshape(-1, 1),
        Lf_t.detach().cpu().numpy(),
        psi2_t.item(),
        other,
    )

    return u, logs

# ---------------------------- TORCH-BASED ----------------------------
def tracking_xi_dot_torch(
    ctx,
    H,
    xi_actual,
    dt_fd=1e-4,
    kt1=1.0,
    kt2=1.0,
    kt3=1.0,
    kn1=1.0,
    kn2=1.0,
    Kp_xi=5.0,
):
    """
    H:          (4, 4) or (N, 4, 4)
    xi_actual:  (6,)   or (N, 6)
    Returns: (xi_dot_des, xi_vf, dist, idx) — squeezed if unbatched.
    """
    H = _to_torch(H, ctx.device, ctx.dtype)
    xi_actual = _to_torch(xi_actual, ctx.device, ctx.dtype)
    squeeze = H.dim() == 2
    if squeeze:
        H = H.unsqueeze(0)
    # Normalize xi to (N, 6) — accept (6,), (6, 1), (1, 6)
    if xi_actual.dim() == 1:
        xi_actual = xi_actual.unsqueeze(0)  # (1, 6)
    elif xi_actual.dim() == 2 and xi_actual.shape == (6, 1):
        xi_actual = xi_actual.squeeze(-1).unsqueeze(0)  # (6, 1) → (1, 6)
    elif xi_actual.dim() == 2 and xi_actual.shape[1] == 6:
        pass  # already (N, 6)
    else:
        raise ValueError(f"Unexpected xi shape {xi_actual.shape}")
    N = H.shape[0]

    # ---- three poses in one batch: [H, H_fwd, H_bwd] ----
    dxi = xi_actual * dt_fd  # (N, 6)
    dH_pos = exp_se3(smap_se3(dxi))  # (N, 4, 4)
    dH_neg = exp_se3(smap_se3(-dxi))
    H_batch = torch.cat([H, dH_pos @ H, dH_neg @ H], dim=0)  # (3N, 4, 4)

    twist, dist, idx = ctx.eval_vf(H_batch, kt1, kt2, kt3, kn1, kn2)

    xi_vf = twist[:N]  # (N, 6)
    xi_fwd = twist[N : 2 * N]
    xi_bwd = twist[2 * N :]

    xi_dot_ff = (xi_fwd - xi_bwd) / (2.0 * dt_fd)  # (N, 6)
    xi_dot_des = xi_dot_ff + Kp_xi * (xi_vf - xi_actual)  # (N, 6)

    if squeeze:
        return (
            xi_dot_des.squeeze(0),
            xi_vf[0].squeeze(0),
            dist[0].squeeze(0),
            idx[0].squeeze(0),
        )
    return xi_dot_des, xi_vf, dist, idx


def solve_qp_single_torch(H, f, A, b):
    """min 0.5 uᵀ H u + fᵀ u   s.t.   A u ≥ b,   A (1,n), b (1,)."""
    Hinv = torch.linalg.inv(H)
    u_star = -Hinv @ f  # unconstrained
    slack = (A @ u_star) - b  # (1,)
    if slack.squeeze() >= 0:
        return u_star
    AHinvAT = (A @ Hinv @ A.T).squeeze()  # scalar > 0
    lam = (-slack.squeeze()) / AHinvAT  # KKT multiplier > 0
    return u_star + lam * (Hinv @ A.T).squeeze(-1)  # boundary solution


print(torch.__version__)
# %%
curve_center = np.asarray([1.0, 0.1, 0.5])
curve, dcurve = make_circle(n_points=2000, radius=2.0, center=curve_center)
curve_derivative = dcurve
ctx = TorchVFContext(curve, dcurve)  # caches curve & VF
# htm0 = Utils.trn([0.0, 0.0, 1.0])
htm0 = curve[0]
drone = UAV(
    htm=htm0,
    inertia=np.ones(3),
    mass=20,
    device=ctx.device,
    dtype=ctx.dtype,
)
# obstacle_htm = Utils.trn([12.5, 0.1, 0.5])
obstacle_htm = np.asarray(Utils.trn([2.0, 0.1, 0.5]))
obstacle = Box(htm=obstacle_htm, width=1, height=1, depth=1, opacity=0.6)
ub_curve = CurveSE3(points=curve, size=0.05, frame_size=0.1, num_frames=10)
kn1, kn2 = 1.0, 1.0
kt1, kt2, kt3 = 0.3, 1.0, kn2
type = "dyn"
# type = "kin"
# type = "dyn"
if type == "dyn":
    eta = 5.0
else:
    eta = 10.0  # kinematic
control = control_1st_torch if "kin" in type else control_2nd_torch

sim = Simulation.create_sim_grid([drone.collision_model, obstacle, ub_curve])
sim.set_parameters(width=1080, height=720)

T, dt = 6.0, 1e-2
log = []
debug = []

completed = False

for t in tqdm(np.arange(0, T, dt)):
    # Snapshot before
    if torch.isnan(ctx.curve).any():
        print(f"[t= {t}] curve has NaN")
        break
    # if not np.all(np.isfinite(drone.htm)):
    #     print(f"[t= {t}] drone.htm has NaN BEFORE step")
    #     break
    # if not np.all(np.isfinite(drone.xi)):
    #     print(f"[t= {t}] drone.xi has NaN BEFORE step")
    #     break
    u, (dist_curve, dist_obs, xi_vf, L_beta, psi1, other) = control(
        uav_model=drone,
        obstacles=[obstacle],
        t=t,
        curve=curve,
        curve_derivative=dcurve,
        ctx=ctx,
        dt_fd=1e-3,
        Kp_xi=20.0,
        eta=eta,
        k2=0.5,
        beta_safe=0.04,
        eps=1e-2,
        kn1=kn1,
        kn2=kn2,
        kt1=kt1,
        kt2=kt2,
        kt3=kt3,
    )
    track_error = np.linalg.norm(
        xi_vf.ravel() - drone.xi.detach().cpu().numpy().ravel()
    )
    drone.step(t, dt, u, type=type)
    # if not np.all(np.isfinite(drone.htm)):
    #     print(f"[t= {t}] drone.htm has NaN AFTER step")
    #     break
    log.append(
        (
            t,
            drone.htm[:3, 3].detach().cpu().numpy().copy(),
            dist_curve,
            dist_obs,
            track_error,
            L_beta,
            psi1,
        )
    )
    debug.append(other)
    if t > (T - dt):
        completed = True

# for t in tqdm(np.arange(0, T, dt)):
#     u, (dist_curve, dist_obs, xi_vf, L_beta, psi1, other) = control(
#         uav_model=drone,
#         obstacles=[obstacle],
#         t=t,
#         curve=curve,
#         curve_derivative=dcurve,
#         dt_fd=1e-3,
#         Kp_xi=20.0,
#         # eta=10.0, # kinematic
#         # eta=5.0,
#         eta=eta,
#         k2=0.5,
#         beta_safe=0.04,
#         eps=1e-2,
#         kn1=kn1,
#         kn2=kn2,
#         kt1=kt1,
#         kt2=kt2,
#         kt3=kt3,
#     )
#     track_error = np.linalg.norm(xi_vf.ravel() - drone.xi.ravel())
#     drone.step(t, dt, u, type=type)
#     log.append(
#         (
#             t,
#             np.asarray(drone.htm)[:3, 3].copy(),
#             dist_curve,
#             dist_obs,
#             track_error,
#             L_beta,
#             psi1,
#         )
#     )
#
#     debug.append(other)

sim.run_in_browser()

dist_log = [d for _, _, d, *_ in log]
dist_obs_log = [d for _, _, _, d, *_ in log]
velocity_error = [d for _, _, _, _, d, *_ in log]
norm_L_beta = [np.linalg.norm(d) for *_, d, _ in log]
L_betas = np.asarray([d for *_, d, _ in log]).reshape(-1, 6)
psis = [d for *_, d in log]

debug_arr = np.asarray(debug)
import plotly.express as px

if completed:
    px.line(dist_log)
    px.line(dist_obs_log)
    px.line(velocity_error)
    px.line(norm_L_beta)
    px.line(L_betas)
    px.line(psis)

    for i in range(debug_arr.shape[1]):
        print(i)
        # other = (quad, lin, b_term, d_cbf)
        fig = px.line(debug_arr[:, i])
        fig.show()
    print(2)

np.mean(velocity_error)

# %%
# """TESTS"""
# htm0 = Utils.trn([0.0, 0.0, 0.0])
# drone = UAV(htm=htm0, inertia=np.ones(3), mass=20, height=0.1, width=0.1, depth=0.1)
# H0 = np.asarray(drone.htm, dtype=float)
#
# for axis in range(3):
#     print(f"Obstacle at 1.0m in axis {axis} (0-x, 1-y, 2-z)")
#     htm_pos = np.zeros(3)
#     htm_pos[axis] = 1.0
#     htm_obs = Utils.trn(htm_pos)
#     obstacle = Box(htm=htm_obs, width=1, height=1, depth=1)
#     _, L_beta, beta0 = get_beta_grad(drone, [obstacle])
#
#     delta_fd = 1e-6
#     for i in range(6):
#         e = np.zeros(6)
#         e[i] = 1.0
#         drone.htm = expSE3(SmapSE3(e * delta_fd)) @ H0
#         _, _, bp = get_beta_grad(drone, [obstacle])
#         drone.htm = expSE3(SmapSE3(-e * delta_fd)) @ H0
#         _, _, bm = get_beta_grad(drone, [obstacle])
#         L_fd = (bp - bm) / (2 * delta_fd)
#         print(
#             f"[{i}] L_an={L_beta[i]:+.6e}  L_fd={L_fd:+.6e}  "
#             f"err={L_beta[i]-L_fd:+.3e}  rel={abs(L_beta[i]-L_fd)/(abs(L_fd)+1e-12):.3e}"
#         )
#     drone.htm = H0
#
# # %%
# """TESTS 3"""
# p = drone.htm[:3, 3]
# p_obs = np.asarray(obstacle.htm[:3, 3]).ravel()
# toward = p_obs - p
# toward /= np.linalg.norm(toward)
#
# H0 = drone.htm.copy()
#
# drone.htm = expSE3(SmapSE3(np.concatenate([0.01 * toward, np.zeros(3)]))) @ H0
# _, _, b_toward = get_beta_grad(drone, [obstacle])
# drone.htm = H0
#
# drone.htm = expSE3(SmapSE3(np.concatenate([-0.01 * toward, np.zeros(3)]))) @ H0
# _, _, b_away = get_beta_grad(drone, [obstacle])
# drone.htm = H0
#
# _, _, b0 = get_beta_grad(drone, [obstacle])
#
# print(f"beta at start            = {b0:.6f}")
# print(f"beta moved toward obs    = {b_toward:.6f}   should be < {b0:.6f}")
# print(f"beta moved away   obs    = {b_away:.6f}   should be > {b0:.6f}")
# # %%
# """TEST 4"""
# reduce_fn = lambda d: shaping_function(
#     holder_min(d, 2, dim=1), 2, eps=1e-3
# ).sum()
# dists, grads = holder_distance_objects_with_grad(
#     [drone.collision_model],
#     [obstacle],
#     htms_a=[drone.htm],
#     gamma=2.0,
#     eps=1e-3,
#     device=None,
#     reduce_fn=reduce_fn,
# )
# d_raw = dists.detach().cpu().numpy()[0, 0]  # raw (unshaped, unmin'd) distance
# d_min = holder_min(dists.detach().cpu(), gamma=2.0, dim=1).item()
# d_shap = shaping_function(
#     holder_min(dists.detach().cpu(), gamma=2.0, dim=1), 2, eps=1e-3
# ).item()
#
# print(f"raw d         = {d_raw:.6f}")
# print(f"holder_min    = {d_min:.6f}")
# print(f"shaped beta   = {d_shap:.6f}")
# print(f"euclidean obs = {np.linalg.norm(drone.htm[:3,3] - obstacle.htm[:3,3]):.6f}")
