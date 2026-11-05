from typing import Optional, Tuple, Union
import torch


class SE3VectorField:
    """
    Batched, GPU-friendly vector field on SE(3).

    All methods operate on tensors of shape (..., 4, 4) (or (..., 6) for
    Lie algebra vectors) and use broadcasting so that the whole
    `vectorfield_SE3` computation is a single fused pipeline.
    """

    def __init__(
        self,
        delta: float = 1e-3,
        ds: float = 1e-3,
        max_cos_theta: float = 0.999,
        theta_zero: float = 1e-6,
        normalize_tangent: bool = False,
    ):
        self.delta = delta
        self.ds = ds
        self.max_cos_theta = max_cos_theta
        self.theta_zero = theta_zero
        self.normalize_tangent = normalize_tangent

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _eye(n: int, like: torch.Tensor) -> torch.Tensor:
        return torch.eye(n, device=like.device, dtype=like.dtype)

    # ------------------------------------------------------------------
    # EEdistSE3Variables – fully vectorised
    # ------------------------------------------------------------------
    def eedist_se3_variables(self, X: torch.Tensor):
        """
        X: (..., 4, 4)
        Returns Q, t, M, theta, cos_theta, sin_theta, alpha
        all batched with shape (...,) / (..., 3, 3) / (..., 3)
        """
        Q = X[..., :3, :3]                        # (..., 3, 3)
        t = X[..., :3, 3]                         # (..., 3)

        Q_inv = torch.linalg.inv(Q)               # batched inverse

        trace = Q.diagonal(dim1=-2, dim2=-1).sum(-1)   # (...,)
        cos_theta0 = 0.5 * (trace - 1.0)

        diff = Q - Q_inv
        fro_norm = torch.linalg.matrix_norm(diff, ord="fro")     # (...,)
        s2 = torch.sqrt(torch.tensor(2.0, device=X.device, dtype=X.dtype))
        sin_theta0 = fro_norm / (2.0 * s2)

        theta = torch.atan2(sin_theta0, cos_theta0)              # (...,)
        cos_theta = torch.cos(theta)
        sin_theta = torch.sin(theta)

        # ------------------------------------------------------------------
        # alpha with numeric safety for cos_theta ~ 1 (uses torch.where
        # which still evaluates both branches => guard the denominator).
        # ------------------------------------------------------------------
        mask = cos_theta > self.max_cos_theta
        numerator = 2.0 - 2.0 * cos_theta - theta**2
        denom = 4.0 * (1.0 - cos_theta) ** 2
        denom_safe = torch.where(mask, torch.ones_like(denom), denom)
        alpha_general = numerator / denom_safe
        alpha = torch.where(
            mask, torch.full_like(cos_theta, -1.0 / 12.0), alpha_general
        )

        I3 = self._eye(3, Q)
        M = (
            alpha[..., None, None] * (Q + Q_inv)
            + (1.0 - 2.0 * alpha[..., None, None]) * I3
        )

        return Q, t, M, theta, cos_theta, sin_theta, alpha

    # ------------------------------------------------------------------
    # EEdistance – broadcasting over arbitrary batch shapes
    # ------------------------------------------------------------------
    def eedistance(self, V: torch.Tensor, W: torch.Tensor) -> torch.Tensor:
        """
        V, W: (..., 4, 4) with broadcasting.
        """
        Z = torch.linalg.inv(V) @ W               # (..., 4, 4)
        Q, t, M, theta, _, _, _ = self.eedist_se3_variables(Z)

        t_col = t.unsqueeze(-1)                   # (..., 3, 1)
        tMt = (t.unsqueeze(-2) @ M @ t_col).squeeze(-1).squeeze(-1)
        return torch.sqrt(2.0 * theta**2 + tMt)

    # ------------------------------------------------------------------
    # ECdistance – closest point on curve (chunked to bound memory)
    # ------------------------------------------------------------------
    def ecdistance(
        self,
        states: torch.Tensor,                    # (N, 4, 4)
        curve: torch.Tensor,                     # (M, 4, 4)
        chunk_size: int = 1024,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns (min_dists: (N,), closest_indices: (N,)).
        """
        N = states.shape[0]
        min_dists = torch.empty(N, device=states.device, dtype=states.dtype)
        min_indices = torch.empty(N, dtype=torch.long, device=states.device)

        W = curve.unsqueeze(0)                   # (1, M, 4, 4)
        for start in range(0, N, chunk_size):
            end = min(start + chunk_size, N)
            V = states[start:end].unsqueeze(1)   # (C, 1, 4, 4)
            d = self.eedistance(V, W)            # (C, M)
            d_min, i_min = d.min(dim=1)
            min_dists[start:end] = d_min
            min_indices[start:end] = i_min
        return min_dists, min_indices

    # ------------------------------------------------------------------
    # Smap / invSmap
    # ------------------------------------------------------------------
    def smap_se3(self, xi: torch.Tensor) -> torch.Tensor:
        """xi: (..., 6) -> (..., 4, 4)"""
        S = torch.zeros(*xi.shape[:-1], 4, 4, device=xi.device, dtype=xi.dtype)
        S[..., 0, 1] = -xi[..., 5]
        S[..., 0, 2] = xi[..., 4]
        S[..., 1, 2] = -xi[..., 3]
        S = S - S.transpose(-2, -1)
        S[..., 0, 3] = xi[..., 0]
        S[..., 1, 3] = xi[..., 1]
        S[..., 2, 3] = xi[..., 2]
        return S

    def inv_smap_se3(self, A: torch.Tensor) -> torch.Tensor:
        """A: (..., 4, 4) -> (..., 6)"""
        xi = torch.zeros(*A.shape[:-2], 6, device=A.device, dtype=A.dtype)
        xi[..., 0] = A[..., 0, 3]
        xi[..., 1] = A[..., 1, 3]
        xi[..., 2] = A[..., 2, 3]
        xi[..., 3] = -A[..., 1, 2]
        xi[..., 4] = A[..., 0, 2]
        xi[..., 5] = -A[..., 0, 1]
        return xi

    # ------------------------------------------------------------------
    # expSE3 – batched, branchless
    # ------------------------------------------------------------------
    def exp_se3(self, X: torch.Tensor) -> torch.Tensor:
        """X: (..., 4, 4) -> exp(X) in SE(3)."""
        A = X[..., :3, :3]
        v = X[..., :3, 3]

        theta = torch.sqrt(A[..., 1, 0] ** 2 + A[..., 0, 2] ** 2 + A[..., 2, 1] ** 2)
        small = theta < self.theta_zero
        theta_safe = torch.where(small, torch.ones_like(theta), theta)

        sin_t = torch.sin(theta_safe)
        cos_t = torch.cos(theta_safe)

        c1 = torch.where(small, torch.ones_like(theta), sin_t / theta_safe)
        c2 = torch.where(small, torch.zeros_like(theta), (1.0 - cos_t) / theta_safe**2)
        c3 = torch.where(
            small, torch.zeros_like(theta), (theta_safe - sin_t) / theta_safe**3
        )

        I3 = self._eye(3, A)
        AA = A @ A
        R = I3 + c1[..., None, None] * A + c2[..., None, None] * AA
        U = I3 + c2[..., None, None] * A + c3[..., None, None] * AA

        t = (U @ v.unsqueeze(-1)).squeeze(-1)

        # small-theta fallback: R = I, t = v
        R = torch.where(small[..., None, None], I3.expand_as(R), R)
        t = torch.where(small[..., None], v, t)

        result = torch.zeros_like(X)
        result[..., :3, :3] = R
        result[..., :3, 3] = t
        result[..., 3, 3] = 1.0
        return result

    # ------------------------------------------------------------------
    # Main vector field – everything batched
    # ------------------------------------------------------------------
    def vectorfield_se3(
        self,
        states: torch.Tensor,                    # (N, 4, 4)
        curve: torch.Tensor,                     # (M, 4, 4)
        kt1: float,
        kt2: float,
        kt3: float,
        kn1: float,
        kn2: float,
        curve_derivative: Optional[Union[torch.Tensor, list]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns (twist: (N, 6), distances: (N,), closest_indices: (N,)).
        """
        device, dtype = states.device, states.dtype
        N, M = states.shape[0], curve.shape[0]

        # ---- closest point on curve for each state (batched) --------------
        min_dists, closest_indices = self.ecdistance(states, curve)
        closest_points = curve[closest_indices]                       # (N, 4, 4)

        # ---- normal: finite differences over 6 basis dirs (batched) ------
        I6 = self._eye(6, states)
        basis = self.smap_se3(I6)                                     # (6, 4, 4)

        # (6, 1, 4, 4) @ (1, N, 4, 4) -> (6, N, 4, 4)
        variations = self.exp_se3(basis[:, None, :, :] * self.delta) @ states.unsqueeze(0)

        variations_flat = variations.reshape(6 * N, 4, 4)
        closest_expanded = closest_points.repeat(6, 1, 1)             # (6N, 4, 4)
        dists_pert = self.eedistance(variations_flat, closest_expanded).reshape(6, N)

        LvDhat = (dists_pert - min_dists.unsqueeze(0)) / self.delta    # (6, N)
        normal = -LvDhat.t().contiguous()                             # (N, 6)

        # ---- tangent: forward-difference or supplied derivative ----------
        if curve_derivative is None:
            next_indices = (closest_indices + 1) % M                  # closed curve
            dHd = (curve[next_indices] - closest_points) / self.ds    # (N, 4, 4)
        else:
            if not isinstance(curve_derivative, torch.Tensor):
                curve_derivative = torch.stack(
                    [torch.as_tensor(d, device=device, dtype=dtype)
                     for d in curve_derivative]
                )
            dHd = curve_derivative[closest_indices]

        tangent = self.inv_smap_se3(dHd @ torch.linalg.inv(closest_points))  # (N, 6)
        if self.normalize_tangent:
            tangent = tangent / tangent.norm(dim=-1, keepdim=True).clamp_min(1e-12)

        # ---- distance-dependent gains ------------------------------------
        eps = 1e-3
        dist_mod = torch.sqrt(min_dists + eps**2) - eps
        kn = kn1 * torch.tanh(kn2 * dist_mod)                          # (N,)
        kt = kt1 * (1.0 - kt2 * torch.tanh(kt3 * dist_mod))            # (N,)

        twist = kt.unsqueeze(-1) * tangent + kn.unsqueeze(-1) * normal # (N, 6)
        return twist, min_dists, closest_indices


# ----------------------------------------------------------------------
# Functional shortcut (mirrors the C++ signature)
# ----------------------------------------------------------------------
def vectorfield_se3(
    states: torch.Tensor,
    curve: torch.Tensor,
    kt1: float,
    kt2: float,
    kt3: float,
    kn1: float,
    kn2: float,
    curve_derivative: Optional[Union[torch.Tensor, list]] = None,
    **kwargs,
):
    return SE3VectorField(**kwargs).vectorfield_se3(
        states, curve, kt1, kt2, kt3, kn1, kn2, curve_derivative
    )
