import torch
import numpy as np
import uaibot as ub
from tqdm import tqdm
from uaibot.simobjects.curve import CurveSE3
from uaibot.simobjects import Box
from uaibot.gpu.lie import exp_se3, smap_se3, inv_smap_se3
from gcbf.curves import build_curve
from gcbf.uav import (
    _to_torch,
    UAV,
    TorchVFContext,
    get_beta_grad_torch,
    solve_qp_single_torch,
)


def _skew_batch(x: torch.Tensor) -> torch.Tensor:
    """(..., 3) -> (..., 3, 3)."""
    x0, x1, x2 = x.unbind(dim=-1)
    z = torch.zeros_like(x0)

    return torch.stack(
        [
            torch.stack([z, -x2, x1], dim=-1),
            torch.stack([x2, z, -x0], dim=-1),
            torch.stack([-x1, x0, z], dim=-1),
        ],
        dim=-2,
    )


def _vee_skew_part(A: torch.Tensor) -> torch.Tensor:
    """Vee of the skew-symmetric part of (..., 3, 3)."""
    return 0.5 * torch.stack(
        [
            A[..., 2, 1] - A[..., 1, 2],
            A[..., 0, 2] - A[..., 2, 0],
            A[..., 1, 0] - A[..., 0, 1],
        ],
        dim=-1,
    )


def _ad_se3(xi: torch.Tensor) -> torch.Tensor:
    """Batched ad_xi, (..., 6) -> (..., 6, 6)."""
    if xi.shape[-2:] == (6, 1):
        xi = xi.squeeze(-1)

    v = xi[..., :3]
    omega = xi[..., 3:]

    S_v = _skew_batch(v)
    S_w = _skew_batch(omega)
    zero = torch.zeros_like(S_w)

    return torch.cat(
        [
            torch.cat([S_w, S_v], dim=-1),
            torch.cat([zero, S_w], dim=-1),
        ],
        dim=-2,
    )


def _coad_se3(xi: torch.Tensor) -> torch.Tensor:
    """Batched -ad_xi.T @ xi, (..., 6) -> (..., 6, 1)."""
    if xi.shape[-2:] == (6, 1):
        xi = xi.squeeze(-1)

    v = xi[..., :3]
    omega = xi[..., 3:]

    linear = torch.linalg.cross(omega, v, dim=-1)
    angular = torch.zeros_like(omega)

    return torch.cat([linear, angular], dim=-1).unsqueeze(-1)


def _diff12(
    F: torch.Tensor,
    ds: float,
    periodic: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    First and second derivatives along dimension 0.

    Uniform sampling in s.
    Second-order finite differences, including open endpoints.
    """
    if periodic:
        Fm = torch.roll(F, 1, dims=0)
        Fp = torch.roll(F, -1, dims=0)

        Fs = (Fp - Fm) / (2.0 * ds)
        Fss = (Fp - 2.0 * F + Fm) / (ds * ds)
        return Fs, Fss

    Fs = torch.empty_like(F)
    Fss = torch.empty_like(F)

    # Interior
    Fs[1:-1] = (F[2:] - F[:-2]) / (2.0 * ds)
    Fss[1:-1] = (F[2:] - 2.0 * F[1:-1] + F[:-2]) / (ds * ds)

    # Open endpoints
    Fs[0] = (-3.0 * F[0] + 4.0 * F[1] - F[2]) / (2.0 * ds)
    Fs[-1] = (3.0 * F[-1] - 4.0 * F[-2] + F[-3]) / (2.0 * ds)

    Fss[0] = (2.0 * F[0] - 5.0 * F[1] + 4.0 * F[2] - F[3]) / (ds * ds)
    Fss[-1] = (2.0 * F[-1] - 5.0 * F[-2] + 4.0 * F[-3] - F[-4]) / (ds * ds)

    return Fs, Fss


def get_L_ell_s(
    C: torch.Tensor,
    gamma: float,
    ds: float = None,
    periodic: bool = True,
) -> torch.Tensor:
    """
    Compute ell_s for a discretized SE(3) curve at fixed t.

    C: (N, 4, 4), uniformly sampled in s.
    gamma: scalar gain.
    ds: parameter spacing.
    periodic: True for a periodic curve sampled without
              duplicating the endpoint.

    Returns: (N, 6, 1).

    Requires N >= 4 for open curves, N >= 3 for periodic.
    """
    if ds is None:
        ds = C.shape[0] - 1

    N = C.shape[0]
    if N < (3 if periodic else 4):
        raise ValueError("Insufficient curve samples.")
    if ds <= 0:
        raise ValueError("ds must be positive.")

    # Only differentiate the meaningful SE(3) entries.
    F = C[:, :3, :4]
    Fs, Fss = _diff12(F, ds, periodic)

    R = F[:, :, :3]
    p = F[:, :, 3]

    Rs = Fs[:, :, :3]
    ps = Fs[:, :, 3]

    Rss = Fss[:, :, :3]
    pss = Fss[:, :, 3]

    Rt = R.transpose(-1, -2)

    # Spatial angular velocity and its s-derivative.
    omega = _vee_skew_part(Rs @ Rt)
    omega_s = _vee_skew_part(Rss @ Rt)

    # Spatial linear velocity and its s-derivative.
    v = ps - torch.linalg.cross(omega, p, dim=-1)

    v_s = (
        pss
        - torch.linalg.cross(omega_s, p, dim=-1)
        - torch.linalg.cross(omega, ps, dim=-1)
    )

    # ell_s = gamma * (-xi_s + coad_xi(xi)).
    linear = -v_s + torch.linalg.cross(omega, v, dim=-1)
    angular = -omega_s

    return gamma * torch.cat([linear, angular], dim=-1).unsqueeze(-1)


def L_half_squared_eedistance(
    V: torch.Tensor,
    W: torch.Tensor,
    zeta: float,
    scale_pos: float = None,
    max_cos_theta: float = 1e-6,
) -> torch.Tensor:
    """
    Compute L_V[0.5 * zeta * eedistance(V,W)^2]^T.

    V, W: (..., 4, 4), broadcastable SE(3) matrices.
    zeta: scalar gain.

    Returns: (..., 6, 1).

    Uses the same alpha branch and beta expression as
    L_eedistance, without computing the distance itself.
    """
    if scale_pos is None:
        scale_pos = 1.0

    s2 = scale_pos**2

    R = V[..., :3, :3]
    Rt = R.transpose(-1, -2)
    pV = V[..., :3, 3]
    pW = W[..., :3, 3]

    # Relative transformation V^{-1} W.
    Q = Rt @ W[..., :3, :3]
    t = (Rt @ (pW - pV).unsqueeze(-1)).squeeze(-1)
    Qt_mat = Q.transpose(-1, -2)

    # Rotation angle.
    skew_vec = _vee_skew_part(Q)

    cos0 = 0.5 * (Q.diagonal(dim1=-2, dim2=-1).sum(-1) - 1.0)
    sin0 = torch.linalg.vector_norm(skew_vec, dim=-1)

    theta = torch.atan2(sin0, cos0)
    cos_theta = torch.cos(theta)

    # Alpha: same branch as eedist_se3_variables.
    mask = cos_theta > max_cos_theta

    d0 = 1.0 - cos_theta
    numerator_alpha = 2.0 * d0 - theta.square()
    denominator_alpha = 4.0 * d0.square()
    denominator_safe = torch.where(
        mask,
        torch.ones_like(denominator_alpha),
        denominator_alpha,
    )

    alpha = torch.where(
        mask,
        torch.full_like(theta, -1.0 / 12.0),
        numerator_alpha / denominator_safe,
    )

    # M @ t, avoiding construction of M.
    Qt = (Q @ t.unsqueeze(-1)).squeeze(-1)
    Qtt = (Qt_mat @ t.unsqueeze(-1)).squeeze(-1)

    Mt = t + alpha.unsqueeze(-1) * (Qt + Qtt - 2.0 * t)

    # Rotation vector r = log(Q)^vee.
    sinc_theta = torch.sinc(theta / torch.pi)
    r = skew_vec / sinc_theta.unsqueeze(-1)

    # beta = alpha'(theta) / theta.
    u = theta.square()

    beta_series = -1.0 / 45.0 + u * (
        -13.0 / 3780.0
        + u * (-23.0 / 75600.0 + u * (-101.0 / 4989600.0 - u * 5263.0 / 4670265600.0))
    )

    small = theta < 0.5

    d = 2.0 * torch.sin(0.5 * theta).square()
    d_safe = torch.where(small, torch.ones_like(d), d)

    beta_general = ((u - d) * sinc_theta - d) / (2.0 * d_safe**3)

    beta = torch.where(small, beta_series, beta_general)
    beta = torch.where(mask, torch.zeros_like(beta), beta)

    # Translation-rotation coupling.
    h = -(Qt - t).square().sum(dim=-1)

    # Numerators from L_eedistance.
    a = s2 * Mt

    b = (-2.0 - 0.5 * s2 * h * beta).unsqueeze(-1) * r + (s2 * alpha).unsqueeze(
        -1
    ) * torch.linalg.cross(t, Qt, dim=-1)

    Ra = (R @ a.unsqueeze(-1)).squeeze(-1)
    Rb = (R @ b.unsqueeze(-1)).squeeze(-1)

    pW = pW + torch.zeros_like(Ra)

    numerator = torch.cat(
        [
            -Ra,
            Rb - torch.linalg.cross(pW, Ra, dim=-1),
        ],
        dim=-1,
    )

    return zeta * numerator.unsqueeze(-1)


def map_model(uav, points):
    modeled_points = []
    for point in points:
        a = uav
        a.htm = point
        modeled_points.append(a)
    return modeled_points


def deform_curve(
    uav_model,
    obstacles,
    t,
    ctx,
    holder_gamma=2,
    holder_eps=1e-3,
    func_zeta=1.0,
    func_gamma=1.0,
    func_ds=1e-3,
):
    device = ctx.device
    dtype = ctx.dtype
    curve_t = ctx.curve

    # ----------------------------------------------------------
    # 2. Functional gradients
    # ----------------------------------------------------------
    # Collision functional term
    ell_d_terms, L_ell_d, ell_d = get_beta_grad_torch(
        # uav_model,
        map_model(uav_model, curve_t),
        obstacles,
        gamma=holder_gamma,
        eps=holder_eps,
        k=holder_gamma,
        device=device,
        dtype=dtype,
    )
    L_ell_d = L_ell_d.reshape(-1, 6, 1)

    # Ellastic functional
    L_ell_e = L_half_squared_eedistance(
        curve_t,
        ctx.curve0,
        func_zeta,
        scale_pos=None,
    )
    # Smoothing functional
    L_ell_s = get_L_ell_s(
        curve_t,
        func_gamma,
        ds=func_ds,
        periodic=True,
    )

    print(f"L_edd_d {L_ell_d.shape}; L_ell_e {L_ell_e.shape}; L_ell_s {L_ell_s.shape}")
    xi_pt = (L_ell_d - L_ell_s - L_ell_e).transpose(-1, -2)
    curve_deform = exp_se3(smap_se3(xi_pt)) @ curve_t

    return curve_deform


if __name__ == "__main__":
    CURVE_N = 5000
    h, w, d = 0.4, 0.4, 0.4
    T = 10
    dt = 1e-2
    holder_eps, holder_gamma = 1e-2, 2.0
    func_zeta, func_gamma, func_ds = 1.0, 1.0, None

    curve, *_ = build_curve(
        "circle",
        radius=1.5,
        center=(0, 0, 0.5),
        n_points=CURVE_N,
    )
    print("Built curve")
    curve_ub = CurveSE3(points=curve, color='cyan', size=0.05)
    ctx = TorchVFContext(curve, None)  # caches curve & VF
    htm0 = np.eye(4)
    drone = UAV(
        htm=htm0,
        inertia=np.ones(3),
        mass=20,
        device=ctx.device,
        dtype=ctx.dtype,
    )

    obs1 = Box(
        htm=ub.Utils.trn([2.0, 0.0, 0.4]),
        width=w,
        height=h,
        depth=d,
        color="red",
        opacity=0.6,
    )
    obs2 = Box(
        htm=ub.Utils.trn([0.0, 2.0, 0.5]),
        width=w,
        height=h,
        depth=d,
        color="blue",
        opacity=0.6,
    )
    obstacles = [obs1, obs2]
    sim = ub.Simulation([curve_ub, obs1, obs2])

    for t_np in tqdm(np.arange(0, T, dt)):
        t = float(t_np)
        deformed_curve = deform_curve(
            uav_model=drone,
            obstacles=obstacles,
            t=t,
            ctx=ctx,
            holder_gamma=holder_gamma,
            holder_eps=holder_eps,
            func_zeta=func_zeta,
            func_gamma=func_gamma,
            func_ds=func_ds,
        )
        ctx.set_curve(deformed_curve)
        deformed_curve_np = deformed_curve.detach().cpu().numpy(),
        curve_ub.add_ani_frame(time=t, points=deformed_curve_np)

    sim.run_in_browser()

