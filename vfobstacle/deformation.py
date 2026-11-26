"""Vectorized deformation of a discretized SE(3) curve.

Dependencies: torch, numpy, tqdm, uaibot, gcbf.curves, gcbf.uav.

The curve evolves sequentially in time, but all samples at a fixed time
are evaluated in batches. Obstacle poses are read from the supplied obstacle
objects at EVERY time step, so obstacles can move.

IMPORTANT: the explicit scheme for the smoothing term is stiff; see notes
below and choose dt/gamma/ds appropriately. This script does not silently
change the mathematical functionals or replace the time integrator.
"""

from __future__ import annotations

import time
import math
import warnings
from dataclasses import dataclass

import numpy as np
import torch
from tqdm import tqdm

import uaibot as ub
from uaibot.simobjects import Box
from uaibot.simobjects.curve import CurveSE3
from vfobstacle.holder_distance_fast import HolderDistanceBatch

from gcbf.curves import build_curve
from gcbf.uav import UAV, _to_torch, TorchVFContext


def _vee_skew_part(A: torch.Tensor) -> torch.Tensor:
    """vee((A - A^T)/2), batched: (...,3,3) -> (...,3)."""
    return 0.5 * torch.stack(
        (
            A[..., 2, 1] - A[..., 1, 2],
            A[..., 0, 2] - A[..., 2, 0],
            A[..., 1, 0] - A[..., 0, 1],
        ),
        dim=-1,
    )


def orthogonal_projection(
    x: torch.Tensor,
    y: torch.Tensor,
    eps: float = 1e-12,
) -> torch.Tensor:
    """
    Orthogonal projection of y onto the subspace
    perpendicular to x.

    x, y: (..., n), with broadcastable batch dimensions.
    Returns: (..., n).

    Assumes x is nonzero. eps prevents division by zero.
    """
    xy = (x * y).sum(dim=-1, keepdim=True)
    xx = (x * x).sum(dim=-1, keepdim=True)

    return y - (xy / xx.clamp_min(eps)) * x


def _diff12(
    F: torch.Tensor, ds: float, periodic: bool
) -> tuple[torch.Tensor, torch.Tensor]:
    """Second-order numerical first and second derivatives along s (axis 0)."""
    if periodic:
        Fm = torch.roll(F, 1, 0)
        Fp = torch.roll(F, -1, 0)
        return (Fp - Fm) / (2 * ds), (Fp - 2 * F + Fm) / (ds * ds)

    Fs = torch.empty_like(F)
    Fss = torch.empty_like(F)
    Fs[1:-1] = (F[2:] - F[:-2]) / (2 * ds)
    Fss[1:-1] = (F[2:] - 2 * F[1:-1] + F[:-2]) / (ds * ds)
    Fs[0] = (-3 * F[0] + 4 * F[1] - F[2]) / (2 * ds)
    Fs[-1] = (3 * F[-1] - 4 * F[-2] + F[-3]) / (2 * ds)
    Fss[0] = (2 * F[0] - 5 * F[1] + 4 * F[2] - F[3]) / (ds * ds)
    Fss[-1] = (2 * F[-1] - 5 * F[-2] + 4 * F[-3] - F[-4]) / (ds * ds)
    return Fs, Fss


def get_L_ell_s(
    C: torch.Tensor,
    gamma: float,
    ds: float | None = None,
    periodic: bool = True,
) -> torch.Tensor:
    """Smoothness functional derivative, (N,4,4) -> (N,6).

    ell_s = gamma * [-v_s + omega x v, -omega_s], where
    xi_s = (v,omega) = S^-1(C_s C^-1).

    If ds=None, s is assumed in [0,1] with N intervals for periodic
    samples (no duplicated endpoint), or N-1 intervals for open samples.
    """
    N = C.shape[0]
    if N < (3 if periodic else 4):
        raise ValueError("Too few curve samples")
    if ds is None:
        ds = 1 / (N if periodic else N - 1)
    if ds <= 0:
        raise ValueError("ds must be positive")

    F = C[:, :3, :4]
    Fs, Fss = _diff12(F, ds, periodic)
    R, p = F[..., :3], F[..., 3]
    Rs, ps = Fs[..., :3], Fs[..., 3]
    Rss, pss = Fss[..., :3], Fss[..., 3]
    Rt = R.transpose(-1, -2)

    omega = _vee_skew_part(Rs @ Rt)
    omega_s = _vee_skew_part(Rss @ Rt)
    v = ps - torch.linalg.cross(omega, p, dim=-1)
    vs = (
        pss
        - torch.linalg.cross(omega_s, p, dim=-1)
        - torch.linalg.cross(omega, ps, dim=-1)
    )

    return gamma * torch.cat(
        (-vs + torch.linalg.cross(omega, v, dim=-1), -omega_s), dim=-1
    )


def L_half_squared_eedistance(
    V: torch.Tensor,
    W: torch.Tensor,
    zeta: float,
    scale_pos: float = 1.0,
    max_cos_theta: float = 0.999,
) -> torch.Tensor:
    """L_V[0.5*zeta*eedistance(V,W)^2]^T, as (...,6).

    Follows the existing supplied formula and alpha branch exactly. There is
    no sqrt(distance_sq) or division by the EE-distance.
    Assumes valid SE(3) rotations, relative angle < pi.
    """
    R = V[..., :3, :3]
    Rt = R.transpose(-1, -2)
    pV, pW = V[..., :3, 3], W[..., :3, 3]
    Q = Rt @ W[..., :3, :3]
    t = (Rt @ (pW - pV).unsqueeze(-1)).squeeze(-1)

    skew_vec = _vee_skew_part(Q)
    cos0 = 0.5 * (Q.diagonal(dim1=-2, dim2=-1).sum(-1) - 1.0)
    sin0 = torch.linalg.vector_norm(skew_vec, dim=-1)
    theta = torch.atan2(sin0, cos0)
    cos_theta = torch.cos(theta)
    mask = cos_theta > max_cos_theta

    d0 = 1.0 - cos_theta
    denominator = 4 * d0.square()
    alpha_general = (2 * d0 - theta.square()) / torch.where(
        mask, torch.ones_like(denominator), denominator
    )
    alpha = torch.where(mask, torch.full_like(theta, -1 / 12), alpha_general)

    Qt = (Q @ t.unsqueeze(-1)).squeeze(-1)
    Qtt = (Q.transpose(-1, -2) @ t.unsqueeze(-1)).squeeze(-1)
    Mt = t + alpha.unsqueeze(-1) * (Qt + Qtt - 2 * t)

    sinc_theta = torch.sinc(theta / torch.pi)
    r = skew_vec / sinc_theta.unsqueeze(-1)
    u = theta.square()
    beta_series = -1 / 45 + u * (
        -13 / 3780 + u * (-23 / 75600 + u * (-101 / 4989600 - u * 5263 / 4670265600))
    )
    small = theta < 0.5
    d = 2 * torch.sin(0.5 * theta).square()
    d_safe = torch.where(small, torch.ones_like(d), d)
    beta_general = ((u - d) * sinc_theta - d) / (2 * d_safe**3)
    beta = torch.where(small, beta_series, beta_general)
    beta = torch.where(mask, torch.zeros_like(beta), beta)

    s2 = scale_pos**2
    h = -(Qt - t).square().sum(-1)
    a = s2 * Mt
    b = (-2 - 0.5 * s2 * h * beta).unsqueeze(-1) * r + (s2 * alpha).unsqueeze(
        -1
    ) * torch.linalg.cross(t, Qt, dim=-1)

    Ra = (R @ a.unsqueeze(-1)).squeeze(-1)
    Rb = (R @ b.unsqueeze(-1)).squeeze(-1)
    pW = pW + torch.zeros_like(Ra)  # supports broadcasting V and W
    return zeta * torch.cat((-Ra, Rb - torch.linalg.cross(pW, Ra, dim=-1)), dim=-1)


def _so3_exp_coefficients(
    w: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Stable coefficients A,B,C of exp(w^) and SE(3) left Jacobian."""
    u = (w * w).sum(-1)  # theta**2
    theta = torch.sqrt(u)
    small = u < 1e-4
    u_safe = torch.where(small, torch.ones_like(u), u)
    theta_safe = torch.where(small, torch.ones_like(theta), theta)
    a = torch.where(
        small, 1 - u / 6 + u * u / 120 - u**3 / 5040, torch.sin(theta) / theta_safe
    )
    b = torch.where(
        small,
        0.5 - u / 24 + u * u / 720 - u**3 / 40320,
        (1 - torch.cos(theta)) / u_safe,
    )
    c = torch.where(
        small,
        1 / 6 - u / 120 + u * u / 5040 - u**3 / 362880,
        (theta - torch.sin(theta)) / (u_safe * theta_safe),
    )
    return a, b, c


def left_step_se3(C: torch.Tensor, xi: torch.Tensor, dt: float) -> torch.Tensor:
    """Closed-form left update exp(dt*S(xi)) @ C, batched without matrix_exp.

    C: (N,4,4), xi: (N,6) (spatial [v,omega]); returns (N,4,4).
    """
    if C.ndim != 3 or C.shape[-2:] != (4, 4) or xi.shape != (C.shape[0], 6):
        raise ValueError(f"Expected (N,4,4) and (N,6), got {C.shape} and {xi.shape}")
    v, w = (dt * xi[..., :3]), (dt * xi[..., 3:])
    a, b, c = _so3_exp_coefficients(w)

    R, p = C[:, :3, :3], C[:, :3, 3]
    # Apply the rotation exponential to R columnwise using cross products.
    cols = R.transpose(-1, -2)
    wxR = torch.linalg.cross(w.unsqueeze(-2), cols, dim=-1)
    wxwxR = torch.linalg.cross(w.unsqueeze(-2), wxR, dim=-1)
    Rnew = R + (a[:, None, None] * wxR + b[:, None, None] * wxwxR).transpose(-1, -2)

    wxp = torch.linalg.cross(w, p, dim=-1)
    wxv = torch.linalg.cross(w, v, dim=-1)
    pnew = (
        p
        + a[:, None] * wxp
        + b[:, None] * torch.linalg.cross(w, wxp, dim=-1)
        + v
        + b[:, None] * wxv
        + c[:, None] * torch.linalg.cross(w, wxv, dim=-1)
    )

    out = torch.empty_like(C)
    out[:, :3, :3] = Rnew
    out[:, :3, 3] = pnew
    out[:, 3, :3] = 0
    out[:, 3, 3] = 1
    return out


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


@dataclass
class DeformationConfig:
    holder_gamma: float = 2.0
    holder_eps: float = 1e-2
    holder_k: int = 2
    func_zeta: float = 1.0
    func_gamma: float = 1.0
    func_ds: float | None = None
    scale_pos: float = 1.0
    max_cos_theta: float = 0.999
    periodic: bool = True
    debug_flag: int = 7  # 1=obstacle, 2=elastic, 4=smoothing; add for combinations
    collision_chunk_size: int | None = 256


class SE3CurveDeformer:
    """Caches only local geometry; reads live obstacles each numerical time step."""

    def __init__(self, curve0: torch.Tensor, uav_model: UAV, cfg: DeformationConfig):
        if curve0.ndim != 3 or curve0.shape[-2:] != (4, 4):
            raise ValueError("curve0 must have shape (N,4,4)")
        self.curve0 = curve0.detach().clone()
        self.collision_model = uav_model.collision_model
        self.cfg = cfg
        self.distance_engine = HolderDistanceBatch(
            self.collision_model,
            curve0.device,
            curve0.dtype,
            chunk_size=cfg.collision_chunk_size or curve0.shape[0],
            compress_repeated_directions=True,
        )
        self._warned_explicit_stability = False

    def step(
        self, C: torch.Tensor, obstacles, t: float, dt: float, profile: bool = False
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """One explicit time step, with batched evaluation over all curve samples.

        Obstacles may have been moved between calls; their current poses are
        used by the collision routine. `t` is for an external motion update.
        """
        if C.device != self.curve0.device or C.dtype != self.curve0.dtype:
            raise ValueError("C and curve0 must have the same device and dtype")
        cfg = self.cfg
        if (
            not self._warned_explicit_stability
            and cfg.debug_flag & 4
            and cfg.func_gamma > 0
        ):
            ds = cfg.func_ds
            if ds is None:
                ds = 1 / (C.shape[0] if cfg.periodic else C.shape[0] - 1)
            limit = ds * ds / (2 * cfg.func_gamma)
            if dt > limit:
                warnings.warn(
                    f"Explicit smoothing is likely unstable: dt={dt:g}, "
                    f"linearized diffusion limit approximately {limit:.3g}. "
                    "Use a stable discretization or adjust dt/gamma/ds.",
                    RuntimeWarning,
                    stacklevel=2,
                )
            self._warned_explicit_stability = True
        timings = {}
        if profile:
            _sync(C.device)
            tic = time.perf_counter()

        # No computation graph is retained across integration steps.
        with torch.no_grad():
            xi = C.new_zeros((C.shape[0], 6))

            if cfg.debug_flag & 1:
                # Batched analytic spatial Lie-group gradient: no autograd.
                # This reads the obstacle poses at each step (dynamic-safe).
                d_pairs, grad, beta = self.distance_engine(
                    C,
                    obstacles,
                    gamma=cfg.holder_gamma,
                    eps=cfg.holder_eps,
                    k=cfg.holder_k,
                )
                xi.add_(grad)
                if profile:
                    _sync(C.device)
                    timings["collision"] = time.perf_counter() - tic
                    tic = time.perf_counter()

            if cfg.debug_flag & 2:
                xi.sub_(
                    L_half_squared_eedistance(
                        C,
                        self.curve0,
                        cfg.func_zeta,
                        cfg.scale_pos,
                        cfg.max_cos_theta,
                    )
                )
                if profile:
                    _sync(C.device)
                    timings["elastic"] = time.perf_counter() - tic
                    tic = time.perf_counter()

            if cfg.debug_flag & 4:
                xi.sub_(
                    get_L_ell_s(
                        C,
                        cfg.func_gamma,
                        cfg.func_ds,
                        cfg.periodic,
                    )
                )
                if profile:
                    _sync(C.device)
                    timings["smoothing"] = time.perf_counter() - tic
                    tic = time.perf_counter()

            # Remove the spatial tangent from temporal deformation.
            ds = cfg.func_ds
            if ds is None:
                ds = 1.0 / (C.shape[0] if cfg.periodic else C.shape[0] - 1)

            # Numerical first derivative of C with respect to s.
            if cfg.periodic:
                Cs = (torch.roll(C, -1, dims=0) - torch.roll(C, 1, dims=0)) / (2.0 * ds)
            else:
                Cs = torch.empty_like(C)
                Cs[1:-1] = (C[2:] - C[:-2]) / (2.0 * ds)
                Cs[0] = (-3 * C[0] + 4 * C[1] - C[2]) / (2.0 * ds)
                Cs[-1] = (3 * C[-1] - 4 * C[-2] + C[-3]) / (2.0 * ds)

            R = C[:, :3, :3]
            p = C[:, :3, 3]

            # xi_s = S^{-1}(Cs @ C^{-1}), spatial twist.
            Rs = Cs[:, :3, :3]
            ps = Cs[:, :3, 3]

            A = Rs @ R.transpose(-1, -2)
            omega_s = _vee_skew_part(A)
            v_s = ps - torch.linalg.cross(omega_s, p, dim=-1)

            xi_s = torch.cat([v_s, omega_s], dim=-1)

            # Project temporal deformation orthogonally to xi_s.
            xi = orthogonal_projection(xi_s, xi)

            next_curve = left_step_se3(C, xi, dt)

        if profile:
            _sync(C.device)
            timings["SE3_step"] = time.perf_counter() - tic
            timings["total_math"] = sum(timings.values())
        return next_curve.detach(), xi.detach(), timings


def tracking_twist(
    ctx,
    H,
    xi,
    kt1=1.0,
    kt2=1.0,
    kt3=1.0,
    kn1=1.0,
    kn2=1.0,
):
    """
    H:          (4, 4) or (N, 4, 4)
    xi_actual:  (6,)   or (N, 6)
    Returns: (xi_dot_des, xi_vf, dist, idx) — squeezed if unbatched.
    """
    H = _to_torch(H, ctx.device, ctx.dtype)
    squeeze = H.dim() == 2
    if squeeze:
        H = H.unsqueeze(0)

    # ---- three poses in one batch: [H, H_fwd, H_bwd] ----

    twist, dist, idx = ctx.eval_vf(H, kt1, kt2, kt3, kn1, kn2)

    xi_vf = xi[idx] + twist  # (N, 6)

    if squeeze:
        return (
            xi_vf[0].squeeze(0),
            dist[0].squeeze(0),
            idx[0].squeeze(0),
        )
    return xi_vf, dist, idx


def main() -> None:
    curve_n = 1000
    T, dt = 10.0, 1e-2
    render_every = 1  # reduce costly animation frames / CPU copies
    render_decimate = 1  # render 1/10 of the simulated curve samples
    profile_every = 1000
    obs_movement_period = 5.0

    kn1 = 1.0
    kn2 = 1.0
    kt1 = 0.3
    kt2 = 1.0
    kt3 = kn2
    # On consumer GPUs, float64 can be much slower than float32, but float64
    # is substantially more robust for small ds and second differences.
    dtype = torch.float64
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    curve, *_ = build_curve("circle", radius=1.5, center=(0, 0, 0.5), n_points=curve_n)
    curve = np.asarray(curve)
    if np.allclose(curve[0], curve[-1], rtol=1e-10, atol=1e-10):
        # Periodic central differences require no duplicate endpoint.
        curve = curve[:-1]
    curve_n = len(curve)
    curve0 = torch.as_tensor(curve, device=device, dtype=dtype).clone()
    C = curve0.clone()

    # drone_htm0 = np.eye(4)
    drone_htm0 = np.asarray(ub.utils.trn([2.0, 0.1, 0.3]))
    drone = UAV(
        htm=drone_htm0,
        inertia=np.ones(3),
        mass=20,
        device=device,
        dtype=dtype,
    )

    h = w = d = 0.4

    # ----------------------- Moving obstacle --------------------------
    radius = 2.0
    linear_speed = 2 * np.pi * radius / obs_movement_period  # m/s
    angular_speed = np.pi / (obs_movement_period * 4)  # rad/s

    center = torch.tensor([0.0, 0.0, 0.5], device=device, dtype=dtype)

    axis_body = torch.tensor([1.0, 1.0, 1.0], device=device, dtype=dtype)
    axis_body = axis_body / torch.linalg.norm(axis_body)

    htm0_obs1 = ub.Utils.trn([0.0, 2.0, 0.5])
    htm0_obs1_t = torch.as_tensor(htm0_obs1, device=device, dtype=dtype).clone()

    def moving_obstacle_pose(t):
        theta = linear_speed / radius * t

        # Integrate body-frame rotation from the initial pose.
        xi = torch.zeros((1, 6), device=device, dtype=dtype)
        xi[0, 3:] = angular_speed * (htm0_obs1_t[:3, :3] @ axis_body)

        H = left_step_se3(
            htm0_obs1_t.unsqueeze(0),
            xi,
            t,
        )[0]

        # Prescribe the circular position independently.
        H[:3, 3] = center + radius * torch.stack(
            (
                torch.sin(torch.as_tensor(theta, device=device, dtype=dtype)),
                torch.cos(torch.as_tensor(theta, device=device, dtype=dtype)),
                torch.zeros((), device=device, dtype=dtype),
            )
        )

        return H

    obs1 = Box(
        htm=htm0_obs1,
        width=w,
        height=h,
        depth=d,
        color="blue",
        opacity=0.6,
    )

    htm_obs2 = ub.Utils.trn([0, 1.2 * 0, 0.4])
    obs2 = Box(
        htm=htm_obs2,
        width=w,
        height=h,
        depth=d,
        color="red",
        opacity=0.6,
    )
    obstacles = [obs1, obs2]

    # For moving obstacles, update obs1.htm/obs2.htm or their supported
    # pose-update API HERE, before each call to engine.step(...).
    # Example-only stable gain for the *linearized explicit smoothing* mode.
    # This is much smaller than the user's func_gamma=1.0 and makes smoothing
    # correspondingly weaker. For substantial smoothing at dt=0.01, a more
    # suitable implicit Lie-group integration method is needed.
    ds_demo = 1.0 / curve_n
    stable_gamma_demo = 0.2 * ds_demo**2 / dt
    print(f"stable gama: {stable_gamma_demo}")
    cfg = DeformationConfig(
        holder_gamma=2.0,
        holder_eps=1e-3,
        holder_k=2,
        func_zeta=3.0,
        # func_gamma=stable_gamma_demo,
        func_gamma=2 * stable_gamma_demo,
        func_ds=None,
        # 1 - obstacle, 2 - elastic, 4 - smoothing
        debug_flag=7,  # All three functionals.
        collision_chunk_size=256,
    )
    engine = SE3CurveDeformer(curve0, drone, cfg)

    render_curve0 = np.asarray(curve)[::render_decimate]
    curve_ub = CurveSE3(points=render_curve0, color="cyan", size=0.05)
    sim = ub.Simulation([curve_ub, obs1, obs2], width=1080, height=720)
    sim.add(drone.collision_model)

    ctx = TorchVFContext(curve0, None)  # caches curve & VF

    # The engine warns when dt violates the approximate linearized
    # explicit-smoothing stability bound. The demo uses a small gamma
    # to avoid immediate numerical blow-up. This is not a substitute
    # for a semi-implicit method when strong smoothing is needed.

    for j, t_np in enumerate(tqdm(np.arange(0, T, dt))):
        t = float(t_np)
        # update_dynamic_obstacle_poses(obstacles, t)  # user-supplied function
        profile = profile_every > 0 and j % profile_every == 0
        # Move only the first obstacle.
        H_obs1 = moving_obstacle_pose(t)
        C, xi, times = engine.step(C, obstacles, t=t, dt=dt, profile=profile)
        ctx.set_curve(C)
        u, *other = tracking_twist(
            ctx,
            drone.htm,
            xi,
            kt1=kt1,
            kt2=kt2,
            kt3=kt3,
            kn1=kn1,
            kn2=kn2,
        )
        drone.step(t, dt, u, type="kinematic")

        if profile:
            print("stage times (s):", {k: round(v, 5) for k, v in times.items()})

        if j % render_every == 0:
            # One synchronization / device transfer per rendered frame.
            viz_tic = time.perf_counter() if profile else None
            points = C[::render_decimate].cpu().numpy()
            curve_ub.add_ani_frame(time=t, points=points)
            obs1.add_ani_frame(time=t, htm=H_obs1.detach().cpu().numpy())
            if profile:
                print(
                    f"visualization transfer + record: {time.perf_counter() - viz_tic:.5f} s"
                )

    sim.run_in_browser()


if __name__ == "__main__":
    main()
