"""Batched Hölder signed distance and exact spatial SE(3) L-gradients.

Optimized for a single rigid collision body at many poses and several (possibly
moving) convex-polyhedral obstacles. The mathematical reduction and direction
ordering match the supplied uaibot/gpu/{distance,functional,geometry}.py.

No autograd, six-generator arrays, per-(pose,obstacle) Python loops, or world
geometry extraction from the same object thousands of times.

Usage:
    engine = HolderDistanceBatch(uav.collision_model, curve.device, curve.dtype)
    d_pairs, L_beta, beta = engine(curve, obstacles, gamma=2, eps=1e-2, k=2)
    # d_pairs (N,M), L_beta (N,6), beta (N,)

Call every time step. Obstacles' current .htm values are fetched at EVERY call.
Local-frame geometry (not obstacle pose) is cached. If the geometry itself
changes, call engine.clear_local_cache() after making that change.
"""

from __future__ import annotations

from collections import defaultdict

import torch


def _shape_and_derivative(x: torch.Tensor, k: float, eps: float):
    """Phi(x) = x |x|^k/(|x|^k+eps), and d Phi/dx."""
    h = x.abs().pow(k)
    den = h + eps
    ratio = h / den
    return x * ratio, ratio + k * (eps / den) * ratio


def _hmin_and_weights(
    x: torch.Tensor, p: float, dim: int = -1, multiplicity: torch.Tensor | None = None
):
    """HMin_{gamma}, p=gamma+1, with analytical partial derivatives.

    Exactly matches functional.holder_min except that powers are evaluated
    relative to a scale, avoiding overflow for extremely small positive x.
    On the zero branch, selects zero derivatives (same convention as the
    supplied torch.where implementation). This is harmless when immediately
    followed by a shaping function, whose derivative at zero vanishes.
    """
    if p <= 1:
        raise ValueError("p=gamma+1 must exceed 1")
    m = x.amin(dim=dim, keepdim=True)
    positive = m > 0
    negative = m < 0

    # Positive branch: f = m / (sum_i (m/x_i)^p)^(1/p).
    # These ratios are evaluated only for the positive branch.
    pos_x = torch.where(x > 0, x, torch.ones_like(x))
    pos_m = torch.where(positive, m, torch.ones_like(m))
    q = pos_m / pos_x
    q_p = q.pow(p)
    if multiplicity is None:
        multiplicity = 1.0
    s_pos = (multiplicity * q_p).sum(dim=dim, keepdim=True)
    s_pos_safe = torch.where(positive, s_pos, torch.ones_like(s_pos))
    factor = s_pos_safe.pow(-1.0 / p)
    out_pos = pos_m * factor
    weight_pos = multiplicity * q.pow(p + 1.0) * (factor / s_pos_safe)

    # Negative branch: f = -||(-x)_+||_p, with scale-safe norm/weights.
    neg_x = x.neg().clamp_min(0)
    max_neg = neg_x.amax(dim=dim, keepdim=True)
    max_neg_safe = torch.where(max_neg > 0, max_neg, torch.ones_like(max_neg))
    neg_q = neg_x / max_neg_safe
    neg_p = neg_q.pow(p)
    s_neg = (multiplicity * neg_p).sum(dim=dim, keepdim=True)
    s_neg_safe = torch.where(negative, s_neg, torch.ones_like(s_neg))
    norm_factor = s_neg_safe.pow(1.0 / p)
    out_neg = -max_neg * norm_factor
    weight_neg = multiplicity * neg_q.pow(p - 1.0) * s_neg_safe.pow((1.0 - p) / p)

    out = torch.where(
        positive, out_pos, torch.where(negative, out_neg, torch.zeros_like(out_pos))
    )
    weight = torch.where(
        positive,
        weight_pos,
        torch.where(negative, weight_neg, torch.zeros_like(weight_pos)),
    )
    return out.squeeze(dim), weight


def _hmax_and_weights(
    x: torch.Tensor, p: float, dim: int = -1, multiplicity: torch.Tensor | None = None
):
    y, w = _hmin_and_weights(-x, p, dim=dim, multiplicity=multiplicity)
    return -y, w


def _check_vef(v, e, f, name: str):
    for x in (v, e, f):
        if x.ndim != 3 or x.shape[-1] != 3:
            raise ValueError(f"{name} components require shape (B,components,3)")


def holder_distance_directions_with_L(
    vertex_a: torch.Tensor,
    directions_a: torch.Tensor,
    counts_a: torch.Tensor,
    vertex_b: torch.Tensor,
    directions_b: torch.Tensor,
    counts_b: torch.Tensor,
    gamma: float = 2.0,
    eps: float = 1e-3,
):
    """HD-SDF and spatial L_A, with exact multiplicity-weighted directions.

    (N,Va,3), (N,Da,3), (Da,), (M,Vb,3), (M,Db,3), (Db,)
    -> distance (N,M), L_A (N,M,6).

    Direction multiplicity is *essential*: Hölder max is not idempotent.
    Each local direction is retained with a count of its repetitions, so
    collapsing exact duplicates here preserves the original distance and L.
    """
    if eps <= 0:
        raise ValueError("eps must be positive")
    if gamma < 1:
        raise ValueError("gamma must be >= 1")
    N, Va = vertex_a.shape[:2]
    M, Vb = vertex_b.shape[:2]
    Da, Db = directions_a.shape[1], directions_b.shape[1]
    p = gamma + 1.0

    directions = torch.cat(
        (
            directions_a[:, None].expand(N, M, Da, 3),
            directions_b[None].expand(N, M, Db, 3),
        ),
        dim=2,
    )
    counts = (
        torch.cat((counts_a, counts_b))
        .to(device=directions.device, dtype=directions.dtype)
        .view(1, 1, -1)
    )
    D = Da + Db

    # Compute projected differences without generating vertex differences.
    projection_a = torch.einsum("nmdc,nac->nmda", directions, vertex_a)
    projection_b = torch.einsum("nmdc,mbc->nmdb", directions, vertex_b)
    pnv = (projection_a[..., :, None] - projection_b[..., None, :]).reshape(
        N, M, D, Va * Vb
    )

    inner, d_inner = _hmin_and_weights(pnv, p, dim=-1)
    shaped_inner, sh_inner = _shape_and_derivative(inner, gamma, eps)
    outer, d_outer = _hmax_and_weights(shaped_inner, p, dim=-1, multiplicity=counts)
    distances, sh_outer = _shape_and_derivative(outer, gamma, eps)

    dir_scale = sh_outer[..., None] * d_outer * sh_inner
    weights = (d_inner * dir_scale[..., None]).reshape(N, M, D, Va, Vb)

    # For a spatial left perturbation of A, d a=v+omega x a.
    # For normals belonging to A: L_A[n.(a-b)] = [n, b x n].
    # For normals belonging to B: L_A[n.(a-b)] = [n, a x n].
    dir_weights = weights.sum(dim=(-1, -2))
    lin = torch.einsum("nmd,nmdc->nmc", dir_weights, directions)

    w_b = weights[:, :, :Da].sum(dim=-2)  # (N,M,Da,Vb)
    w_a = weights[:, :, Da:].sum(dim=-1)  # (N,M,Db,Va)
    weighted_b = torch.einsum("nmdb,mbc->nmdc", w_b, vertex_b)
    weighted_a = torch.einsum("nmda,nac->nmdc", w_a, vertex_a)
    ang_a = torch.linalg.cross(weighted_b, directions[:, :, :Da], dim=-1).sum(dim=2)
    ang_b = torch.linalg.cross(weighted_a, directions[:, :, Da:], dim=-1).sum(dim=2)
    return distances, torch.cat((lin, ang_a + ang_b), dim=-1)


def holder_distance_vef_with_L(
    vertex_a: torch.Tensor,
    edges_a: torch.Tensor,
    facenormals_a: torch.Tensor,
    vertex_b: torch.Tensor,
    edges_b: torch.Tensor,
    facenormals_b: torch.Tensor,
    gamma: float = 2.0,
    eps: float = 1e-3,
):
    """Original VEF interface, preserving direction repetitions and ordering.

    Returns: d (N,M), L_A (N,M,6), both derived analytically.
    """
    _check_vef(vertex_a, edges_a, facenormals_a, "A")
    _check_vef(vertex_b, edges_b, facenormals_b, "B")
    N = vertex_a.shape[0]
    M = vertex_b.shape[0]
    Ea, Eb = edges_a.shape[1], edges_b.shape[1]
    Fa, Fb = facenormals_a.shape[1], facenormals_b.shape[1]
    # Each of these directions is associated with one moving body.
    dA = torch.cat((edges_a, -edges_a, facenormals_a), dim=1)
    dB = torch.cat((edges_b, -edges_b, facenormals_b), dim=1)
    ca = dA.new_ones((2 * Ea + Fa,))
    cb = dB.new_ones((2 * Eb + Fb,))
    return holder_distance_directions_with_L(
        vertex_a, dA, ca, vertex_b, dB, cb, gamma=gamma, eps=eps
    )


def compress_local_vef(local):
    """Collapse *exact* duplicated local directions without changing HD-SDF.

    Direction origins (A vs B) remain separate, because their L derivatives
    involve different world vertices. Do not remove duplicates without counts.
    """
    v, e, f = local
    all_dirs = torch.cat((e, -e, f), dim=0)
    unique_dirs, counts = torch.unique(all_dirs, dim=0, return_counts=True)
    return v, unique_dirs, counts.to(device=v.device, dtype=v.dtype)


class HolderDistanceBatch:
    """Reusable geometry engine for one robot shape at many SE(3) poses.

    Local VEF is cached per object; obstacle *world poses* are not. Reusing
    this instance across numerical time steps is recommended. If an obstacle's
    shape changes (not merely its .htm), call clear_local_cache().

    The full set of current obstacle distances is always reduced together.
    This also fixes the old code's heterogeneous-group issue where reduce_fn
    was applied separately to each geometry group.
    """

    def __init__(
        self,
        collision_model,
        device: torch.device | str,
        dtype: torch.dtype = torch.float64,
        chunk_size: int = 128,
        compress_repeated_directions: bool = True,
    ):
        if chunk_size < 1:
            raise ValueError("chunk_size must be positive")
        self.collision_model = collision_model
        self.device = torch.device(device)
        self.dtype = dtype
        self.chunk_size = chunk_size
        self.compress_repeated_directions = compress_repeated_directions
        self._local_cache = {}
        self._polytope_cache = {}
        self._signature_cache = {}

    def clear_local_cache(self):
        self._local_cache.clear()
        self._polytope_cache.clear()
        self._signature_cache.clear()

    def _get_local(self, obj):
        key = id(obj)
        cached = self._local_cache.get(key)
        if cached is not None and cached[0] is obj:
            return cached[1]

        # Deferred import means the mathematical VEF kernel can be tested
        # independently of UAIBot. extract_VEF is only run once per shape.
        from uaibot.gpu.geometry import extract_VEF

        identity = torch.eye(4, dtype=self.dtype, device=self.device)
        local = extract_VEF(
            obj,
            identity,
            device=self.device,
            dtype=self.dtype,
            local_cache=self._polytope_cache,
        )
        local = tuple(t.detach().contiguous() for t in local)
        if self.compress_repeated_directions:
            local = compress_local_vef(local)
        else:
            v, e, f = local
            directions = torch.cat((e, -e, f), dim=0)
            local = (v, directions, v.new_ones((directions.shape[0],)))
        self._local_cache[key] = (obj, local)
        # One-time host copy of a handful of multiplicities, not per step.
        self._signature_cache[key] = (
            local[0].shape[0],
            local[1].shape[0],
            tuple(local[2].cpu().tolist()),
        )
        return local

    @staticmethod
    def _world(local, poses):
        verts, directions, counts = local
        R_t = poses[..., :3, :3].transpose(-1, -2)
        t = poses[..., :3, 3]
        if verts.ndim == 2:
            verts, directions = verts[None], directions[None]
        return verts @ R_t + t[..., None, :], directions @ R_t

    @torch.no_grad()
    def __call__(
        self,
        curve_htms: torch.Tensor,
        obstacles,
        gamma: float = 2.0,
        eps: float = 1e-3,
        k: int = 2,
    ):
        """Return (d_pairs, L_beta, beta) with shapes (N,M),(N,6),(N,).

        beta_i = shaping_function(holder_min(d_pairs[i,:], gamma), k),
        L_beta is the *spatial* L-operator derivative of beta_i w.r.t.
        curve_htms[i]. Each obstacle pose is read anew on every call.
        """
        if curve_htms.ndim != 3 or curve_htms.shape[-2:] != (4, 4):
            raise ValueError("curve_htms must be (N,4,4)")
        if curve_htms.device != self.device or curve_htms.dtype != self.dtype:
            raise ValueError(
                f"Expected {self.device}/{self.dtype}; got "
                f"{curve_htms.device}/{curve_htms.dtype}"
            )
        obstacles = list(obstacles)
        if not obstacles:
            raise ValueError("At least one obstacle is required")
        M = len(obstacles)
        N = curve_htms.shape[0]

        # Group obstacles by VEF cardinalities. Geometry and normals are
        # transformed using LIVE poses from the supplied objects.
        groups = defaultdict(list)
        for j, obj in enumerate(obstacles):
            local = self._get_local(obj)
            sig = self._signature_cache.get(id(obj))
            if sig is None:  # Only for custom subclasses overriding _get_local.
                sig = (
                    local[0].shape[0],
                    local[1].shape[0],
                    tuple(local[2].cpu().tolist()),
                )
            groups[sig].append((j, obj, local))
        local_a = self._get_local(self.collision_model)

        # Prepare obstacle world-frame geometry ONCE per iteration.
        # Obstacles may move between calls; no pose-dependent cache survives.
        prepared_b = []
        for members in groups.values():
            index = [item[0] for item in members]
            lvs = torch.stack([item[2][0] for item in members], dim=0)
            lds = torch.stack([item[2][1] for item in members], dim=0)
            counts = members[0][2][2]
            Hb = torch.stack(
                [
                    torch.as_tensor(item[1].htm, device=self.device, dtype=self.dtype)
                    for item in members
                ],
                dim=0,
            )
            vb, db = self._world((lvs, lds, counts), Hb)
            prepared_b.append((index, vb, db, counts))

        # Assemble all pairs (not just within groups) to reduce consistently.
        dist = curve_htms.new_empty((N, M))
        pair_L = curve_htms.new_empty((N, M, 6))
        for first in range(0, N, self.chunk_size):
            end = min(first + self.chunk_size, N)
            Ha = curve_htms[first:end]
            va, da = self._world(local_a, Ha)
            for index, vb, db, counts in prepared_b:
                dij, gij = holder_distance_directions_with_L(
                    va, da, local_a[2], vb, db, counts, gamma, eps
                )
                dist[first:end, index] = dij
                pair_L[first:end, index, :] = gij

        # Compute gradient of the *global* Hölder min over ALL obstacles.
        with torch.no_grad():
            reduced, w_min = _hmin_and_weights(dist, gamma + 1.0, dim=1)
            beta, phi_derivative = _shape_and_derivative(reduced, k, eps)
            L_beta = torch.einsum("nm,nmk->nk", w_min * phi_derivative[:, None], pair_L)
        return dist, L_beta, beta
