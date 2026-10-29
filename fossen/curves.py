import numpy as np


# ==========================================================================
# Custom frame: Inward-pointing (for the vertical coil)
# ==========================================================================
def _inward_frame(p, dp_ds, dp_ds2):
    """
    x-body = points inward to cylinder center.
    z-body = points up.
    y-body = completes right-handed frame (which points BACKWARDS along the path).
    """
    # 1. Get the tangent direction in the xy-plane
    t_xy = dp_ds[:2]
    t_xy = t_xy / np.linalg.norm(t_xy)

    # 2. Inward normal (rotate tangent by 90 degrees CCW)
    x_axis = np.array([-t_xy[1], t_xy[0], 0.0])
    z_axis = np.array([0.0, 0.0, 1.0])

    # 3. Complete the frame: y = z cross x
    y_axis = np.cross(z_axis, x_axis)

    R = np.column_stack([x_axis, y_axis, z_axis])

    # 4. Calculate dR/ds using numerical differentiation for robustness
    eps = 1e-6
    t_next = (dp_ds + dp_ds2 * eps)[:2]
    t_next = t_next / np.linalg.norm(t_next)

    x_next = np.array([-t_next[1], t_next[0], 0.0])
    y_next = np.cross(z_axis, x_next)
    R_next = np.column_stack([x_next, y_next, z_axis])

    dR = (R_next - R) / eps
    return R, dR


# ==========================================================================
# Generic frame construction (default: tangent-aligned)
# ==========================================================================
def _frame_and_derivative(p, dp_ds, dp_ds2):
    # ... (keep the existing default frame code exactly as it is) ...
    v = float(np.linalg.norm(dp_ds))
    t = dp_ds / v
    k = np.array([0.0, 0.0, 1.0])

    t_dot_ddp = float(np.dot(t, dp_ds2))
    dt = (dp_ds2 - t * t_dot_ddp) / v

    w = np.cross(k, t)
    w_norm = float(np.linalg.norm(w))
    y = w / w_norm

    dw = np.cross(k, dt)
    dw_norm = float(np.dot(w, dw)) / w_norm
    dy = dw / w_norm - w * dw_norm / (w_norm**2)

    z = np.cross(t, y)
    dz = np.cross(dt, y) + np.cross(t, dy)

    R = np.column_stack([t, y, z])
    dR = np.column_stack([dt, dy, dz])
    return R, dR


def _htm_and_derivative(p, dp_ds, dp_ds2, frame_func=_frame_and_derivative):
    R, dR = frame_func(p, dp_ds, dp_ds2)
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = p
    dT = np.zeros((4, 4))
    dT[:3, :3] = dR
    dT[:3, 3] = dp_ds
    return T, dT


def _sample(p_func, dp_func, ddp_func, n_points, frame_func=_frame_and_derivative):
    curve = np.zeros((n_points, 4, 4))
    dcurve = np.zeros((n_points, 4, 4))
    for i in range(n_points):
        s = i / n_points
        p = p_func(s)
        dp = dp_func(s)
        ddp = ddp_func(s)
        T, dT = _htm_and_derivative(p, dp, ddp, frame_func)
        curve[i] = T
        dcurve[i] = dT
    return curve, dcurve


def barrel_roll(p, dp_ds, dp_ds2, n_rolls, s):
    # Base frame — call the existing default unchanged
    R0, dR0 = _frame_and_derivative(p, dp_ds, dp_ds2)

    t, y0, z0 = R0[:, 0], R0[:, 1], R0[:, 2]
    dt, dy0, dz0 = dR0[:, 0], dR0[:, 1], dR0[:, 2]

    theta = 2.0 * np.pi * n_rolls * s
    dtheta = 2.0 * np.pi * n_rolls

    c, sn = np.cos(theta), np.sin(theta)

    y = c * y0 + sn * z0
    z = -sn * y0 + c * z0

    dy = -sn * dtheta * y0 + c * dy0 + c * dtheta * z0 + sn * dz0
    dz = -c * dtheta * y0 - sn * dy0 - sn * dtheta * z0 + c * dz0

    R = np.column_stack([t, y, z])
    dR = np.column_stack([dt, dy, dz])
    return R, dR


def _sample_barrel_roll(p_func, dp_func, ddp_func, n_points, n_rolls):
    curve = np.zeros((n_points, 4, 4))
    dcurve = np.zeros((n_points, 4, 4))
    for i in range(n_points):
        s = i / n_points

        # Bind s via a default arg so _htm_and_derivative can call
        # frame_func(p, dp, ddp) with its unchanged 3-argument signature.
        def frame_func(p, dp_ds, dp_ds2, _s=s):
            return barrel_roll(p, dp_ds, dp_ds2, n_rolls, _s)

        T, dT = _htm_and_derivative(p_func(s), dp_func(s), ddp_func(s), frame_func)
        curve[i] = T
        dcurve[i] = dT
    return curve, dcurve


# ==========================================================================
# Curve builders
# ==========================================================================


def make_circle(radius=1.5, z0=-1.0, center_xy=(0.0, 0.0), n_points=5000, n_rolls=0):
    cx, cy = center_xy
    w = 2.0 * np.pi
    w2 = w * w

    def p(s):
        th = w * s
        return np.array([cx + radius * np.cos(th), cy + radius * np.sin(th), z0])

    def dp(s):
        th = w * s
        return np.array([-w * radius * np.sin(th), w * radius * np.cos(th), 0.0])

    def ddp(s):
        th = w * s
        return np.array([-w2 * radius * np.cos(th), -w2 * radius * np.sin(th), 0.0])

    if n_rolls > 0:
        return _sample_barrel_roll(p, dp, ddp, n_points, n_rolls)
    else:
        return _sample(p, dp, ddp, n_points)


def make_ellipse(a=2.0, b=1.0, z0=-1.0, center_xy=(0.0, 0.0), n_points=5000):
    cx, cy = center_xy
    w = 2.0 * np.pi
    w2 = w * w

    def p(s):
        th = w * s
        return np.array([cx + a * np.cos(th), cy + b * np.sin(th), z0])

    def dp(s):
        th = w * s
        return np.array([-w * a * np.sin(th), w * b * np.cos(th), 0.0])

    def ddp(s):
        th = w * s
        return np.array([-w2 * a * np.cos(th), -w2 * b * np.sin(th), 0.0])

    return _sample(p, dp, ddp, n_points)


def make_lissajous_3d(
    A=2.0, B=1.5, C=0.5, z0=-1.0, center_xy=(0.0, 0.0), n_points=5000
):
    """
    Closed 3D curve: 2:1 frequency ratio between (x,y) and z.
    No self-intersections because the vertical oscillation breaks the
    2D projection's degeneracy.
    """
    cx, cy = center_xy
    w = 2.0 * np.pi
    w2 = 2.0 * w  # z has twice the angular frequency
    w2_sq = w2 * w2

    def p(s):
        th = w * s
        ph = w2 * s
        return np.array([cx + A * np.cos(th), cy + B * np.sin(th), z0 + C * np.sin(ph)])

    def dp(s):
        th = w * s
        ph = w2 * s
        return np.array([-w * A * np.sin(th), w * B * np.cos(th), w2 * C * np.cos(ph)])

    def ddp(s):
        th = w * s
        ph = w2 * s
        return np.array(
            [-w * w * A * np.cos(th), -w * w * B * np.sin(th), -w2_sq * C * np.sin(ph)]
        )

    return _sample(p, dp, ddp, n_points)


# ==========================================================================
# Curve builders
# ==========================================================================
def make_vertical_coil(
    R=2.0, z_amp=0.8, N_oscillations=3, z0=-2.0, center_xy=(0.0, 0.0), n_points=5000
):
    cx, cy = center_xy
    w = 2.0 * np.pi
    w2 = w * w

    def p(s):
        th = w * s
        return np.array(
            [
                cx + R * np.cos(th),
                cy + R * np.sin(th),
                z0 + z_amp * np.sin(N_oscillations * th),
            ]
        )

    def dp(s):
        th = w * s
        return np.array(
            [
                -w * R * np.sin(th),
                w * R * np.cos(th),
                w * z_amp * N_oscillations * np.cos(N_oscillations * th),
            ]
        )

    def ddp(s):
        th = w * s
        return np.array(
            [
                -w2 * R * np.cos(th),
                -w2 * R * np.sin(th),
                -w2 * z_amp * (N_oscillations**2) * np.sin(N_oscillations * th),
            ]
        )

    # Use the custom inward frame!
    return _sample(p, dp, ddp, n_points, frame_func=_inward_frame)


def make_trefoil_knot(R=3.0, z_amp=0.4, z0=-2.0, center_xy=(0.0, 0.0), n_points=5000):
    cx, cy = center_xy
    w = 2.0 * np.pi
    w2 = w * w

    def p(s):
        th = w * s
        return np.array(
            [
                cx + R * (np.sin(th) + 2.0 * np.sin(2.0 * th)),
                cy + R * (np.cos(th) - 2.0 * np.cos(2.0 * th)),
                z0 + z_amp * np.sin(3.0 * th),
            ]
        )

    def dp(s):
        th = w * s
        return np.array(
            [
                w * R * (np.cos(th) + 4.0 * np.cos(2.0 * th)),
                w * R * (-np.sin(th) + 4.0 * np.sin(2.0 * th)),
                w * z_amp * 3.0 * np.cos(3.0 * th),
            ]
        )

    def ddp(s):
        th = w * s
        return np.array(
            [
                w2 * R * (-np.sin(th) - 8.0 * np.sin(2.0 * th)),
                w2 * R * (-np.cos(th) + 8.0 * np.cos(2.0 * th)),
                -w2 * z_amp * 9.0 * np.sin(3.0 * th),
            ]
        )

    # Uses default tangent frame
    return _sample(p, dp, ddp, n_points)


# ==========================================================================
# Registry
# ==========================================================================
CURVE_BUILDERS = {
    "circle": make_circle,
    "ellipse": make_ellipse,
    "lissajous_3d": make_lissajous_3d,
    "trefoil": make_trefoil_knot,
    "vertical_coil": make_vertical_coil,
}


def build_curve(curve_type, **kwargs):
    if curve_type not in CURVE_BUILDERS:
        raise ValueError(
            f"Unknown curve type: {curve_type}. " f"Available: {list(CURVE_BUILDERS)}"
        )
    return CURVE_BUILDERS[curve_type](**kwargs)
