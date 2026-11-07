import numpy as np
from uaibot import Utils


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


def generate_barrel_roll_circle(n_points, r, center_xy, z0, n_rolls=1.0):
    """
    Generates a circle in position with a roll along the x-axis (tangent).

    Parameters
    ----------
    n_points : int
        Number of samples.
    r : float
        Circle radius.
    x0, y0, z0 : float
        Center of the circle.
    n_rolls : float
        Number of 360-degree rolls per lap.

    Returns
    -------
    curve : (N, 4, 4) ndarray
        Homogeneous transformation matrices.
    dcurve : (N, 4, 4) ndarray
        Derivatives of the HTMs with respect to the normalized parameter s in [0, 1).
    """
    x0, y0 = center_xy
    curve = np.zeros((n_points, 4, 4))
    dcurve = np.zeros((n_points, 4, 4))

    # Pre-compute the skew-symmetric matrices for the world x and z axes
    S_ex = np.asarray(Utils.S(np.array([1.0, 0.0, 0.0])))
    S_ez = np.asarray(Utils.S(np.array([0.0, 0.0, 1.0])))

    for i in range(n_points):
        s = i / n_points
        theta = 2.0 * np.pi * s
        phi = 2.0 * np.pi * n_rolls * s

        # 1. Position and Position Derivative
        p = np.array([x0 + r * np.cos(theta), y0 + r * np.sin(theta), z0])
        dp_ds = np.array(
            [-2.0 * np.pi * r * np.sin(theta), 2.0 * np.pi * r * np.cos(theta), 0.0]
        )

        # 2. Orientation
        # Base frame: x points along the tangent, z points up
        # Rotating by (theta + pi/2) around z aligns the local x-axis with the tangent
        R_base = np.asarray(Utils.rotz(theta + np.pi / 2))[:3, :3]

        # Roll frame: rotation along the local x-axis (tangent)
        R_roll = np.asarray(Utils.rotx(phi))[:3, :3]

        # Total rotation
        rot = R_base @ R_roll

        # 3. Orientation Derivative
        # Using the chain rule: dR/ds = (dR_base/ds) @ R_roll + R_base @ (dR_roll/ds)
        # Since R_base is rotz(theta + pi/2), its derivative is (dtheta/ds) * S_ez @ R_base
        dR_base_ds = 2.0 * np.pi * (S_ez @ R_base)

        # Since R_roll is rotx(phi), its derivative is (dphi/ds) * S_ex @ R_roll
        dR_roll_ds = 2.0 * np.pi * n_rolls * (S_ex @ R_roll)

        dR_ds = dR_base_ds @ R_roll + R_base @ dR_roll_ds

        # 4. Assemble Homogeneous Matrices
        T = np.eye(4)
        T[:3, :3] = rot
        T[:3, 3] = p
        curve[i] = T

        dT = np.zeros((4, 4))
        dT[:3, :3] = dR_ds
        dT[:3, 3] = dp_ds
        dcurve[i] = dT

    return curve, dcurve


def generate_vertical_loop_roll_old(n_points, r, center_xy, z0, n_rolls=1.0):
    """
    Generates a vertical loop (circle in the XZ plane) with a roll along the tangent.
    This curve forces the pitch angle to sweep through +/- 90 degrees, which
    triggers Gimbal Lock in Euler-angle-based metrics (like the baseline),
    but remains perfectly smooth in SE(3).
    """
    x0, y0 = center_xy
    curve = np.zeros((n_points, 4, 4))
    dcurve = np.zeros((n_points, 4, 4))

    # Skew-symmetric matrices for the world x and y axes
    S_ex = np.asarray(Utils.S(np.array([1.0, 0.0, 0.0])))
    S_ey = np.asarray(Utils.S(np.array([0.0, 1.0, 0.0])))

    for i in range(n_points):
        s = i / n_points
        theta = 2.0 * np.pi * s
        phi = 2.0 * np.pi * n_rolls * s

        # 1. Position: Circle in XZ plane (Vertical Loop)
        # Center is at (x0, y0, z0 + r). Lowest point is at theta=0.
        p = np.array([x0 + r * np.sin(theta), y0, z0 + r - r * np.cos(theta)])
        dp_ds = np.array(
            [2.0 * np.pi * r * np.cos(theta), 0.0, 2.0 * np.pi * r * np.sin(theta)]
        )

        # 2. Orientation
        # Base frame: Pitch by theta around the y-axis so local x points along tangent.
        # (If your Utils lacks roty, use Utils.rot(np.array([0,1,0]), theta))
        R_base = np.asarray(Utils.roty(theta))[:3, :3]

        # Roll frame: rotation along the local x-axis (tangent)
        R_roll = np.asarray(Utils.rotx(phi))[:3, :3]

        # Total rotation
        rot = R_base @ R_roll

        # 3. Orientation Derivative
        # dR_base/ds = (dtheta/ds) * S_ey @ R_base
        dR_base_ds = 2.0 * np.pi * (S_ey @ R_base)
        # dR_roll/ds = (dphi/ds) * S_ex @ R_roll
        dR_roll_ds = 2.0 * np.pi * n_rolls * (S_ex @ R_roll)

        dR_ds = dR_base_ds @ R_roll + R_base @ dR_roll_ds

        # 4. Assemble Homogeneous Matrices
        T = np.eye(4)
        T[:3, :3] = rot
        T[:3, 3] = p
        curve[i] = T

        dT = np.zeros((4, 4))
        dT[:3, :3] = dR_ds
        dT[:3, 3] = dp_ds
        dcurve[i] = dT

    return curve, dcurve


def generate_vertical_loop_roll(n_points, r, center_xy, z0, n_rolls=1.0):
    """
    Generates a vertical loop (circle in the XZ plane) with a roll along the tangent.
    """
    x0, y0 = center_xy
    curve = np.zeros((n_points, 4, 4))
    dcurve = np.zeros((n_points, 4, 4))

    S_ex = np.asarray(Utils.S(np.array([1.0, 0.0, 0.0])))
    S_ey = np.asarray(Utils.S(np.array([0.0, 1.0, 0.0])))

    for i in range(n_points):
        s = i / n_points
        theta = 2.0 * np.pi * s
        phi = 2.0 * np.pi * n_rolls * s

        # 1. Position: Circle in XZ plane
        p = np.array([x0 + r * np.sin(theta), y0, z0 + r - r * np.cos(theta)])
        dp_ds = np.array(
            [2.0 * np.pi * r * np.cos(theta), 0.0, 2.0 * np.pi * r * np.sin(theta)]
        )

        # 2. Orientation (FIXED: roty(-theta) to align body x-axis with dp_ds)
        R_base = np.asarray(Utils.roty(-theta))[:3, :3]

        # Roll frame: rotation along the local x-axis (tangent)
        R_roll = np.asarray(Utils.rotx(phi))[:3, :3]

        rot = R_base @ R_roll

        # 3. Orientation Derivative (FIXED: leading negative sign)
        dR_base_ds = -2.0 * np.pi * (S_ey @ R_base)
        dR_roll_ds = 2.0 * np.pi * n_rolls * (S_ex @ R_roll)

        dR_ds = dR_base_ds @ R_roll + R_base @ dR_roll_ds

        # 4. Assemble
        T = np.eye(4)
        T[:3, :3] = rot
        T[:3, 3] = p
        curve[i] = T

        dT = np.zeros((4, 4))
        dT[:3, :3] = dR_ds
        dT[:3, 3] = dp_ds
        dcurve[i] = dT

    return curve, dcurve


def generate_toroidal_coil(n_points, R, r, center_xy, z0, n_coils=3.0, n_rolls=0.0):
    """
    Generates a coil that wraps `n_coils` times around a torus of major radius R
    and minor radius r. The AUV tracks the tangent and uses the tube's outward
    normal as its "up" vector.

    Parameters
    ----------
    n_points : int
        Number of samples.
    R : float
        Major radius (radius of the torus center ring).
    r : float
        Minor radius (radius of the tube).
    center_xyz : tuple
        Center of the torus in the world frame.
    n_coils : float
        Number of times the coil wraps around the tube.
    n_rolls : float
        Optional additional roll around the tangent axis.
    """
    x0, y0 = center_xy
    curve = np.zeros((n_points, 4, 4))
    dcurve = np.zeros((n_points, 4, 4))

    for i in range(n_points):
        s = i / n_points
        Theta = 2.0 * np.pi * s  # Major angle (around the torus)
        Phi = 2.0 * np.pi * n_coils * s  # Minor angle (around the tube)

        dTheta = 2.0 * np.pi
        dPhi = 2.0 * np.pi * n_coils

        cT, sT = np.cos(Theta), np.sin(Theta)
        cP, sP = np.cos(Phi), np.sin(Phi)

        # Local frame for the major circle
        e_r = np.array([cT, sT, 0.0])  # radial outward
        e_theta = np.array([-sT, cT, 0.0])  # tangent to major circle
        e_z = np.array([0.0, 0.0, 1.0])  # world up

        # 1. Position and Velocity (tangent)
        p = np.array([x0, y0, z0]) + (R + r * sP) * e_r + r * (1.0 - cP) * e_z

        v = (dTheta * (R + r * sP)) * e_theta + dPhi * r * (cP * e_r + sP * e_z)

        # 2. Acceleration (derivative of velocity)
        acc = (
            -(dTheta**2 * (R + r * sP) + dPhi**2 * r * sP + dTheta * dPhi * r * sP)
            * e_r
            + (2.0 * dTheta * dPhi * r * cP) * e_theta
            + (dPhi**2 * r * cP) * e_z
        )

        # 3. Outward normal to the torus tube and its derivative
        n = sP * e_r - cP * e_z
        dn = dPhi * cP * e_r + dPhi * sP * e_z + sP * dTheta * e_theta

        # 4. Construct the orthonormal basis
        v_norm = np.linalg.norm(v)
        x_b = v / v_norm
        z_b = n
        y_b = np.cross(z_b, x_b)

        R_base = np.column_stack((x_b, y_b, z_b))

        # 5. Compute angular velocity of the base frame
        # x_dot = d/ds (v / |v|)
        x_dot = (acc - x_b * np.dot(x_b, acc)) / v_norm
        z_dot = dn
        y_dot = np.cross(z_dot, x_b) + np.cross(z_b, x_dot)

        # omega = 1/2 (x cross x_dot + y cross y_dot + z cross z_dot)
        omega_base = 0.5 * (
            np.cross(x_b, x_dot) + np.cross(y_b, y_dot) + np.cross(z_b, z_dot)
        )

        # 6. Apply optional roll around the tangent (local X axis)
        phi_roll = 2.0 * np.pi * n_rolls * s
        dphi_roll = 2.0 * np.pi * n_rolls
        R_roll = np.asarray(Utils.rotx(phi_roll))[:3, :3]

        rot = R_base @ R_roll

        # Total angular velocity in the world frame
        omega_world = omega_base + R_base @ np.array([dphi_roll, 0.0, 0.0])

        # Derivative of rotation matrix using Utils.S logic
        S_omega = np.asarray(Utils.S(omega_world))[:3, :3]
        dR_ds = S_omega @ rot

        # 7. Assemble Homogeneous Matrices
        T = np.eye(4)
        T[:3, :3] = rot
        T[:3, 3] = p
        curve[i] = T

        dT = np.zeros((4, 4))
        dT[:3, :3] = dR_ds
        dT[:3, 3] = v
        dcurve[i] = dT

    return curve, dcurve


# ==========================================================================
# Registry
# ==========================================================================
CURVE_BUILDERS = {
    "circle": make_circle,
    "ellipse": make_ellipse,
    "lissajous_3d": make_lissajous_3d,
    "trefoil": make_trefoil_knot,
    "vertical_coil": make_vertical_coil,
    "test": generate_barrel_roll_circle,
    # "gimbal": generate_vertical_loop_roll,
    "gimbal": generate_toroidal_coil,
}


def build_curve(curve_type, **kwargs):
    if curve_type not in CURVE_BUILDERS:
        raise ValueError(
            f"Unknown curve type: {curve_type}. " f"Available: {list(CURVE_BUILDERS)}"
        )
    return CURVE_BUILDERS[curve_type](**kwargs)
