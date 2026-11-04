import numpy as np
from scipy.spatial.transform import Rotation as R

TWO_PI = 2.0 * np.pi


def wrap_vec(a):
    """Component-wise wrap to [-pi, pi] (Eq. 60 specialized to T = 2*pi)."""
    return (a + np.pi) % TWO_PI - np.pi


def se3_curve_to_r3t3(curve):
    """
    Convert an SE(3) curve (N, 4, 4) to coordinates on R^3 x T^3.

    Returns:
        p_curve   : (N, 3) positions in R^3
        phi_curve : (N, 3) Euler angles (scipy 'xyz'), UNWRAPPED along s.
                    The physical angle is phi_curve % 2*pi; the unwrapping
                    makes the finite-difference tangent smooth.
    """
    N = len(curve)
    p_curve = curve[:, :3, 3].copy()
    phi_curve = np.zeros((N, 3))
    for i in range(N):
        phi_curve[i] = R.from_matrix(curve[i, :3, :3]).as_euler("xyz")
    phi_curve = np.unwrap(phi_curve, axis=0)
    return p_curve, phi_curve


def rezende_distance_gradient(p0, phi0, p_curve, phi_curve):
    """
    D(p0, phi0) = min_i sqrt(|p0 - p_i|^2 + sum_j wrap(phi0_j - phi_ij)^2)

    Returns
        D        : scalar
        s_star   : index of the closest sample
        grad     : 6-vector = (D_vec/D) = unit-norm distance gradient
                   (holds automatically: |grad| = 1 by construction)
    """
    diff_p = p_curve - p0
    sq_pos = np.sum(diff_p**2, axis=1)

    diff_phi = wrap_vec(phi_curve - phi0)
    sq_ang = np.sum(diff_phi**2, axis=1)

    sq_total = sq_pos + sq_ang
    s_star = int(np.argmin(sq_total))
    D = float(np.sqrt(sq_total[s_star]))
    if D < 1e-10:
        return 0.0, s_star, np.zeros(6)

    grad_p = (p0 - p_curve[s_star]) / D
    grad_phi = -diff_phi[s_star] / D
    return D, s_star, np.concatenate([grad_p, grad_phi])


def rezende_vf_r3t3(
    p0,
    phi0,
    p_curve,
    phi_curve,
    v_r=1.0,
    k_f=1.0,
    kn1=1.0,
    kn2=1.0,
    kt1=1.0,
    kt2=1.0,
    kt3=1.0,
):
    """
    Rezende's vector field on R^3 x T^3.

    Returns
        psi : 6-vector [dp/dt; dphi/dt]
              dp/dt  is the INERTIAL frame linear velocity (m/s)
              dphi/dt is the Euler-angle rate (rad/s) in scipy 'xyz'
    """
    N = len(p_curve)
    D, s_star, grad = rezende_distance_gradient(p0, phi0, p_curve, phi_curve)

    # Tangent (central difference on the unwrapped curve, circular in s)
    s_next = (s_star + 1) % N
    s_prev = (s_star - 1) % N
    dp = 0.5 * (p_curve[s_next] - p_curve[s_prev])
    dphi = 0.5 * (phi_curve[s_next] - phi_curve[s_prev])
    tangent = np.concatenate([dp, dphi])
    t_norm = np.linalg.norm(tangent)
    if t_norm > 1e-12:
        tangent = tangent / t_norm

    # G-type function G(D) = (2/pi) arctan(k_f D), H(D) = sqrt(1-G^2)
    H = kt1 * (1 - kt2 * np.tanh(kt3 * np.sqrt(D)))
    G = kn1 * np.tanh(kn2 * np.sqrt(D))
    # G = (2.0 / np.pi) * np.arctan(k_f * D)
    # H = np.sqrt(max(0.0, 1.0 - G * G))

    # Ψ = v_r * (-G ∇D + H T*)
    return -G * grad + H * tangent


def euler_rates_to_omega_world(phi, phi_dot, eps=1e-7):
    """
    World-frame angular velocity ω such that R_dot = S(ω) R,
    given Euler angles φ and Euler rates φ_dot in scipy 'xyz' convention.
    """
    R_old = R.from_euler("xyz", phi).as_matrix()
    R_new = R.from_euler("xyz", phi + phi_dot * eps).as_matrix()
    dR = (R_new - R_old) / eps
    S_omega = dR @ R_old.T
    return (
        np.array(
            [
                S_omega[2, 1] - S_omega[1, 2],
                S_omega[0, 2] - S_omega[2, 0],
                S_omega[1, 0] - S_omega[0, 1],
            ]
        )
        / 2.0
    )


def vf_to_fossen_twist(psi, phi):
    """
    Rezende VF output psi = [dp/dt; dphi/dt] -> Fossen target_twist
    = [geometric_v_inertial; omega_inertial].
    """
    v_inertial = psi[0:3]
    omega_world = euler_rates_to_omega_world(phi, psi[3:6])
    return np.concatenate([v_inertial, omega_world])
