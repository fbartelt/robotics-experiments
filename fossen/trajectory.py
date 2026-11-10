import numpy as np
import uaibot as ub
import uaibot_cpp_bind as ub_cpp

E = np.zeros((6, 4, 4))
for i in range(6):
    ei = np.eye(6)[i, :]
    E[i, :, :] = ub_cpp.SmapSE3(ei)


def compute_traj_normal(H, H_d_t, eps=1e-3):
    """
    Computes the normal component -L_V[D]^T for the trajectory tracking baseline.
    """
    D = ub_cpp.ee_distance(H, H_d_t)
    L = np.zeros(6)

    for i in range(6):
        # H_pert = exp(eps * E_i) @ H
        # For finite differences, 1st-order Taylor expansion is exact enough
        H_pert = ub_cpp.expSE3(E[i] * eps) @ H

        D_pert = ub_cpp.ee_distance(H_pert, H_d_t)
        L[i] = (D_pert - D) / eps

    return -L, D


# ==========================================================================
# Trajectory
# ==========================================================================


def s_inv_se3(V):
    """
    Extracts the 6D twist [v; w] from a 4x4 matrix V in se(3).
    V is assumed to be of the form S(xi) = [[skew(w), v], [0, 0]].
    """
    return np.array(
        [
            V[0, 3],
            V[1, 3],
            V[2, 3],  # v (linear)
            V[2, 1],
            V[0, 2],
            V[1, 0],  # w (angular)
        ]
    )


def compute_trajectory_timing(curve, curve_deriv, alpha=0.3):
    """
    Computes the time-scaling factor beta for a trajectory tracking baseline.

    Args:
        curve: (N, 4, 4) array of SE(3) poses H_d(s) for s in [0, 1).
        curve_deriv: (N, 4, 4) array of derivatives dH_d/ds.
        alpha: Average speed of the nominal controller (k_T at D=0).

    Returns:
        beta: Time scaling factor such that s(t) = beta * t.
        gamma: Average tangent speed w.r.t the curve parameter s.
    """
    N = len(curve)
    norms = []

    for i in range(N):
        # V = (dH_d/ds) * H_d^{-1}
        V = curve_deriv[i] @ np.linalg.inv(curve[i])

        # Extract the Lie algebra vector and take its norm
        xi_d_s = s_inv_se3(V)
        norms.append(np.linalg.norm(xi_d_s))

    gamma = np.mean(norms)
    beta = alpha / gamma

    return beta, gamma


def get_trajectory_reference(t, curve, curve_deriv, beta):
    """
    Evaluates the time-parameterized trajectory H_d(t) and its
    time-derivative Xi[H_d](t) at a given time t.

    Args:
        t: Current time in seconds.
        curve: (N, 4, 4) array of SE(3) poses.
        curve_deriv: (N, 4, 4) array of derivatives.
        beta: Time scaling factor.

    Returns:
        H_d_t: (4, 4) Target pose at time t.
        Xi_traj: (6,) Target tangent twist at time t.
    """
    N = len(curve)

    # Time to parameter s in [0, 1)
    s = (beta * t) % 1.0

    # Interpolate H_d and dH_d/ds using linear interpolation
    idx_float = s * N
    idx0 = int(np.floor(idx_float)) % N
    idx1 = (idx0 + 1) % N
    frac = idx_float - np.floor(idx_float)

    # 1. Interpolate Pose H_d(t)
    H_d_t = (1 - frac) * curve[idx0] + frac * curve[idx1]

    # Re-orthogonalize rotation matrix to stay exactly in SO(3)
    R = H_d_t[:3, :3]
    U, _, Vt = np.linalg.svd(R)
    H_d_t[:3, :3] = U @ Vt
    H_d_t[3, :] = [0, 0, 0, 1]

    # 2. Interpolate Derivative dH_d/ds
    dH_ds_t = (1 - frac) * curve_deriv[idx0] + frac * curve_deriv[idx1]

    # 3. Compute Twist Xi[H_d](s) = S^{-1}(dH_d/ds * H_d^{-1})
    V = dH_ds_t @ np.linalg.inv(H_d_t)
    Xi_d_s = s_inv_se3(V)

    # 4. Chain Rule: Xi[H_d](t) = ds/dt * Xi[H_d](s) = beta * Xi[H_d](s)
    Xi_traj = beta * Xi_d_s

    return H_d_t, Xi_traj


def precompute_trajectory_baseline(curve, curve_deriv, dt, T, alpha=0.3):
    """
    Precomputes the time-parameterized reference trajectory for the baseline.

    Args:
        curve: (N, 4, 4) array of SE(3) poses H_d(s).
        curve_deriv: (N, 4, 4) array of derivatives dH_d/ds.
        dt: Simulation time step (e.g., 0.01).
        T: Total simulation time in seconds.
        alpha: Average nominal controller tangent speed (k_T at D=0).

    Returns:
        traj_poses: (N_steps, 4, 4) array of target poses at each time step.
        traj_twists: (N_steps, 6) array of target twists at each time step.
    """
    N = len(curve)

    # 1. Compute gamma (average curve tangent speed w.r.t parameter s)
    norms = []
    for i in range(N):
        V = curve_deriv[i] @ np.linalg.inv(curve[i])
        xi_d_s = s_inv_se3(V)
        norms.append(np.linalg.norm(xi_d_s))
    gamma = np.mean(norms)

    # 2. Compute time scaling beta
    beta = alpha / gamma

    # 3. Allocate arrays
    N_steps = int(T / dt) + 1
    traj_poses = np.zeros((N_steps, 4, 4))
    traj_twists = np.zeros((N_steps, 6))

    # 4. Precompute every step
    for step in range(N_steps):
        t = step * dt
        s = (beta * t) % 1.0

        # Find interpolation indices and fraction
        idx_float = s * N
        idx0 = int(np.floor(idx_float)) % N
        idx1 = (idx0 + 1) % N
        frac = idx_float - np.floor(idx_float)

        # Extract spatial twists at the two adjacent samples
        V0 = curve_deriv[idx0] @ np.linalg.inv(curve[idx0])
        V1 = curve_deriv[idx1] @ np.linalg.inv(curve[idx1])
        xi0 = s_inv_se3(V0)
        xi1 = s_inv_se3(V1)

        # Interpolate twist (in the Lie Algebra)
        xi_interp = (1 - frac) * xi0 + frac * xi1

        # Compute the advance in the curve parameter (ds = frac / N)
        ds = frac / N

        # Compute H_d_t = exp( ds * S(xi_interp) ) @ curve[idx0]
        # SmapSE3 converts the 6D twist to a 4x4 se(3) matrix
        S_xi = ub_cpp.SmapSE3(xi_interp * ds)
        exp_S_xi = ub_cpp.expSE3(S_xi)
        H_d_t = exp_S_xi @ curve[idx0]

        traj_poses[step] = H_d_t
        traj_twists[step] = beta * xi_interp

        # # Interpolate Pose
        # H_d_t = (1 - frac) * curve[idx0] + frac * curve[idx1]
        #
        # # Re-orthogonalize rotation matrix to stay exactly in SO(3)
        # R = H_d_t[:3, :3]
        # U, _, Vt = np.linalg.svd(R)
        # H_d_t[:3, :3] = U @ Vt
        # H_d_t[3, :] = [0, 0, 0, 1]

        # # Interpolate Derivative
        # dH_ds_t = (1 - frac) * curve_deriv[idx0] + frac * curve_deriv[idx1]
        #
        # # Compute Twist and apply chain rule (ds/dt = beta)
        # V = dH_ds_t @ np.linalg.inv(H_d_t)
        # Xi_d_s = s_inv_se3(V)
        # Xi_traj = beta * Xi_d_s
        #
        # # Store
        # traj_poses[step] = H_d_t
        # traj_twists[step] = Xi_traj

    return traj_poses, traj_twists
