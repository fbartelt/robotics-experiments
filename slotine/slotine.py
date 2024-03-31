#%%
import numpy as np
from scipy.linalg import block_diag
from scipy.optimize import root

def quat_to_rot(q):
    """Converts a quaternion to a rotation matrix"""
    w, x, y, z = q.ravel()
    return np.array([[2*x**2+2*w**2-1, 2*(x*y-z*w), 2*(x*z+y*w)],
                      [2*(x*y+w*z), 2*y**2+2*w**2-1, 2*(y*z-w*x)],
                      [2*(x*z-y*w), 2*(y*z+w*x), 2*z**2+2*w**2-1]])

def skew(q):
    """Maps a vector to a skew-symmetric matrix"""
    q = q.ravel()
    return np.array([[0, -q[2, 0], q[1, 0]],
                     [q[2, 0], 0, -q[0, 0]],
                     [-q[1, 0], q[0, 0], 0]])

def Pa(R):
    """Returns the antisymmetric part of a matrix"""
    return 0.5*(R - R.T)

def skew_inv(S):
    """Maps a vector to a skew-symmetric matrix"""
    return np.array([S[2,1], S[0,2], S[1,0]])


def matricize_inertia_param(v):
    """Converts a vector of inertia parameters to a matrix"""
    return np.array([[v[0, 0], v[1, 0], v[2, 0], 0, 0, 0],
                     [0, v[0, 0], 0, v[1, 0], v[2, 0], 0],
                     [0, 0, v[0, 0], 0, v[1, 0], v[2, 0]]])

#%%
#Physical parameters
rho = 8050 #density, kg/m^3 
r = 0.5 #cylinder radius, m
h = 3. #cylinder height, m
m = rho*np.pi*(r^2)*h #cylinder mass, kg
num_freq = 5 #number of frequencies in reference signal

fixed_sample = True #change to false if you want to study regularization; otherwise use true to replicate baselines

if fixed_sample:
    r_p = np.array([0,0,h/2]).reshape(-1, 1) #measurement point
    N = 6
    r_i = np.array([[0,0,h/2],[0,0,-h/2],[r,0,0],[-r,0,0],[0,r,0],[0,-r,0]])
else:
    pass
    # N_v = 10 #number of attachment points in vertical dim.
    # N_h = 10 #number in horizontal dim
    # N = N_v*N_h
    # _, r_i = sample_points(N_v,N_h,h,r,0.1) #seed robots semi-randomly around surface of cylinder

I_cm = (1/12) * np.diag([m*(3*r**2+h**2),m*(3*r**2+h**2),6*m*r**2])
I_p = I_cm + m*(r_p.T @ r_p @ np.eye(3) -r_p @ r_p.T)

#frequencies & phase shifts for reference signal computation
rng = np.random.default_rng()
freqs = rng.uniform(0,1,(6,num_freq))
phases = np.zeros((6,num_freq))

#define  reference signals (combinations of sinusoids)
def w_d(t):
    return (1/num_freq)*np.sum(np.vstack([np.cos(freqs[0,:]*t+phases[0,:]),np.sin(freqs[1,:]*t+phases[1,:]),np.cos(freqs[2,:]*t+phases[2,:])]),axis=0)

def alpha_d(t):
    return (1/num_freq)*np.sum(np.vstack([-freqs[0,:]*np.sin(freqs[0,:]*t+phases[0,:]),freqs[1,:]*np.cos(freqs[1,:]*t+phases[1,:]),-freqs[2,:]*np.sin(freqs[2,:]*t+phases[2,:])]),axis=0)

def v_d(t):
    return (10/num_freq)*np.sum(np.vstack([np.cos(freqs[3,:]*t+phases[3,:]),np.cos(freqs[4,:]*t+phases[4,:]),np.cos(freqs[5,:]*t+phases[5,:])]),axis=0)

def a_d(t):
    return (10/num_freq)*np.sum(np.vstack([-freqs[3,:]*np.sin(freqs[3,:]*t+phases[3,:]),-freqs[4,:]*np.sin(freqs[4,:]*t+phases[4,:]),-freqs[5,:]*np.sin(freqs[5,:]*t+phases[5,:])]),axis=0)

def dq_d(t):
    return np.vstack([v_d(t),w_d(t)])

def ddq_d(t):
    return np.vstack([a_d(t),alpha_d(t)])

def adaptive_dynamics(x, x_d, R, R_d, dq, a_hat, r_hat, t, lambda_, Kd, t_drop=np.inf, verbose=False):
    """Computes the adaptive dynamics for cooperative task

    Parameters:
    ----------
    x : np.ndarray
        current position
    x_d : np.ndarray
        desired position
    R : np.ndarray
        current rotation matrix
    R_d : np.ndarray
        desired rotation matrix
    dq : np.ndarray
        current velocity
    a_hat : np.ndarray
        estimated parameter
    r_hat : np.ndarray
        estimated parameter
    t : float
        current time
    t_drop : float, optional
        drop time, by default np.inf
    
    Returns:
    -------
    ddq : np.ndarray
        acceleration
    da : np.ndarray
        parameter derivative
    dr : np.ndarray
        parameter derivative
    dV : float
        Lyapunov function derivative
    ds : np.ndarray
        derivative of the composite error
    """

    dx = dq[:3].copy()
    w = dq[3:].copy()

    # Compute error signals
    Re = R_d.T @ R
    we = w - w_d(t)
    sigma = we + lambda_ * R_d @ skew_inv(Pa(Re))
    dx_e = dx - v_d(t)
    x_e = x - x_d
    s = np.vstack([dx_e + lambda_ * x_e, sigma])

    alpha_r = alpha_d(t) - lambda_ * skew(w_d(t)) @ R_d @ skew_inv(Pa(Re)) - lambda_ * R_d @ skew_inv(Pa(skew(R_d.T @ we) @ Re))
    w_r = w_d(t) - lambda_ * R_d @ skew_inv(Pa(Re))
    a_r = a_d(t) - lambda_ * dx_e
    v_r = v_d(t) - lambda_ * x_e

    ddq_r = np.vstack([a_r, alpha_r])
    dq_r = np.vstack([v_r, w_r])

    # Compute regressors (from equations 3 and 4)
    Y_linear = np.hstack([a_r, -skew(alpha_r) @ R - skew(w) @ skew(w_r) @ R, np.zeros((3,6))])
    Y_rotational = np.hstack([np.zeros((3,1)),
                              skew(a_r) @ R + skew(w) @ skew(v_r) @ R - skew(w_r) @ skew(dx) @ R,
                              R @ matricize_inertia_param(R.T @ alpha_r) @ R.T + skew(w) @ R @ matricize_inertia_param(R.T @ w_r)])
    Y = np.vstack([Y_linear, Y_rotational])

    # Compute true dynamics matrices
    H = np.vstack([np.hstack([m * np.eye(3), m * skew(R @ r_p)]),
                   np.hstack([-m * skew(R @ r_p), R @ I_p @ R.T])])
    C = np.vstack([np.hstack([np.zeros((3,3)), m * skew(w) @ skew(R @ r_p)]),
                   np.hstack([-m * skew(w) @ skew(R @ r_p), skew(w) @ R @ I_p @ R.T - m * skew(skew(R @ r_p) @ dx)])])
    
    off_diag = m * skew(w) @ R @ skew(r_p) @ R.T - m * R @ skew(r_p) @ R.T @ skew(w)
    H_dot = np.vstack([np.hstack([np.zeros((3,3)), off_diag]),
                       np.hstack([-off_diag, skew(w) @ R @ I_p @ R.T - R @ I_p @ R.T @ skew(w)])])
    
    # Apply control law
    input_ = np.zeros((1, 6))
    F = np.zeros((N, 0))
    for i in range(N):
        # G is the M matrix in the paper (grasping matrix)
        G = np.vstack([np.hstack([np.eye(3), np.zeros((3,3))]),
                       np.hstack([skew(R @ r_i[i, :].T), np.eye(3)])])
        G_hat = np.vstack([np.hstack([np.eye(3), np.zeros((3,3))]),
                           np.hstack([skew(R @ r_hat[i, :].T), np.eye(3)])])
        G_hat_inv = np.vstack([np.hstack([np.eye(3), np.zeros((3,3))]),
                               np.hstack([-skew(R @ r_hat[i, :].T), np.eye(3)])])
        F[i, :] = Y @ a_hat[i, :] - Kd @ s
        tau = G_hat_inv @ F[i, :]
        input_ += G @ tau

    #use root finding to solve for accelerations (avoid inverting H)
    def root_eq(ddq):
        return H @ ddq + C @ dq - input_
    ddq = root(root_eq, np.zeros((6, 1)), method='lm').x

    # Implement adaptation laws
    Y_g, a_e, r_e = np.zeros((N, 6)), np.zeros((N, 3)), np.zeros((N, 3))
    # dr, da, g_o, g_r = np.zeros((N, 3)), np.zeros((N, 3)), np.zeros((N, 6)), np.zeros((N, 6))
    for i in range(N):
        Y_g[i, :] = np.hstack([np.zeros((3,3)), skew(F[i, :3]) @ R])
        a_e[i, :] = a_hat[i, :] - a_r
        r_e[i, :] = r_hat[i, :] - r_i[i, :]
        # g_o[i] = np.linalg.inv(P_o) @ np.linalg.inv(pnorm_hessian(P_o @ a_hat[i, :], p, tol)) @ np.linalg.inv(P_o)
        # g_r[i] = np.linalg.inv(P_r) @ np.linalg.inv(pnorm_hessian(P_r @ r_hat[i, :], p, tol)) @ np.linalg.inv(P_r)
        # da[i, :] = -g_o[i] @ Y_o.T @ s
        # dr[i, :] = -g_r[i] @ Y_g[i, :].T @ s
    
    ds = ddq - ddq_r
    # ds2 = 
    
    dV = s.T @ (-H @ ddq_r - C @ dq_r)
    dV2 = s.T @ H @ ds + 0.5 * s.T @ H_dot @ s
    r_err = 0
    o_err = 0
    for i in range(N):
        # pass
        dV += s.T @ (F[i, :] + Y_g[i, :] @ r_e[i, :])
        # dV2 += a_e[i, :] @ np.linalg.inv(g_o[i]) @ da[i, :] + r_e[i, :] @ np.linalg.inv(g_r[i]) @ dr[i, :]
        # o_err += np.abs(s.T @ Y_o @ a_e[i, :] + a_e[i, :].T @ P_o @ pnorm_hessian(P_o @ a_hat[i, :], p, tol) @ P_o @ da[i, :])
        # r_err += np.abs(s.T @ Y_g[i, :] @ r_e[i, :] + r_e[i, :].T @ P_r @ pnorm_hessian(P_r @ r_hat[i, :], p, tol) @ P_r @ dr[i, :])
    
    if verbose:
        pass
    
    return ddq, da, dr, dV, ds

        


#%%
"""Julia code"""
