#%%
import numpy as np
from scipy.linalg import block_diag
from scipy.optimize import root, least_squares

def quat_to_rot(q):
    """Converts a quaternion to a rotation matrix"""
    w, x, y, z = q.ravel()
    return np.array([[2*x**2+2*w**2-1, 2*(x*y-z*w), 2*(x*z+y*w)],
                      [2*(x*y+w*z), 2*y**2+2*w**2-1, 2*(y*z-w*x)],
                      [2*(x*z-y*w), 2*(y*z+w*x), 2*z**2+2*w**2-1]])

def skew(q):
    """Maps a vector to a skew-symmetric matrix"""
    q = q.ravel()
    return np.array([[0, -q[2], q[1]],
                     [q[2], 0, -q[0]],
                     [-q[1], q[0], 0]])

def Pa(R):
    """Returns the antisymmetric part of a matrix"""
    return 0.5*(R - R.T)

def skew_inv(S):
    """Maps a vector to a skew-symmetric matrix"""
    return np.array([[S[2,1]], [S[0,2]], [S[1,0]]])


def matricize_inertia_param(v):
    """Converts a vector of inertia parameters to a matrix"""
    v = v.ravel()
    return np.array([[v[0], v[1], v[2], 0, 0, 0],
                     [0, v[0], 0, v[1], v[2], 0],
                     [0, 0, v[0], 0, v[1], v[2]]])

def p_grad(x, p, tol=1e-10):
    """Computes the gradient of the l-p norm"""
    x = x.ravel()
    fl_abs = np.maximum(np.abs(x), tol*np.ones(len(x)))
    return ((fl_abs**(p-2)) * (x + tol*np.sign(x))).reshape(-1, 1)

def p_norm(x, p, tol=1e-10):
    """The name is misleading; this does not compute the p-norm, but rather (p-norm)^(p) * 1/p"""
    x = x.ravel()
    fl_abs = np.maximum(np.abs(x), tol*np.ones(len(x)))
    return (1/p) * np.sum(fl_abs**p)

def pnorm_divergence(y, x, p, tol=1e-10):
    """Computes the divergence of the l-p norm"""
    return p_norm(y, p, tol) - p_norm(x, p, tol) - (y-x).T @ p_grad(x, p, tol)

def pnorm_hessian(x, p, tol=1e-10):
    """Computes the Hessian of the l-p norm"""
    x = x.ravel()
    fl_abs = np.maximum(np.abs(x), tol*np.ones(len(x)))
    norm_diag = (p-1) * fl_abs**(p-4) * (x**2 + tol*np.ones(len(x)))
    return np.diag(norm_diag)
#%%
#Physical parameters
rho = 8050 #density, kg/m^3 
r = 0.5 #cylinder radius, m
h = 3. #cylinder height, m
m = rho*np.pi*(r ** 2)*h #cylinder mass, kg
num_freq = 5 #number of frequencies in reference signal

fixed_sample = True #change to false if you want to study regularization; otherwise use true to replicate baselines

if fixed_sample:
    r_p = np.array([0,0,h/2]).reshape(-1, 1) #measurement point
    N = 6
    r_i = np.array([[0,0,h/2],[0,0,-h/2],[r,0,0],[-r,0,0],[0,r,0],[0,-r,0]]).reshape(-1, 3, 1)
else:
    pass
    # N_v = 10 #number of attachment points in vertical dim.
    # N_h = 10 #number in horizontal dim
    # N = N_v*N_h
    # _, r_i = sample_points(N_v,N_h,h,r,0.1) #seed robots semi-randomly around surface of cylinder

I_cm = (1/12) * np.diag([m*(3*r**2+h**2),m*(3*r**2+h**2),6*m*r**2])
I_p = I_cm + m*(r_p.T @ r_p * np.eye(3) -r_p @ r_p.T)

#frequencies & phase shifts for reference signal computation
rng = np.random.default_rng()
freqs = rng.uniform(0,1,(6,num_freq))
phases = np.zeros((6,num_freq))

#define  reference signals (combinations of sinusoids)
def w_d(t):
    return (1/num_freq)*np.sum(np.vstack([np.cos(freqs[0,:]*t+phases[0,:]),np.sin(freqs[1,:]*t+phases[1,:]),np.cos(freqs[2,:]*t+phases[2,:])]),axis=1).reshape(-1, 1)

def alpha_d(t):
    return (1/num_freq)*np.sum(np.vstack([-freqs[0,:]*np.sin(freqs[0,:]*t+phases[0,:]),freqs[1,:]*np.cos(freqs[1,:]*t+phases[1,:]),-freqs[2,:]*np.sin(freqs[2,:]*t+phases[2,:])]),axis=1).reshape(-1, 1)

def v_d(t):
    return (10/num_freq)*np.sum(np.vstack([np.cos(freqs[3,:]*t+phases[3,:]),np.cos(freqs[4,:]*t+phases[4,:]),np.cos(freqs[5,:]*t+phases[5,:])]),axis=1).reshape(-1, 1)

def a_d(t):
    return (10/num_freq)*np.sum(np.vstack([-freqs[3,:]*np.sin(freqs[3,:]*t+phases[3,:]),-freqs[4,:]*np.sin(freqs[4,:]*t+phases[4,:]),-freqs[5,:]*np.sin(freqs[5,:]*t+phases[5,:])]),axis=1).reshape(-1, 1)

def dq_d(t):
    return np.vstack([v_d(t),w_d(t)])

def ddq_d(t):
    return np.vstack([a_d(t),alpha_d(t)])

def adaptive_dynamics(x, x_d, R, R_d, dq, a_hat, r_hat, t, lambda_, Kd, t_drop=np.inf, verbose=False):
    """Computes the adaptive dynamics for cooperative task of N agents.

    Parameters:
    ----------
    x : np.ndarray
        Current position. Shape (3, 1)
    x_d : np.ndarray
        Desired position. Shape (3, 1)
    R : np.ndarray
        Current rotation matrix. Shape (3, 3)
    R_d : np.ndarray
        Desired rotation matrix. Shape (3, 3)
    dq : np.ndarray
        Current velocity. Shape (6, 1)
    a_hat : np.ndarray
        Estimated parameter. Shape (N, 10, 1)
    r_hat : np.ndarray
        Estimated parameter. Shape (N, 10, 1)
    t : float
        current time
    t_drop : float, optional
        drop time, by default np.inf
    
    Returns:
    -------
    ddq : np.ndarray
        acceleration. Shape (6, 1)
    da : np.ndarray
        parameter derivative. Shape (N, 10, 1)
    dr : np.ndarray
        parameter derivative. Shape (N, 3, 1)
    dV : float
        Lyapunov function derivative
    ds : np.ndarray
        derivative of the composite error. Shape (6, 1)
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
                              R @ matricize_inertia_param(R.T @ alpha_r) + skew(w) @ R @ matricize_inertia_param(R.T @ w_r)])
    Y_o = np.vstack([Y_linear, Y_rotational])

    # Compute true dynamics matrices
    H = np.vstack([np.hstack([m * np.eye(3), m * skew(R @ r_p)]),
                   np.hstack([-m * skew(R @ r_p), R @ I_p @ R.T])])
    C = np.vstack([np.hstack([np.zeros((3,3)), m * skew(w) @ skew(R @ r_p)]),
                   np.hstack([-m * skew(w) @ skew(R @ r_p), skew(w) @ R @ I_p @ R.T - m * skew(skew(R @ r_p) @ dx)])])
    
    off_diag = m * skew(w) @ R @ skew(r_p) @ R.T - m * R @ skew(r_p) @ R.T @ skew(w)
    H_dot = np.vstack([np.hstack([np.zeros((3,3)), off_diag]),
                       np.hstack([-off_diag, skew(w) @ R @ I_p @ R.T - R @ I_p @ R.T @ skew(w)])])
    
    # Apply control law
    input_ = np.zeros((6, 1))
    F = np.zeros((N, 6, 1))
    for i in range(N):
        # G is the M matrix in the paper (grasping matrix)
        G = np.vstack([np.hstack([np.eye(3), np.zeros((3,3))]),
                       np.hstack([skew(R @ r_i[i]), np.eye(3)])])
        G_hat = np.vstack([np.hstack([np.eye(3), np.zeros((3,3))]), 
                           np.hstack([skew(R @ r_hat[i]), np.eye(3)])])
        G_hat_inv = np.vstack([np.hstack([np.eye(3), np.zeros((3,3))]),
                               np.hstack([-skew(R @ r_hat[i]), np.eye(3)])])
        F[i] = Y_o @ a_hat[i] - Kd @ s
        tau = G_hat_inv @ F[i]
        input_ += G @ tau

    #use root finding to solve for accelerations (avoid inverting H)
    def root_eq(ddq):
        ddq = np.array(ddq).reshape(-1, 1)
        return (H @ ddq + C @ dq - input_).ravel()
    # ddq = root(root_eq, np.zeros((6, )), method='lm').x.reshape(-1, 1)
    ddq = least_squares(root_eq, np.zeros((6, ))).x.reshape(-1, 1)
    # ddq = np.linalg.pinv(H) @ (input_ - C @ dq)

    # Implement adaptation laws
    Y_g, a_e, r_e = np.zeros((N, 6, 3)), np.zeros((N, 10, 1)), np.zeros((N, 3, 1))
    da, dr, g_o, g_r = np.zeros((N, 10, 1)), np.zeros((N, 3, 1)), np.zeros((N, 10, 10)), np.zeros((N, 3, 3))
    for i in range(N):
        Y_g[i] = np.vstack([np.zeros((3,3)), skew(F[i][:3, :]) @ R])
        a_e[i] = a_hat[i] - a_i
        r_e[i] = r_hat[i] - r_i[i]
        # da[i] = -np.linalg.inv(P_o) @ Y_o.T @ s
        # dr[i] = -np.linalg.inv(P_r) @ Y_g[i].T @ s
        # The following is the Bregman divergence modification shown in page 10.
        g_o[i] = np.linalg.inv(P_o) @ np.linalg.inv(pnorm_hessian(P_o @ a_hat[i], p, tol)) @ np.linalg.inv(P_o) 
        g_r[i] = np.linalg.inv(P_r) @ np.linalg.inv(pnorm_hessian(P_r @ r_hat[i], p, tol)) @ np.linalg.inv(P_r)
        da[i] = -g_o[i] @ Y_o.T @ s
        dr[i] = -g_r[i] @ Y_g[i].T @ s
    
    ds = ddq - ddq_r
    ds2 = np.vstack([ddq[:3] - a_d(t) + lambda_ * (dx - v_d(t)), ddq[3:] - alpha_d(t) + lambda_ * (skew(w_d(t)) @ R_d @ skew_inv(Pa(Re)) + R_d @ skew_inv(Pa(skew(R_d.T @ we) @ Re)))])
    
    dV = s.T @ (-H @ ddq_r - C @ dq_r)
    dV2 = s.T @ H @ ds + 0.5 * s.T @ H_dot @ s
    r_err = 0
    o_err = 0
    for i in range(N):
        # pass
        dV += s.T @ (F[i] + Y_g[i] @ r_e[i]) + a_e[i].T @ np.linalg.inv(P_o) @ da[i] + r_e[i].T @ np.linalg.inv(P_r) @ dr[i]
        # dV2 += a_e[i, :] @ np.linalg.inv(g_o[i]) @ da[i, :] + r_e[i, :] @ np.linalg.inv(g_r[i]) @ dr[i, :]
        # o_err += np.abs(s.T @ Y_o @ a_e[i, :] + a_e[i, :].T @ P_o @ pnorm_hessian(P_o @ a_hat[i, :], p, tol) @ P_o @ da[i, :])
        # r_err += np.abs(s.T @ Y_g[i, :] @ r_e[i, :] + r_e[i, :].T @ P_r @ pnorm_hessian(P_r @ r_hat[i, :], p, tol) @ P_r @ dr[i, :])
    
    if verbose:
        pass
    
    return ddq, da, dr, dV, ds

#%%
# Choose preset mode
sim_mode = 'l2' #can be ['l2', 'l1', 'no_arms', 'pd']

# Setup constants, etc.
rng = np.random.default_rng(200) #seed rand for repeatability

a_hat = rng.normal(0, 1, (N, 10, 1)) #initial a_hat
if not fixed_sample:
    a_hat = 100 * a_hat
if sim_mode == 'pd':
    a_hat = np.zeros((N, 10, 1))

# True a_i
a_I = np.array([I_p[0,0], I_p[0,1], I_p[0,2], I_p[1,1], I_p[1,2], I_p[2,2]]).reshape(-1 ,1)
a = np.vstack([m, m*r_p, a_I])
a_i = (1/N) * a

# Initial r_hat
r_hat = rng.normal(0, 2, (N, 3, 1))
if sim_mode == 'pd':
    r_hat = np.zeros((N, 3, 1))
if not fixed_sample:
    r_hat = 0.5 * r_hat

# Kd matrix
Kd = np.diag(np.hstack([(5e4/N)*np.ones(3), (5e3/N)*np.ones(3)]))

# Sample random initial orientations
quat = rng.uniform(0, 1, (4, 1))
quat = quat / np.linalg.norm(quat) #random quaternion
quat_d = rng.uniform(0, 1, (4, 1))
quat_d = quat_d / np.linalg.norm(quat_d) #random desired quaternion
# Cast both to rotation matrices
R = quat_to_rot(quat)
R_d = quat_to_rot(quat_d)

dq = np.zeros((6, 1)) #initial velocity = 0

# Inital state
w = np.zeros((3, 1))
x = np.zeros((3, 1))
x_d = np.zeros((3, 1))
dx = np.zeros((3, 1))

# Simulation parameters
dt = 1e-2 #timestep -- numerical integration via Heun's; error is ~ O(dt^2)
T = 60 #final time
lambda_ = 1.5 #sliding parameter
deadband = 0.01 #deadband in which to stop adaptation
t_drop = T + 10 #time for drop -- use t_drop < T for agents to turn off

p = 2 #l-p norm to be used for regularization;
#note: we only implement l1, l2, since physical scaling issues => numerical problems for p > 2.

match sim_mode:
    case ['l2', 'no_arms', 'pd']:
        p = 2
    case 'l1':
        p = 1.001

# P_o and P_r are the inverse of Gamma_o and Gamma_r in the paper
if p == 1.001:
    P_o = 1e6 * np.linalg.inv(np.diag(np.abs(a_i).ravel() + 1e4)) #p = 1.001
    P_r = 1e6 * np.linalg.inv(np.diag(np.ones(3))) #p=1.001
    tol = 1e-5
elif p == 2:
    P_o = 3e1 * np.linalg.inv(np.diag(np.abs(a_i).ravel() + 1e-2))
    P_r = 3e1 * np.linalg.inv(np.diag(np.ones(3)))
    tol = 1e-5

if sim_mode in ['no_arms', 'pd']:
    P_r = 1e20 * np.diag(np.ones(3)) #essentially zero gain
    if sim_mode == 'pd':
        P_o = 1e20 * np.diag(np.abs(a_i).ravel() + 1e4)
        tol = 1e3

# Storage for simulation data
s_tilde, rot_err, V, Vdot = [], [], [], []
hist_ddq, hist_s = [], []
s = np.zeros((6, 1))
s_hat, V_hat, lyap_params, lyap_dynamics = [], [], [], []
err_mean, param_est, arm_est, dV_params, dV_dynamics = [], [], [], [], []

H = np.vstack([np.hstack([m * np.eye(3), m * skew(R @ r_p)]),
              np.hstack([-m * skew(R @ r_p), R @ I_p @ R.T])])

verbose = False
dropped = False

for t in np.arange(0, T + dt, dt):
    print(t)
    if (t >= t_drop) and (not dropped):
        N = int(N / 2)
        dropped = True
    #unpack variables
    dx = dq[:3].copy()
    w = dq[3:].copy() 
    # Compute error signals
    Re = R_d.T @ R
    we = w - w_d(t)
    sigma = we + lambda_ * R_d @ skew_inv(Pa(Re))
    dx_tilde = dx - v_d(t)
    x_tilde = x - x_d
    s = np.vstack([dx_tilde + lambda_ * x_tilde, sigma])

    #check if simulation has diverged -- if so, decrease dt
    if np.linalg.norm(s) > 1e3:
        print(f'At t:{t}  diverged')
        break
    
    # Dynamics matrices (for debug)
    H = np.vstack([np.hstack([m * np.eye(3), m * skew(R @ r_p)]),
                   np.hstack([-m* skew(R @ r_p), R @ I_p @ R.T])])
    
    C = np.vstack([np.hstack([np.zeros((3,3)), m * skew(w) @ skew(R @ r_p)]),
                   np.hstack([-m * skew(w) @ skew(R @ r_p), skew(w) @ R @ I_p @ R.T - m * skew(skew(R @ r_p) @ dx)])])
    
    off_diag = m * skew(w) @ R @ skew(r_p) @ R.T - m * R @ skew(r_p) @ R.T @ skew(w)
    H_dot = np.vstack([np.hstack([np.zeros((3,3)), off_diag]),
                       np.hstack([-off_diag, skew(w) @ R @ I_p @ R.T - R @ I_p @ R.T @ skew(w)])])
    
    # Compute Lyapunov function & parameter errors
    a_t, r_t, g_o, g_r = [], [], [], []
    V_curr = 0.5 * (s.T @ H @ s)
    V_params = 0

    for i in range(N):
        g_o.append(np.linalg.inv(P_o) @ np.linalg.inv(pnorm_hessian(P_o @ a_hat[i], p, tol)) @ np.linalg.inv(P_o))
        g_r.append(np.linalg.inv(P_r) @ np.linalg.inv(pnorm_hessian(P_r @ r_hat[i], p, tol)) @ np.linalg.inv(P_r))
        a_t.append(a_hat[i] - a_i)
        r_t.append(r_hat[i] - r_i[i])
        V_params += pnorm_divergence(P_o @ a_i, P_o @ a_hat[i], p, tol) + pnorm_divergence(P_r @ r_i[i], P_r @ r_hat[i], p, tol)
        V_curr += pnorm_divergence(P_o @ a_i, P_o @ a_hat[i], p, tol) + pnorm_divergence(P_r @ r_i[i], P_r @ r_hat[i], p, tol)
    
    V.append(V_curr)
    if t > 0:
        pass
        # lyap_arms = sum([pnorm_divergence(P_r @ r_i[i], P_r @ arm_est[-1][1+3*(i-1):3*i], 10, 1e-10) for i in range(N)])
        # lyap_obj = sum([pnorm_divergence(P_o @ a_i, P_o @ param_est[-1][1+10*(i-1):10*i], 10, 1e-10) for i in range(N)])

    at_mean = (1/N) * sum(a_t)
    rt_mean = (1/N) * sum(r_t)

    # Logging
    err_mean.append(np.linalg.norm(at_mean) + np.linalg.norm(rt_mean))
    s_tilde.append(np.linalg.norm(s))
    rot_err.append(np.linalg.norm(R_d.T @ R - np.eye(3)))
    lyap_params.append(V_params)
    lyap_dynamics.append(0.5 * s.T @ H @ s)

    # Get dynamics at current time
    ddq, da, dr, dV, ds = adaptive_dynamics(x, x_d, R, R_d, dq, a_hat, r_hat, t, lambda_, Kd, t_drop, verbose)

    if t == 0:
        V_hat.append(V[0])
        s_hat.append(s)

    a_int, r_int = [], []

    # Intermediate step of Heun's
    for i in range(N):
        if np.linalg.norm(s) > deadband:
            a_int.append(a_hat[i] + dt * da[i])
            r_int.append(r_hat[i] + dt * dr[i])
        else:
            da[i] = np.zeros((6, 1))
            a_int.append(a_hat[i])
            dr[i] = np.zeros((3, 1))
            r_int.append(r_hat[i])

    R_d_int = np.exp(skew(w_d(t)) * dt) @ R_d
    R_int = np.exp(skew(w) * dt) @ R

    x_d_int = x_d + dt * v_d(t)
    x_int = x + dt * dx
    dq_int = dq + dt * ddq

    dx_int = dq_int[:3].copy()
    w_int = dq_int[3:].copy()

    # Second step of Heun's
    ddq_p, da_p, dr_p, dV_p, ds_p = adaptive_dynamics(x_int, x_d_int, R_int, R_d_int, dq_int, a_int, r_int, t + dt, lambda_, Kd, t_drop, verbose)
    V_hat.append(V_hat[-1] + (dt / 2) * (dV + dV_p))

    for i in range(N):
        if np.linalg.norm(s) > deadband:
            a_hat[i] = a_hat[i] + (dt / 2) * (da[i] + da_p[i])
            r_hat[i] = r_hat[i] + (dt / 2) * (dr[i] + dr_p[i])
        else:
            da_p[i] = np.zeros((6, 1))
            dr_p[i] = np.zeros((3, 1))

    dV_dynamics.append(0.5 * s.T @ H @ (ds + ds_p) + 0.5 * s.T @ H_dot @ s)

    vdot_params = 0
    for i in range(N):
        vdot_params += 0
        # vdot_params += 0.5 * a_t[i].T @ np.linalg.inv(g_o[i]) @ (da[i] + da_p[i]) + 0.5 * r_t[i].T @ np.linalg.inv(g_r[i]) @ (dr[i] + dr_p[i])
    
    dV_params.append(vdot_params)
    s_hat.append(s_hat[-1] + (dt / 2) * (ds + ds_p))

    R_d = np.exp((dt/2) * skew(w_d(t) + w_d(t + dt))) @ R_d
    R = np.exp((dt / 2) * (skew(w) + skew(w_int))) @ R

    x_d = x_d + (dt / 2) * (v_d(t) + v_d(t + dt))
    x = x + (dt / 2) * (dx + dx_int)
    dq = dq + (dt / 2) * (ddq + ddq_p)

    dx = dq[:3].copy()
    w = dq[3:].copy()

    param_est.append(np.vstack([a_hat[i] for i in range(N)]))
    arm_est.append(np.vstack([r_hat[i] for i in range(N)]))
    hist_ddq.append(ddq)
    hist_s.append(s)

match sim_mode:
    case 'l1':
        s_l1, rot_l1, lyap_l1, o_l1, arm_l1 = s_tilde, rot_err, V, param_est, arm_est
    case 'l2':
        s_l2, rot_l2, lyap_l2, o_l2, arm_l2 = s_tilde, rot_err, V, param_est, arm_est
    case 'no_arms':
        s_na, rot_na, lyap_na, o_na, arm_na = s_tilde, rot_err, V, param_est, arm_est
    case 'pd':
        s_pd, rot_pd, lyap_pd, o_pd, arm_pd = s_tilde, rot_err, V, param_est, arm_est
    
#%%
"""Julia code"""

# %%
