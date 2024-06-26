# %%
""" CASE OF STUDY:2-DOFROBOT from 
C. D. Cruz-Ancona, M. A. Estrada and L. Fridman, "Barrier Function-Based 
Adaptive Lyapunov Redesign for Systems Without A Priori Bounded Perturbations," 
in IEEE Transactions on Automatic Control, vol. 67, no. 8, pp. 3851-3862, 
Aug. 2022, doi: 10.1109/TAC.2021.3107453.

Could not achieve the same results as the paper. Both plots shown have magnitude
greater than what is expected.
"""
import sys
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy.linalg import solve_continuous_are
from scipy.integrate import solve_ivp, RK45
import pickle

def rk4(f, t, x, dt, *args, **kwargs):
    k1 = f(t, x, *args, **kwargs, save_hist=True)
    k2 = f(t + dt / 2, x + dt / 2 * k1, *args, **kwargs)
    k3 = f(t + dt / 2, x + dt / 2 * k2, *args, **kwargs)
    k4 = f(t + dt, x + dt * k3, *args, **kwargs)
    return x + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

def sytem_model(q, qdot):
    l1, l2 = 0.45, 0.45
    lc1, lc2 = 0.091, 0.048
    m1, m2 = 23.902, 3.88
    g_ = 9.81
    I1, I2 = 1.266, 0.093
    m1_bar, m2_bar = 0, 2

    J11 = m1 * lc1**2 + I1 + m2 * (l1**2 + lc2**2 + 2 * l1 * lc2 * np.cos(q[1, 0])) + I2
    J12 = m2 * (lc2**2 + l1 * lc2 * np.cos(q[1, 0])) + I2
    J21 = m2 * (lc2**2 + l1 * lc2 * np.cos(q[1, 0])) + I2
    J22 = m2 * lc2**2 + I2
    delta1 = m1_bar * lc1**2 + m2_bar * (
        l1**2 + lc2**2 + 2 * l1 * lc2 * np.cos(q[1, 0])
    )
    delta2 = m2_bar * (lc2**2 + l1 * lc2 * np.cos(q[1, 0]))
    delta3 = m2_bar * lc2**2
    h = m2 * l1 * lc2 * np.sin(q[1, 0])
    g1 = m1 * lc1 * g_ * np.sin(q[0, 0]) + m2 * g_ * (
        l1 * np.sin(q[0, 0]) + lc2 * np.sin(q[0, 0] + q[1, 0])
    )
    g2 = m2 * lc2 * g_ * np.sin(q[0, 0] + q[1, 0])

    J = np.array([[float(J11), float(J12)], [float(J21), float(J22)]])
    g = np.array([g1, g2]).reshape(-1, 1)
    Delta = np.array([[float(delta1), float(delta2)], [float(delta2), float(delta3)]])
    C = np.array(
        [[-h * qdot[1, 0], -h * (qdot[0, 0] + qdot[1, 0])], [h * qdot[0, 0], 0]]
    )

    return J, Delta, C, g


def psbf(z, epsilon):
    return np.linalg.norm(z) / (epsilon - np.linalg.norm(z))


def qqdotqddot(t):
    q_d = np.array([np.sin(t), np.cos(t)]).reshape(-1, 1)
    qdot_d = np.array([np.cos(t), -np.sin(t)]).reshape(-1, 1)
    qddot_d = np.array([-np.sin(t), -np.cos(t)]).reshape(-1, 1)
    return q_d, qdot_d, qddot_d


def eta(t):
    if t < 4 * np.pi:
        return 2 * np.sin(4 * t) * 0
    elif t >= 4 * np.pi and t < 8 * np.pi:
        return 5 * np.sin(4 * t) * 0
    else:
        return 0.5 * np.sin(4 * t) * 0

def sysdiffeq(t, z, A, B, P, L, Xi, epsilon, varrho, l, n, save_hist=False):
    global hist_psbf
    q_d, qdot_d, qddot_d = qqdotqddot(t)
    z = np.array(z).reshape(-1, 1)
    x = z[0:4]
    b = z[4:7]
    rho = z[7]
    x1 = x[0:2, :]
    q = x1 + q_d
    x2 = x[2:4, :]
    qdot = x2 + qdot_d
    
    J, Delta, C, g = sytem_model(q, qdot)
    Jtilde = J + Delta
    
    G = np.linalg.inv(J)
    DeltaG = J @ np.linalg.inv(Jtilde) - np.eye(n)
    phi = np.ones((2, 1)) * eta(t)
    h = G @ (np.eye(n) + DeltaG) @ (-(C @ x2) - g + phi - (Jtilde @ qddot_d))

    w = B.T @ P @ x
    psi = -varrho * w
    B_bar = B.T @ P @ B @ G
    # w_bar = B_bar.T @ w
    w_bar = B_bar @ w
    w_norm = np.linalg.norm(w)
    # print(np.linalg.norm(x))
    # print()
    # print(w_norm)
    w_bar_norm = np.linalg.norm(w_bar)
    kappa = np.block([[1], [np.linalg.norm(x)], [x.T @ x]])
    Gamma_bar = kappa.T @ b
    psbf_active = False
    H = 0.01
    S = 1/(np.exp(-(w_norm - epsilon/2) / H)+1)
    k = S * w_bar_norm + Gamma_bar + rho / w_bar_norm + (1 - S) * psbf(w, epsilon)
    # if w_norm > (epsilon / 2):
    #     k = w_bar_norm + Gamma_bar + rho / w_bar_norm
    # else:
    #     # print('psbf')
    #     k = psbf(w, epsilon)
    #     psbf_active = True
    v = -k * w_bar / w_bar_norm
    tau = psi + v
    K = np.ones((2, 4))
    tau = -K @ x

    if save_hist:
        hist_psbf.append(psbf_active)

    xdot = A @ x + B @ (G @ (np.eye(n) + DeltaG) @ tau + h)
    bdot = L @ (kappa * w_bar_norm - Xi @ b) # conferir
    rhodot = l - rho
    zdot = np.block([[xdot], [bdot], [rhodot]])

    return zdot

def progress_bar(i, imax):
    sys.stdout.write('\r')
    sys.stdout.write("[%-20s] %d%%" % ('=' * round(20 * i / (imax - 1)), round(100 * i / (imax - 1))))
    sys.stdout.flush()
#%%
dt = 0.0001
t_max = 10
imax = int(t_max / dt)
# imax = 500

n = 2
varrho = 5.03
l = 10
epsilon = 0.05
L = np.diag([0.1, 0.1, 1])
Xi = np.diag([0.01, 0.1, 0.01])
A = np.block([[np.zeros((n, n)), np.eye(n)], [np.zeros((n, n)), np.zeros((n, n))]])
B = np.block([[np.zeros((n, n))], [np.eye(n)]])
P = solve_continuous_are(A, B, 2*np.eye(2*n), varrho * np.eye(n))

x0 = np.array([1, -1, 0, 0]).reshape(-1, 1)
b0 = np.array([0.01, 0.01, 0.01]).reshape(-1, 1)
rho0 = 2

# sol = sol = solve_ivp(sysdiffeq, [0, 10], np.block([[x0], [b0], [rho0]]).flatten(), args=(A, B, P, L, Xi, epsilon, varrho, l, n), rtol=1e-3, atol=1e-3, method='RK23')
# sol = solve_ivp(sysdiffeq, [0, 10], np.block([[x0], [b0], [rho0]]).flatten(), args=(A, B, P, L, Xi, epsilon, varrho, l, n))
#%%
dt = 1e-5
t_max = 5
imax = int(t_max / dt)

# x, b, rho = x0, b0, rho0
# hist_x = np.matrix(np.zeros((2*n,0)))
# hist_w = np.matrix(np.zeros((n,0)))
# hist_psbf = []

# for i in range(imax):
#     progress_bar(i, imax)
#     q_d, qdot_d, qddot_d = qqdotqddot(i * dt)
#     x1 = x[0:2, :]
#     q = x1 + q_d
#     x2 = x[2:4, :]
#     qdot = x2 + qdot_d

#     J, Delta, C, g = sytem_model(q, qdot)
#     Jtilde = J + Delta
    
#     G = np.linalg.inv(J)
#     DeltaG = J @ np.linalg.inv(Jtilde) - np.eye(n)
#     phi = np.ones((2, 1)) * eta(i * dt)
#     h = G @ (np.eye(n) + DeltaG) @ (-(C @ x2) - g + phi - (Jtilde @ qddot_d))

#     w = B.T @ P @ x
#     psi = -varrho * w
#     B_bar = B.T @ P @ B @ G
#     w_bar = B_bar.T @ w
#     # w_bar = B_bar @ w
#     w_norm = np.linalg.norm(w)
#     # print(np.linalg.norm(x))
#     # print()
#     # print(w_norm)
#     w_bar_norm = np.linalg.norm(w_bar)
#     kappa = np.block([[1], [np.linalg.norm(x)], [x.T @ x]])
#     Gamma_bar = kappa.T @ b
#     if w_norm > (epsilon / 2):
#         k = w_bar_norm + Gamma_bar + rho / w_bar_norm
#         hist_psbf.append(False)
#     else:
#         # print('psbf')
#         k = psbf(w, epsilon)
#         hist_psbf.append(True)
#     v = -k * w_bar / np.linalg.norm(w_bar)
#     tau = psi + v

#     xdot = A @ x + B @ (G @ (np.eye(n) + DeltaG) @ tau + h)
#     bdot = L @ (kappa * w_bar_norm - Xi @ b) # conferir
#     rhodot = l - rho

#     x = x + dt * xdot
#     b = b + dt * bdot
#     rho = rho + dt * rhodot

#     hist_x = np.block([hist_x, x])
#     hist_w = np.block([hist_w, w])

# time_vec = np.arange(0, t_max, dt)
# go.Figure(go.Scatter(x= time_vec,y=np.linalg.norm(hist_x, axis=0))).show()
# go.Figure(go.Scatter(x= time_vec,y=np.linalg.norm(hist_w, axis=0))).show()
# %%
"""RK4 MOD"""

n = 2
varrho = 5.03
l = 10
epsilon = 0.05
L = np.diag([0.1, 0.1, 1])
Xi = np.diag([0.01, 0.1, 0.01])
A = np.block([[np.zeros((n, n)), np.eye(n)], [np.zeros((n, n)), np.zeros((n, n))]])
B = np.block([[np.zeros((n, n))], [np.eye(n)]])
P = solve_continuous_are(A, B, 2*np.eye(2*n), varrho * np.eye(n))

x0 = np.array([1, -1, 0, 0]).reshape(-1, 1)
b0 = np.array([0.01, 0.01, 0.01]).reshape(-1, 1)
rho0 = 2
z0 = np.block([[x0], [b0], [rho0]])
z = z0

dt = 1e-4
t_max = 5
imax = int(t_max / dt)

x, b, rho = x0, b0, rho0
hist_x = np.matrix(np.zeros((2*n,0)))
hist_w = np.matrix(np.zeros((n,0)))
hist_b = np.matrix(np.zeros((len(b0),0)))
hist_rho = []
hist_psbf = []

for i in range(imax):
    progress_bar(i, imax)
    z = rk4(sysdiffeq, i * dt, z, dt, A, B, P, L, Xi, epsilon, varrho, l, n)
    x = z[0:4]
    b = z[4:7]
    rho = z[7]
    hist_x = np.block([hist_x, x])
    hist_b = np.block([hist_b, b])
    hist_rho.append(rho)

time_vec = np.arange(0, t_max, dt)
# with open(f'lyapcbfreprod_{dt}.pkl', 'wb') as f:
#     data = {}
#     data['hist_x'] = hist_x
#     data['hist_b'] = hist_b
#     data['hist_rho'] = hist_rho
#     data['time_vec'] = time_vec
#     data['hist_psbf'] = hist_psbf
#     data['B'] = B
#     data['P'] = P
#     data['epsilon'] = epsilon
#     data['Xi'] = Xi
#     data['L'] = L
#     data['varrho'] = varrho
#     pickle.dump(data, f)
fig = px.line(x=time_vec, y=np.linalg.norm(hist_x, axis=0).T, title="|x|")
fig.add_scatter(x=time_vec, y=np.array(hist_psbf)*np.max(hist_x), fill='tozeroy', name='psbf')
fig.show()
px.line(x=time_vec,y=np.linalg.norm(hist_b, axis=0), title='|b|').show()
hist_w_norm = np.linalg.norm(np.array([B.T @ P @ x.T for x in hist_x.T]).reshape(-1, n), axis=1)
fig=px.line(x=time_vec, y=hist_w_norm, title='||w||')
fig.add_scatter(x=time_vec, y=[epsilon]*len(time_vec), fill='tozeroy', name='<=epsilon')
# %%
with open('lyapcbfreprod_1e-05.pkl', 'rb') as f:
    data = pickle.load(f)

fig = px.line(x=data['time_vec'], y=np.linalg.norm(data['hist_x'], axis=0).T, title="|x|")
fig.add_scatter(x=data['time_vec'], y=np.array(data['hist_psbf'])*np.max(data['hist_x']), fill='tozeroy', name='psbf')
fig.show()
# %%
