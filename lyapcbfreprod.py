#%%
""" CASE OF STUDY:2-DOFROBOT from 
C. D. Cruz-Ancona, M. A. Estrada and L. Fridman, "Barrier Function-Based 
Adaptive Lyapunov Redesign for Systems Without A Priori Bounded Perturbations," 
in IEEE Transactions on Automatic Control, vol. 67, no. 8, pp. 3851-3862, 
Aug. 2022, doi: 10.1109/TAC.2021.3107453.
"""
import numpy as np
from scipy.linalg import solve_continuous_are

def sytem_model(q, qdot):
    l1, l2 = 0.45, 0.45
    lc1, lc2 = 0.091, 0.048
    m1, m2 = 23.902, 3.88
    g_ = 9.81
    I1, I2 = 1.266, 0.093
    m1_bar, m2_bar = 0, 2

    J11 = m1*lc1**2 + I1 + m2*(l1**2 + lc2**2 + 2*l1*lc2*np.cos(q[1])) + I2
    J12 = m2*(lc2**2 + l1*lc2*np.cos(q[1])) + I2
    J21 = m2*(lc2**2 + l1*lc2*np.cos(q[1])) + I2
    J22 = m2*lc2**2 + I2
    delta1 = m1_bar*lc1**2 + m2_bar*(l1**2 + lc2**2 + 2*l1*lc2*np.cos(q[1]))
    delta2 = m2_bar*(lc2**2 + l1*lc2*np.cos(q[1]))
    delta3 = m2_bar*lc2**2
    h = m2*l1*lc2*np.sin(q[1])
    g1 = m1*lc1*g_*np.sin(q[0]) + m2*g_*(l1*np.sin(q[0]) + lc2*np.sin(q[0] + q[1]))
    g2 = m2*lc2*g_*np.sin(q[0] + q[1])

    J = np.array([[J11, J12], [J21, J22]])
    g = np.array([g1, g2])
    Delta = np.array([[delta1, delta2], [delta2, delta3]])
    C = np.array([[-h*qdot[1], -h*(qdot[0] + qdot[1])], [h*qdot[0], 0]])

    return J, Delta, C, g

n = 2
varrho = 5.03

x1 = q - q_d
x2 = qdot - qdot_d
x = np.block([[x1], [x2]])
J, Delta, C, g = sytem_model(q, qdot)
Jtilde = J + Delta
A = np.block([[np.zeros((n, n)), np.eye(n)], [np.zeros((n, n)), np.zeros((n, n))]])
B = np.block([[np.zeros((n, n))], [np.eye(n)]])
G = np.linalg.inv(J)
DeltaG = (J @ np.linalg.inv(Jtilde) - np.eye(n))
h = G @ (np.eye(n) + DeltaG) @ (-C @ x2 - g + phi - Jtilde @ qddot_d)

P = solve_continuous_are(A, B, np.eye(n), varrho)
psi = -varrho @ B.T @ P @ x
tau = psi + v
xdot = A @ x + B @ (G @ (np.eye(2*n) + DeltaG) @ tau + h)


# %%
