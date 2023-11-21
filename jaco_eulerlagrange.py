# %%
import sympy as sp
from IPython.display import display, clear_output, Latex
import numpy as np
from numpy import pi
from functools import reduce
from typing import List


def translation_htm(vector: sp.Matrix):
    return sp.eye(4, 3).col_insert(4, vector.row_insert(4, sp.Matrix([[1]])))


def rotation_htm(rotation_matrx: sp.Matrix):
    return rotation_matrx.row_insert(4, sp.zeros(1, 3)).col_insert(4, sp.Matrix([[0], [0], [0], [1]]))


def dht_htms(theta_list: list, d_list: list, alpha_list: list, a_list: list, time_symb: sp.Symbol) -> sp.Matrix:
    htms = []
    for i, theta in enumerate(theta_list):
        rotz = sp.rot_ccw_axis3(theta)
        rotx = sp.rot_ccw_axis1(alpha_list[i])
        d = sp.Matrix([[0], [0], [d_list[i]]])
        a = sp.Matrix([[a_list[i]], [0], [0]])
        htm = translation_htm(d) @ rotation_htm(rotz)
        htm = htm @ translation_htm(a) @ rotation_htm(rotx)
        htms.append(htm)
    return htms

def map_dhthtm_to_world(htm_list: list, axis: str = 'dh', com_coords: list | None = None) -> list | sp.Matrix:
    new_htm = []
    match axis:
        case 'dh':
            new_htm = [sp.trigsimp(
                reduce(lambda x, y: x@y, htms[:i+1], sp.eye(4))) for i, _ in enumerate(htms)]
        case 'com':
            htm_old = sp.eye(4)
            if com_coords is None:
                com_coords = [sp.zeros(3, 1) for _ in range(len(htm_list))]
            for i, htm in enumerate(htm_list):
                htm_ = htm_list[i].as_mutable()
                displacement = (htm_[:3, :3] @ com_coords[i]
                                ).row_insert(4, sp.Matrix([[0]]))
                htm_ = htm_ + sp.zeros(4, 3).col_insert(4, displacement)
                htm_ = sp.trigsimp(htm_old @ htm_)
                new_htm.append(htm_)
                htm_old = htm
        case 'eef':
            new_htm = sp.trigsimp(reduce(lambda x, y: x@y, htms, sp.eye(4)))
        case _:
            raise ValueError(
                f'{axis} is not a valid axis. Use "dh", "com" or "eef"')

    return new_htm

def get_W(q: list | sp.Matrix, htm_list: list) -> sp.Matrix:
    W_list = []
    htm_list = map_dhthtm_to_world(htm_list)
    for htm in htm_list:
        rot = htm[:3, :3]
        rx, ry, rz = rot[:, 0], rot[:, 1], rot[:, 2]
        dry, drz = ry.jacobian(q), rz.jacobian(q)
        W = sp.Matrix([[-ry.T @ drz], [rx.T @ drz], [-rx.T @ dry]])
        W_list.append(sp.trigsimp(W))
    return W_list


def steiner_theorem(inertia: sp.Matrix, displacement: list | sp.Matrix, mass: float, reverse=False) -> sp.Matrix:
    displacement = sp.Matrix(displacement)
    Inew = inertia + (-1)**(reverse) * mass * (displacement.dot(displacement)
                                               * sp.eye(3) - displacement @ displacement.T)
    return Inew


def get_M(q: list | sp.Matrix, htm_list: list, W_list: list, mass_list: list, inertia_list: list, com_coords: list) -> sp.Matrix:
    n = len(q)
    M = sp.ZeroMatrix(n, n)
    htm_list = map_dhthtm_to_world(htm_list, axis='com', com_coords=com_coords)
    for i, htm in enumerate(htm_list):
        p = sp.eye(3, 4) @ htm @ sp.Matrix([[0], [0], [0], [1]])
        Jp = p.jacobian(q)
        inertia = steiner_theorem(inertia_list[i], com_coords[i], mass_list[i])
        M += mass_list[i] * Jp.T @ Jp + W_list[i].T @ inertia @ W_list[i]
    return sp.simplify(M)

def christoffel(M, theta_list):
    n = len(theta_list)
    cijk = {}
    for k in range(n):
        for i in range(n):
            for j in range(0, i+1):
                c = sp.simplify(0.5*(M[k, j].diff(theta_list[i]) + M[k, i].diff(theta_list[j]) - M[i ,j ].diff(theta_list[k])))
                cijk[f'{i+1}{j+1}{k+1}'] = c
                cijk[f'{j+1}{i+1}{k+1}'] = c
    return cijk

def get_C(q: list | sp.Matrix, M: sp.Matrix) -> sp.Matrix:
    n = len(q)
    cijk = christoffel(M, q)
    C = sp.zeros(n, n)
    for j in range(n):
        for k in range(n):
            gamma_kj = reduce(lambda x, y:x+y, [cijk[f'{i+1}{j+1}{k+1}'] * q[i].diff() for i in range(n)], 0)
            C[k, j] = gamma_kj
    return C

def get_G(q: list | sp.Matrix, htm_list: list, mass_list: list, com_coords: list, height_axis: int = 1, gravity: float = 9.81) -> sp.Matrix:
    n = len(q)
    G = sp.ZeroMatrix(n, 1)
    htm_list = map_dhthtm_to_world(htm_list, axis='com', com_coords=com_coords)
    for i, htm in enumerate(htm_list):
        grad = ((htm @ sp.zeros(3, 1).row_insert(4,
                sp.Matrix([[1]]))).jacobian(theta_list))[height_axis, :]
        G += sp.trigsimp(mass_list[i] * gravity * grad.T)
    return sp.simplify(G)

def dynamic_model(q: list | sp.Matrix, htm_list: list, mass_list: list, inertia_list: list, com_coords: list, height_axis: int = 1, gravity: float = 9.81) -> List[sp.Matrix]:
    n = len(q)
    M = sp.ZeroMatrix(n, n)
    G = sp.ZeroMatrix(n, 1)
    W_list = get_W(q, htm_list)
    htm_list = map_dhthtm_to_world(htm_list, axis='com', com_coords=com_coords)
    
    # Get M and G
    for i, htm in enumerate(htm_list):
        p = sp.eye(3, 4) @ htm @ sp.Matrix([[0], [0], [0], [1]])
        Jp = p.jacobian(q)
        inertia = steiner_theorem(inertia_list[i], com_coords[i], mass_list[i])
        M += mass_list[i] * Jp.T @ Jp + W_list[i].T @ inertia @ W_list[i]
        grad = ((htm @ sp.zeros(3, 1).row_insert(4,
                sp.Matrix([[1]]))).jacobian(theta_list))[height_axis, :]
        G += sp.trigsimp(mass_list[i] * gravity * grad.T)
    
    # Get C
    cijk = christoffel(M, q)
    C = sp.zeros(n, n)
    for j in range(n):
        for k in range(n):
            gamma_kj = reduce(lambda x, y:x+y, [cijk[f'{i+1}{j+1}{k+1}']*q[i].diff() for i in range(n)], 0)
            C[k, j] = gamma_kj
    
    M = sp.nsimplify(sp.simplify(M))
    G = sp.nsimplify(sp.simplify(G))
    C = sp.nsimplify(sp.simplify(C))
    return M, C, G 

# Mass and Geometry properties


# Time
t = sp.Symbol('t', real=True)

# Gravity acceleration
g0 = sp.Symbol('g_0', real=True)

# Manipulated variable: force applied to mass A
tau = sp.Symbol('tau', real=True)

# Link masses
m1 = sp.Symbol('m_1', real=True)
m2 = sp.Symbol('m_2', real=True)
m3 = sp.Symbol('m_3', real=True)
m4 = sp.Symbol('m_4', real=True)
m5 = sp.Symbol('m_5', real=True)
m6 = sp.Symbol('m_6', real=True)

# Links angle
theta1 = sp.Function('theta_1', real=True)  # Shoulder
theta2 = sp.Function('theta_2', real=True)  # Upperarm
theta3 = sp.Function('theta_3', real=True)  # Forearm
theta4 = sp.Function('theta_4', real=True)  # Wrist1
theta5 = sp.Function('theta_5', real=True)  # Wrist2
theta6 = sp.Function('theta_6', real=True)  # Gripper

# Configuration variables vector
q = sp.Matrix([[theta1(t)], [theta2(t)], [theta3(t)], [theta4(t)], [theta5(t)],
               [theta6(t)]])

# Links DHT
d1 = 0.2755
d2 = 0.41
d3 = 0.2073
d4 = 0.0741
d5 = 0.0741
d6 = 0.16
e2 = 0.0098
aa = 30*pi/180
sa = np.sin(aa)
s2a = np.sin(2*aa)
d4b = d3 + (sa/s2a) * d4
d5b = (sa/s2a)*d4 + (sa/s2a)*d5
d6b = (sa/s2a)*d5 + d6
alpha1 = pi/2
alpha2 = pi
alpha3 = pi/2
alpha4 = 2*aa
alpha5 = 2*aa
alpha6 = pi

l = sp.Symbol('ell', real=True)

rotz1 = sp.rot_axis3(theta1(t))
rotx1 = sp.rot_axis1(alpha1)
dx1 = sp.Matrix([[0], [0], [d1]])
htm1 = rotz1.col_insert(4, dx1)
htm1 = htm1.row_insert(4, sp.sympify(sp.Matrix([[0, 0, 0, 1]])))
test = sp.Matrix([[m2], [m3], [m4], [m5]])
htm1 @ test

# %%
"""Spong(2020) Planar Elbow Manipulator (Cap 6.4, P. 186)"""
theta_list = [theta1(t), theta2(t)]
d_list = [0, 0, 0]
alpha_list = [0, 0, 0]
a1, a2 = sp.Symbol('l1', real=True), sp.Symbol('l2', real=True)        # link length
l1, l2 = sp.Symbol('l_{c1}', real=True), sp.Symbol('l_{c2}', real=True)# distance to COM
a_list = [a1, a2]

htms = dht_htms(theta_list, d_list, alpha_list, a_list, t)
com_coords = [sp.Matrix([[l1 - a1], [0], [0]]), sp.Matrix([[l2 - a2], [0], [0]])]
mass_list = [m1, m2]

Ixx1 = sp.Symbol('I_{xx1}', real=True)
Iyy1 = sp.Symbol('I_{yy1}', real=True)
Izz1 = sp.Symbol('I_{zz1}', real=True)
Ixy1 = sp.Symbol('I_{xy1}', real=True)
Ixz1 = sp.Symbol('I_{xz1}', real=True)
Iyz1 = sp.Symbol('I_{yz1}', real=True)
Ixx2 = sp.Symbol('I_{xx2}', real=True)
Iyy2 = sp.Symbol('I_{yy2}', real=True)
Izz2 = sp.Symbol('I_{zz2}', real=True)
Ixy2 = sp.Symbol('I_{xy2}', real=True)
Ixz2 = sp.Symbol('I_{xz2}', real=True)
Iyz2 = sp.Symbol('I_{yz2}', real=True)

# Inertias in respect to COM
inertia1 = sp.Matrix(
    [[Ixx1, -Ixy1, -Ixz1], [Ixy1, Iyy1, -Iyz1], [Ixz1, Iyz1, Izz1]])
inertia2 = sp.Matrix(
    [[Ixx2, -Ixy2, -Ixz2], [Ixy2, Iyy2, -Iyz2], [Ixz2, Iyz2, Izz2]])

# Inertias in respect to DH
inertia1 = steiner_theorem(inertia1, com_coords[0], mass_list[0], reverse=True)
inertia2 = steiner_theorem(inertia2, com_coords[1], mass_list[1], reverse=True)

W_list = get_W(theta_list, htms)
inertia_list = [inertia1, inertia2]

M, C, G = dynamic_model(theta_list, htms, mass_list, inertia_list, com_coords, height_axis=1, gravity=sp.Symbol('g_0'))
display(Latex('Mass Matrix $M(q)$:'), M)
display(Latex('Coriolis Matrix $C(q, \dot{q}):$'), C)
display(Latex('Gravity vector $G(q)$:'), G)

# %%
