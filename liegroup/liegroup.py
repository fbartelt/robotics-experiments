#%%
import numpy as np
from scipy.optimize import minimize
from scipy.linalg import logm, expm

def generic_skew(omega):
    omega = omega.ravel()
    S = np.diag(omega[:-1], 1)
    S[0, -1] = omega[-1]
    return S - S.T

def geodesic(H, Hd):
    return np.linalg.norm(logm(np.linalg.inv(Hd) @ H), 'fro') ** 2

def geodesic_diff(H, Hd, delta, wrt=0):
    """ Computes partial{D}partial{q}, where q=H if wrt=0 or q=Hd if wrt=1
    """
    curr_geodesic = geodesic(H, Hd)
    dgeo = np.zeros((1, H.size))
    for i, _ in enumerate(H.ravel()):
        row, col = divmod(i, H.shape[0])
        delta_H = np.zeros(H.shape)
        delta_H[row, col] = delta
        if wrt == 1:
            next_geodesic = geodesic(H, Hd + delta_H)
        else:
            next_geodesic = geodesic(H + delta_H, Hd)
        dgeo[0, i] = (next_geodesic - curr_geodesic) / delta
    return dgeo

def chart_Onm(H, n, m, X=None):
    if X is None:
        X = create_X(n, m)
    W = np.array([-X @ generic_skew(H[:, i]) for i, _ in enumerate(H)])
    return W.reshape(-1, n+m)

def create_X(n, m):
    return np.block([
        [np.eye(n), np.zeros((n, m))],
        [np.zeros((m, n)), -np.eye(m)]
    ])

def random_O(n, m):
    """ Generates random matrix in O(n, m) -- indefinite orthogonal group
    """
    X = create_X(n, m)
    d = X.shape[0]
    omega = np.random.rand(n+m)
    A_optimized = expm(X @ generic_skew(omega))
    # A0 = np.random.randn(d, d)

    # def objective(A_vec, X):
    #     d = X.shape[0]
    #     A = A_vec.reshape((d, d))
    #     residual = A.T @ X @ A - X
    #     return np.linalg.norm(residual, 'fro')
    
    # result = minimize(objective, A0.flatten(), args=(X,), method='BFGS', options={'disp': False, 'maxiter': 5000})
    
    # A_optimized = result.x.reshape((d, d))
    return A_optimized

# Example usage
n = 2
m = 1
delta = 1e-6
X = create_X(n, m)
H = random_O(n, m)
print(H.T @ X @ H - X)
Hd = 10*random_O(n, m)
Wq = chart_Onm(H, n, m)
Wqd = chart_Onm(Hd, n, m)
dDdq = geodesic_diff(H, Hd, delta, 0)
dDdqd = geodesic_diff(H, Hd, delta, 1)
def cond(alpha, dDdq, dDdqd, Wq, Wqd):
    return np.linalg.norm(dDdq @ Wq - (alpha * dDdqd @ Wqd))
res = minimize(cond, -1, args=(dDdq, dDdqd, Wq, Wqd,), 
               method='BFGS', options={'disp': False, 'maxiter': 5000})
alpha = res.x.item()
print(f'alpha = {alpha}')
print(f'cond = {np.linalg.norm(dDdq @ Wq - (alpha * dDdqd @ Wqd))}')

# %%
