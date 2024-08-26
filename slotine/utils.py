import numpy as np
import sys

def quat_to_rot(q):
    """Converts a quaternion to a rotation matrix"""
    w, x, y, z = q.ravel()
    return np.array([[2*x**2 + 2*w**2 - 1, 2*(x*y - z*w), 2*(x*z + y*w)],
                     [2*(x*y + w*z), 2*y**2 + 2*w**2 - 1, 2*(y*z - w*x)],
                     [2*(x*z - y*w), 2*(y*z + w*x), 2*z**2 + 2*w**2 - 1]])

def skew(q):
    """Maps a vector to a skew-symmetric matrix"""
    q = q.ravel()
    return np.array([[0, -q[2], q[1]],
                     [q[2], 0, -q[0]],
                     [-q[1], q[0], 0]])

def Pa(R):
    """Returns the antisymmetric part of a matrix"""
    return 0.5 * (R - R.T)

def p_norm(x, p, tol=1e-10):
    """The name is misleading; this does not compute the p-norm, but rather (p-norm)^(p) * 1/p"""
    x = x.ravel()
    return (1/p) * np.sum(np.maximum(np.abs(x), tol * np.ones(len(x))) ** p)

def p_grad(x, p, tol=1e-10):
    """Computes the gradient of the l-p norm"""
    x = x.ravel()
    fl_abs = np.maximum(np.abs(x), tol * np.ones(len(x)))
    return (fl_abs ** (p - 2) * (x + tol * np.sign(x))).reshape(-1, 1)

def pnorm_divergence(y, x, p, tol=1e-10):
    return p_norm(y, p, tol) - p_norm(x, p, tol) - np.dot((y - x).T, p_grad(x, p, tol))

def pnorm_hessian(x, p, tol=1e-10):
    x = x.ravel()
    fl_abs = np.maximum(np.abs(x), tol * np.ones(len(x)))
    norm_diag = (p - 1) * fl_abs ** (p - 4) * (x ** 2 + tol * np.ones(len(x)))
    return np.diag(norm_diag)

# Helper function for computing regressor
def L(v):
    if v.shape != (3, 1):
        print(f'Vector shape is {v.shape}, but should be (3, 1)')
    v = v.ravel()
    return np.array([[v[0], v[1], v[2], 0, 0, 0],
                     [0, v[0], 0, v[1], v[2], 0],
                     [0, 0, v[0], 0, v[1], v[2]]])

def vee(R):
    return np.array([[R[2,1]], [R[0,2]], [R[1,0]]])

def progress_bar(i, imax):
    sys.stdout.write("\r")
    sys.stdout.write(
        "[%-20s] %d%%" % ("=" * round(20 * i / (imax - 1)), round(100 * i / (imax - 1)))
    )
    sys.stdout.flush()