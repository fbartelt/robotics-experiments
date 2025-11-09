import numpy as np
from numba import njit


def holder_mean(values, r=0.1):
    v = np.array(values) + 1e-10
    if np.any(v < 0):
        raise ValueError("All values must be non-negative")
    # Raise error if values is a list of lists or nested arrays
    if v.ndim > 1:
        raise ValueError("Input must be a 1D array or list")

    raised = list(map(lambda x: x ** (-1 / r), v))
    res = np.sum(raised) ** (-r)
    return res


def holder_mean_derivative(values, r=0.1):
    v = np.array(values) + 1e-10
    if np.any(v < 0):
        raise ValueError("All values must be non-negative")
    # Raise error if values is a list of lists or nested arrays
    if v.ndim > 1:
        raise ValueError("Input must be a 1D array or list")

    outer_der = np.array(list(map(lambda x: x ** (-1 / r), v)))
    outer_der = -r * (np.sum(outer_der) ** (-r - 1))
    inner_der = np.array(list(map(lambda x: -1 / r * (x ** (-1 / r - 1)), v)))
    return outer_der * inner_der


def _smooth_min_two_elements(x, y, r=0.1):
    if x >= 0 and y >= 0:
        return holder_mean([x, y], r)
    elif x < 0 and y < 0:
        return -1 / holder_mean([-1 / x, -1 / y], r)
    else:
        return min(x, y)


def _smooth_min_list(values, r=0.1):
    if len(values) == 0:
        raise ValueError("List of values cannot be empty")
    if isinstance(values, np.ndarray):
        if values.ndim > 1:
            raise NotImplementedError(
                "Input must be a 1D array in the current implementation"
            )
        values = values.tolist()
    if len(values) == 1:
        return values[0]
    min_value = values[0]
    for val in values[1:]:
        min_value = _smooth_min_two_elements(min_value, val, r)
    return min_value


def smooth_min(x, y=None, r=0.1):
    match (x, y):
        case (list() | np.ndarray(), None):
            return _smooth_min_list(x, r)
        case (_, None):
            return x
        case (list() | np.ndarray(), _):
            if isinstance(y, (list, np.ndarray)):
                raise NotImplementedError(
                    "Cannot perform smooth_min on two lists or arrays directly"
                )
            return _smooth_min_two_elements(_smooth_min_list(x, r), y, r)
        case (_, _):
            return _smooth_min_two_elements(x, y, r)


def smooth_max(x, y=None, r=0.1):
    match (x, y):
        case (list() | np.ndarray(), None):
            return -smooth_min(list(-np.array(x)), None, r)
        case (_, None):
            return x
        case (list() | np.ndarray(), _):
            if isinstance(y, (list, np.ndarray)):
                raise NotImplementedError(
                    "Cannot perform smooth_max on two lists or arrays directly"
                )
            return -smooth_min(list(-np.array(x)), y=-y, r=r)
        case (_, _):
            return -smooth_min(-x, -y, r)


def _smooth_argmin_two_elements(x, y, r=0.1):
    if x >= 0 and y >= 0:
        grad = holder_mean_derivative([x, y], r)
        return np.argmax(grad)
    elif x < 0 and y < 0:
        grad = holder_mean_derivative([-1 / x, -1 / y], r)
        return np.argmax(grad)
    else:
        xs = np.array([x, y])
        imin = np.argmin(xs)
        return imin
        # return xs[imin]


def _smooth_argmin_list(values, r=0.1):
    if len(values) == 0:
        raise ValueError("List of values cannot be empty")
    if isinstance(values, np.ndarray):
        if values.ndim > 1:
            raise NotImplementedError(
                "Input must be a 1D array in the current implementation"
            )
        values = values.tolist()
    if len(values) == 1:
        return 0
    min_index = 0
    min_value = values[0]
    for i, val in enumerate(values[1:]):
        min_index_ = _smooth_argmin_two_elements(min_value, val, r)
        if min_index_ == 1:
            min_value = val
        min_index = min_index_ if min_index_ == 0 else i + 1
    return min_index


def smooth_argmin(x, y=None, r=0.1):
    match (x, y):
        case (list() | np.ndarray(), None):
            return _smooth_argmin_list(x, r)
        case (_, None):
            return x
        case (list() | np.ndarray(), _):
            if isinstance(y, (list, np.ndarray)):
                raise NotImplementedError(
                    "Cannot perform smooth_argmin on two lists or arrays directly"
                )
            aux = _smooth_argmin_list(x, r)  # This will be an array
            return _smooth_argmin_list([aux, y], r)
        case (_, _):
            return _smooth_argmin_two_elements(x, y, r)


@njit
def phi(s, h=0.1, r=0.1):
    """Smooth approximation of the distance function"""
    if s < 0:
        val = 0.0
        grad = 0.0
        hess = 0.0
    else:
        val = (s**3) / (2 * (s + h))
        grad = ((s**2) * (2 * s + 3 * h)) / (2 * (s + h) ** 2)
        hess = 0.0

    return val, grad, hess


@njit
def id_phi(s, h=0.5, r=0.1):
    """Phi Identity version for testing"""
    if s < 0:
        val = 0.0
        grad = 0.0
        hess = 0.0
    else:
        val = s
        grad = 1.0
        hess = 0.0

    return val, grad, hess


# @njit
# def inner_distance(f, p, A, b, r=0.1, h=0.5):
#     """Uses short-circuit algorithm"""
#     N, m = A.shape
#     in_dists, v_grad = 0.0, np.zeros((1, m))
#     v_hessian = np.zeros((m, m))  # This nonzero only for i=j
#     for i, ai_ in enumerate(A):
#         ai = ai_.copy().reshape(-1, 1)
#         s = (b[i] - ai.T @ p).item()
#         f_val, f_grad, f_hess = f(s, h)
#         if f_val > 1e-6:
#             # in_dists.append(f_val ** (-1 / r))
#             in_dists += (f_val ** (-1 / r))
#             v_grad += (-1 / r) * (f_val ** ((-1 / r) - 1)) * f_grad * (-ai.T)
#             v_hessian += (-1 / r) * (
#                 (-1/r - 1) * (f_val ** (-1 / r - 2)) * (f_grad ** 2)
#                 + (f_val ** (-1/r - 1) * f_hess)
#             ) * (ai @ ai.T)
#         else:
#             # in_dists.append(np.inf)
#             # in_dists.append((1e-6) ** (-1 / r))
#             in_dists += 1e-6 ** (-1 / r)
#
#     # S = np.sum(in_dists)
#     S = in_dists
#     in_dist = (1 / N * S) ** (-r)
#     # -r/N * (S ** (-r - 1)) * 1/N [* d(in_dist)/dS]
#     first_chain = -r * (in_dist / S)  # * 1 / N
#     in_grad = first_chain * v_grad.reshape(1, -1)
#     in_hessian = first_chain * v_hessian + (
#         # (r(r+1)) * (S/N)**(-r-2) * 1/N * 1/N [d(v)/dS ** 2]
#         (r * (r + 1)) * (in_dist / (S**2)) * (v_grad.T @ v_grad)
#     )
#     return -in_dist, -in_grad, -in_hessian


# @njit
def signed_dist2convex(f, p, A, b, r=0.1, h=0.5, test=None):
    """Paper version"""
    N, m = A.shape
    raw_inner_distances, raw_outer_distances = [], []

    for i, ai_ in enumerate(A):
        ai = ai_.copy().reshape(-1, 1)
        s = (b[i] - ai.T @ p).item()
        s_out = -s
        f_val, f_grad, f_hess = f(s, h)
        # if f_val > 1e-6:
        raw_inner_distances.append(f_val)
        # else:
        #     raw_inner_distances.append(1e-6 ** (-1 / r))
        f_val_out, f_grad_out, f_hess_out = f(s_out, h)
        # if f_val_out > 1e-6:
        raw_outer_distances.append(f_val_out)
        # else:
        #     raw_outer_distances.append(1e-6 ** (-1 / r))

    if test == "in":
        dist = -smooth_min(raw_inner_distances, r=r)
    elif test == "out":
        dist = smooth_max(raw_outer_distances, r=r)
    else:
        dist = -smooth_min(raw_inner_distances, r=r) + smooth_max(raw_outer_distances, r=r)
    return dist

def signed_dist2nonconvex(f, p, A_list, b_list, inter_indices, r=0.1, h=0.5, test=None):
    n_polytopes = len(A_list)
    ith_distances = []
    for i, (A, b, I) in enumerate(zip(A_list, b_list, inter_indices)):
        N, m = A.shape
        d_P_i = signed_dist2convex(f, p, A, b, r=r, h=h, test="in")
        R_i = []
        for j, Iij in enumerate(I):
            if Iij is not None and j != i:
                # Get A_tilde, b_tilde by ignoring the indices in Iij
                select = np.array([k for k in range(N) if k not in Iij])
                A_tilde = A[select, :]
                b_tilde = b[select]
                d_Ptilde_ij = signed_dist2convex(f, p, A_tilde, b_tilde, r=r, h=h, test="in")
                # j-th polytope
                Aj = A_list[j]
                bj = b_list[j]
                Iji = inter_indices[j][i]
                select_j = np.array([k for k in range(Aj.shape[0]) if k not in Iji])
                A_tilde_j = Aj[select_j, :]
                b_tilde_j = bj[select_j]
                d_Ptilde_ji = signed_dist2convex(f, p, A_tilde_j, b_tilde_j, r=r, h=h, test="in")
                R_i.append(smooth_max(d_Ptilde_ij, d_Ptilde_ji, r=r))

        if len(R_i) > 0:
            d_hat_P_i = smooth_min(smooth_min(R_i, r=r), d_P_i, r=r)
        else:
            d_hat_P_i = d_P_i

        inner_dist = d_hat_P_i
        outer_dist = signed_dist2convex(f, p, A, b, r=r, h=h, test="out")
        if test == "in":
            ith_distances.append(inner_dist)
        elif test == "out":
            ith_distances.append(outer_dist)
        else:
            ith_distances.append(inner_dist + outer_dist)

    dist = smooth_min(ith_distances, r=r)
    return dist





@njit
def outter_distance(f, p, A, b, r=0.1, h=0.5):
    N, m = A.shape
    out_dist, out_grad = 0.0, np.zeros((1, m))
    out_hessian = np.zeros((m, m))

    for i, ai_ in enumerate(A):
        ai = ai_.copy().reshape(-1, 1)
        s = (ai.T @ p - b[i]).item()
        f_val, f_grad, _ = f(s, r=r, h=h)
        # out_dist.append(f_val)
        out_dist += f_val
        out_grad += f_grad * ai.T

    out_dist = (1 / N) * out_dist  # np.sum(out_dist)
    out_grad = (1 / N) * out_grad.reshape(1, -1)
    return out_dist, out_grad, out_hessian


@njit
def bulging(dist, grad_dist, hess_dist, p, pc, R, eps=1e-3, out=True):
    p_ = np.asarray(p).reshape(-1, 1)
    m = p_.shape[0]
    pc_ = np.asarray(pc).reshape(-1, 1)
    grad = np.asarray(grad_dist).reshape(1, -1)
    hess = np.asarray(hess_dist).reshape((m, m))
    rho = 0.5 * (((p_ - pc_).T @ (p_ - pc_)) - R**2).item()
    grad_rho = (p_ - pc_).T
    hess_rho = np.eye(m)
    if not out:
        rho = -rho
        grad_rho = -grad_rho
        hess_rho = -hess_rho
    beta = (eps**2 * rho**2) + (1 - 2 * eps) * (dist**2)
    # Add 1e-12 for numerical stability
    sqrt_term = np.sqrt(beta + 1e-12)
    bulge_dist = eps * rho + sqrt_term
    grad_beta = 2 * (eps**2) * rho * grad_rho + 2 * (1 - 2 * eps) * dist * grad
    hess_beta = 2 * (eps**2) * ((grad_rho.T @ grad_rho) + rho * hess_rho) + 2 * (
        1 - 2 * eps
    ) * ((grad.T @ grad) + dist * hess)

    bulge_grad = eps * grad_rho + 0.5 * (1 / sqrt_term) * grad_beta
    bghess_t1 = eps * hess_rho
    bghess_t2 = (1 / (2 * sqrt_term)) * hess_beta
    bghess_t3 = -(1 / (4 * beta**1.5)) * (grad_beta.T @ grad_beta)

    bulge_hess = bghess_t1 + bghess_t2 + bghess_t3
    return bulge_dist, bulge_grad, bulge_hess


@njit
def e_s_hat(
    p,
    A,
    b,
    kind="both",
    eps=1e-3,
    r=0.1,
    h=0.5,
    eta=1.0,
):
    out_dist, out_grad, out_hess = outter_distance(phi, p, A, b, r=r, h=h)
    in_dist, in_grad, in_hess = inner_distance(phi, p, A, b, r=r, h=h)
    # if bulge:
    #     # in_dist, in_grad, in_hess = bulging(in_dist, in_grad, in_hess, p, pc, R, eps=eps, out=False)
    #     # in_dist, in_grad, in_hess = -in_dist, -in_grad, -in_hess
    #     out_dist, out_grad, out_hess = bulging(out_dist, out_grad, out_hess, p, pc, R, eps=eps)
    #
    in_dist *= eta
    in_grad *= eta

    if kind == "out":
        return out_dist, out_grad, out_hess
    elif kind == "in":
        return in_dist, in_grad, in_hess
    else:
        return out_dist + in_dist, out_grad + in_grad, out_hess + in_hess
