# %%
import numpy as np
import plotly.graph_objects as go
import plotly.colors as pc
import cyipopt

# from distances import signed_dist2convex, phi, smooth_min
from smoothfunctions import signedDist2Convex, smoothMinListWithGradient, smoothMinList
from scipy.optimize import minimize
from typing import List, Tuple, Optional, Callable
from polygon import (
    Polytope,
    add_polygon,
    create_level_sets,
    NonConvexPolygon,
    generate_random_polyhedron,
    generate_random_polyhedron_set,
)
from polyhedron import add_polyhedron


def add_path(
    fig, path_hist, num_paths, base_color="#000000", q0_symbol="square", qd_symbol="x"
):
    base_color = "#000000"
    base_color_rgb = pc.convert_colors_to_same_type(base_color, colortype="rgb")[0][0]
    base_color_rgba = (
        base_color_rgb.replace(" ", "").replace(")", "").replace("rgb", "rgba")
    )

    # Adds num_paths paths from path_hist to the figure
    # each path is colored with a gradient from base_color 0.1 opacity to base_color 1.0 opacity
    # q0 and qd are marked with different symbols
    # paths2add = path_hist[:: max(1, len(path_hist) // num_paths)]
    idxs2add = np.round(
        np.linspace(0, len(path_hist) - 1, num=min(num_paths, len(path_hist)))
    ).astype(int)
    paths2add = np.array(path_hist)[idxs2add]
    print(f"Adding {len(paths2add)} paths to the figure.")
    for i, path in enumerate(paths2add):
        alpha = 0.1 + (0.5 * i) / (len(paths2add) - 1) if len(paths2add) > 1 else 1.0
        alpha = alpha if i < len(paths2add) - 1 else 1.0
        color = base_color_rgba + f", {alpha})"
        fig.add_trace(
            go.Scatter(
                x=path[0, :],
                y=path[1, :],
                mode="lines+markers",
                line=dict(color=color, width=2),
                marker=dict(size=6, color=color),
                name=f"Path {i+1}",
                showlegend=False,
            )
        )
        # Mark q0
        fig.add_trace(
            go.Scatter(
                x=[path[0, 0]],
                y=[path[1, 0]],
                mode="markers",
                marker=dict(symbol=q0_symbol, size=10, color=color),
                name="Start",
                showlegend=(i == len(paths2add) - 1),
            )
        )
        # Mark qd
        fig.add_trace(
            go.Scatter(
                x=[path[0, -1]],
                y=[path[1, -1]],
                mode="markers",
                marker=dict(symbol=qd_symbol, size=10, color=color),
                name="Goal",
                showlegend=(i == len(paths2add) - 1),
            )
        )


def add_path3d(
    fig, path_hist, num_paths, base_color="#000000", q0_symbol="square", qd_symbol="x"
):
    base_color = "#000000"
    base_color_rgb = pc.convert_colors_to_same_type(base_color, colortype="rgb")[0][0]
    base_color_rgba = (
        base_color_rgb.replace(" ", "").replace(")", "").replace("rgb", "rgba")
    )

    # Adds num_paths paths from path_hist to the figure
    # each path is colored with a gradient from base_color 0.1 opacity to base_color 1.0 opacity
    # q0 and qd are marked with different symbols
    # paths2add = path_hist[:: max(1, len(path_hist) // num_paths)]
    idxs2add = np.round(
        np.linspace(0, len(path_hist) - 1, num=min(num_paths, len(path_hist)))
    ).astype(int)
    paths2add = np.array(path_hist)[idxs2add]
    print(f"Adding {len(paths2add)} paths to the figure.")
    for i, path in enumerate(paths2add):
        alpha = 0.1 + (0.5 * i) / (len(paths2add) - 1) if len(paths2add) > 1 else 1.0
        alpha = alpha if i < len(paths2add) - 1 else 1.0
        color = base_color_rgba + f", {alpha})"
        marker_size = 4
        fig.add_trace(
            go.Scatter3d(
                x=path[0, :],
                y=path[1, :],
                z=path[2, :],
                mode="lines+markers",
                line=dict(color=color, width=4),
                marker=dict(size=2, color=color),
                name=f"Path {i+1}",
                showlegend=False,
            )
        )
        # Mark q0
        fig.add_trace(
            go.Scatter3d(
                x=[path[0, 0]],
                y=[path[1, 0]],
                z=[path[2, 0]],
                mode="markers",
                marker=dict(symbol=q0_symbol, size=marker_size, color=color),
                name="Start",
                showlegend=(i == len(paths2add) - 1),
            )
        )
        # Mark qd
        fig.add_trace(
            go.Scatter3d(
                x=[path[0, -1]],
                y=[path[1, -1]],
                z=[path[2, -1]],
                mode="markers",
                marker=dict(symbol=qd_symbol, size=marker_size, color=color),
                name="Goal",
                showlegend=(i == len(paths2add) - 1),
            )
        )


def deform_path(
    init_path,
    obstacles,
    kind="in",
    h=0.01,
    r=0.1,
    min_path=False,
    zeta=1.0,
    alpha=1.0,
):
    if zeta == 0:
        min_path = False
    path = init_path.copy()
    N = path.shape[1]  # number of points in the path
    m = path.shape[0]  # dimension of the space
    n_obstacles = len(obstacles)
    dists, grads = np.zeros((N,)), np.zeros((N, m))
    test = kind if kind in ["in", "out"] else ""
    for j, p_ in enumerate(init_path.T):
        p = p_.copy().reshape(-1, 1)
        dists_, grads_ = np.zeros((n_obstacles,)), np.zeros((n_obstacles, m))
        for i, obstacle in enumerate(obstacles):
            A, b = obstacle.A, obstacle.b
            # Compute each d_S and gradient (1 x m)
            dist_, grad_ = signedDist2Convex(
                p, A, b.reshape(-1, 1), r=r, eps=h, test=test
            )
            # dist_, grad_ = signed_dist2convex(
            if np.any(np.isnan(grad_)):
                print(f"NaN in grad: dist={dist_}, grad={grad_})")
                raise ValueError("NaN in grad from signed distance.")
            #     phi, p, A, b, r=r, h=h, test=kind, compute_gradient=True
            # )
            dists_[i] = np.round(dist_, 6)
            if np.round(dist_, 4) >= 0:
                grads_[i, :] = grad_.ravel()
            else:
                grads_[i, :] = grad_.ravel()
        # Compute D_O and gradient (1 x n_obstacles)
        # dist, grad = smooth_min(dists_, r=r, compute_gradient=True)
        dist, grad = smoothMinListWithGradient(dists_, r=r)
        if np.any(np.isnan(grad)):
            print(f"NaN in grad: dist={dist}, grad={grad})")
            raise ValueError("NaN in grad from smooth min.")

        grad_sat = 1 / (1 + np.exp(alpha * dist))
        dist = (-1 / alpha) * np.log(
            0.5 * (1 + np.exp(-alpha * dist))
        )  # Smooth saturation
        grad_full = grad_sat * grad @ grads_  # (1 x m)
        # idx_min = np.argmin(dists_)
        # grad_full = grads_[idx_min, :].reshape(1, -1)
        dists[j] = dist
        grads[j, :] = grad_full.ravel()

    for j in range(path.shape[1] - 2):
        k = j + 1  # Do not change initial and final point
        # const_obs = [b - A @ path[:, j].reshape(-1, 1) for A, b in obstacles]
        # err = np.max(const_obs)
        # coeff = np.abs(dists[j])  # np.sign(dists[j]) * np.sqrt(np.abs(dists[j]))
        coeff = 1.0
        if dists[k] > 0:
            coeff = 1.0
            coeff = np.sqrt(np.abs(dists[k]))
        else:
            coeff = np.sqrt(np.abs(dists[k]))
        path[:, k] += grads[k] * coeff

    if min_path:
        for j, point in enumerate(path.T[1:-1]):
            k = j + 1
            prev_grad = path[:, k - 1].ravel() - point.ravel()
            next_grad = path[:, k + 1].ravel() - point.ravel()
            path[:, k] = point + zeta * (prev_grad + next_grad)

    return path, dists, grads


class PathDeformationProblem:
    def __init__(
        self,
        init_path,
        obstacles,
        kind=None,
        h=0.01,
        r=0.1,
        alpha=1.0,
        zeta=1.0,
        min_path=True,
    ):
        self.init_path = init_path
        self.obstacles = obstacles
        self.n_obstacles = len(obstacles)
        self.kind = kind
        self.test = kind if kind in ["in", "out"] else ""
        self.h = h
        self.r = r
        self.alpha = alpha
        self.zeta = zeta
        self.min_path = min_path or (zeta > 0)

        self.m, self.N = init_path.shape
        self.n_vars = self.m * (self.N - 2)
        self.n_constraints = self.N - 2
        self.path_history = []
        self.x_history = []

    def unpack_path(self, x):
        path = self.init_path.copy()
        for k in range(self.N - 2):
            path[:, k + 1] = x[k * self.m : (k + 1) * self.m]
        return path

    def objective(self, x):
        self._last_x = x.copy()
        path = self.unpack_path(x)
        # Collision avoidance cost + path smoothness cost
        cost = 0.0

        for j, p_ in enumerate(path.T):
            if 0 < j < self.N - 1:
                p = p_.copy().reshape(-1, 1)
                dists_ = np.zeros((self.n_obstacles,))
                for i, obstacle in enumerate(obstacles):
                    A, b = obstacle.A, obstacle.b
                    # Compute each d_S and gradient (1 x m)
                    dist_, grad_ = signedDist2Convex(
                        p, A, b.reshape(-1, 1), r=self.r, eps=self.h, test=self.test
                    )
                    # dists_[i] = np.round(dist_, 6)
                    dists_[i] = dist_
                # Compute D_O and gradient (1 x n_obstacles)
                dist = smoothMinList(dists_, r=r)

                dist = (-1 / self.alpha) * np.log(
                    0.5 * (1 + np.exp(-self.alpha * dist))
                )  # Smooth saturation
                cost += -dist

                if self.min_path:
                    nxt = path[:, j + 1]
                    cost += self.zeta / 2 * np.linalg.norm(nxt - p.ravel()) ** 2

        return cost

    def gradient(self, x):
        path = self.unpack_path(x)
        grad = np.zeros_like(x)

        for j, p_ in enumerate(path.T):
            p = p_.copy().reshape(-1, 1)
            dists_, grads_ = np.zeros((self.n_obstacles,)), np.zeros(
                (self.n_obstacles, self.m)
            )
            for i, obstacle in enumerate(obstacles):
                A, b = obstacle.A, obstacle.b
                # Compute each d_S and gradient (1 x m)
                dist_, grad_ = signedDist2Convex(
                    p, A, b.reshape(-1, 1), r=self.r, eps=self.h, test=self.test
                )
                dists_[i] = dist_
                grads_[i, :] = grad_.ravel()
            # Compute D_O and gradient (1 x n_obstacles)
            dist, grad_min = smoothMinListWithGradient(dists_, r=self.r)

            grad_sat = 1 / (1 + np.exp(self.alpha * dist))
            dist = (-1 / self.alpha) * np.log(
                0.5 * (1 + np.exp(-self.alpha * dist))
            )  # Smooth saturation
            grad_full = -grad_sat * grad_min @ grads_  # (1 x m)

            if 0 < j < self.N - 1:
                idx = (j - 1) * self.m
                grad[idx : idx + self.m] = -grad_full.ravel()
                if self.min_path:
                    # Gradient of segment lengths around point j
                    segment_grad = np.zeros((self.m,))
                    if j > 0:
                        segment_grad += path[:, j] - path[:, j - 1]
                    elif j < self.N - 1:
                        segment_grad += path[:, j + 1] - path[:, j]

                    # path[:, j] += self.zeta * segment_grad
                    grad[idx : idx + self.m] += self.zeta * segment_grad.ravel()

        return grad

    def intermediate(
        self,
        alg_mod,
        iter_count,
        obj_value,
        inf_pr,
        inf_du,
        mu,
        d_norm,
        regularization_size,
        alpha_du,
        alpha_pr,
        ls_trials,
    ):
        """
        Official IPOPT iteration callback.
        Called once per iteration.
        """

        # IPOPT guarantees that the current iterate is the last x
        x = self._last_x

        self.x_history.append(x.copy())
        self.path_history.append(self.unpack_path(x))

        return True


def deform_path_ipopt(
    init_path,
    obstacles,
    max_iter=200,
    **kwargs,
):
    problem = OptimalPathProblem(
        init_path=init_path,
        obstacles=obstacles,
        **kwargs,
    )

    print(f"{problem.n} variables, {problem.m} constraints. N x d = {problem.N} x {problem.d}")

    x0 = init_path.flatten()

    x_L = np.full(problem.n, -np.inf)
    x_U = np.full(problem.n, np.inf)

    # Constraint bounds
    c_L = np.zeros(problem.m)
    c_U = np.zeros(problem.m)
    # First 2*d are equalities
    c_U[: 2 * problem.d] = 0.0
    # Last N-1 are upper bounds only (||p_{i+1} - p_i|| ≤ δ)
    print(f"Delta: {problem.delta}. Shape of c_U: {c_U.shape}")
    c_U[2 * problem.d :] = delta

    nlp = cyipopt.Problem(
        n=problem.n,
        m=problem.m,
        problem_obj=problem,
        lb=x_L,
        ub=x_U,
        cl=c_L,
        cu=c_U,
    )

    options = {
        # Force first-order method (no second derivatives)
        # "hessian_approximation": "exact",
        "derivative_test": "first-order",
        "derivative_test_print_all": "yes",
        # "hessian_approximation": "limited-memory",
        # 'gradient_approximation': 'finite-difference-values',
        # 'jacobian_approximation': 'finite-difference-values',
        # Configure L-BFGS parameters
        # 'limited_memory_update_type': 'bfgs',  # Standard BFGS update
        # 'limited_memory_max_history': 10,      # History size (10-50 is typical)
        # Disable second-order features
        # 'mehrotra_algorithm': 'no',           # Disable second-order correction
        # 'fast_step_computation': 'no',         # Disable advanced step calc
        "alpha_for_y": "primal",  # More aggressive step sizing
        "recalc_y": "no",  # Reduces computational overhead
        "mu_strategy": "adaptive",
        # 'linear_solver': 'mumps',
        # Adjust convergence criteria for first-order method
        "tol": 1e-1,  # Relax tolerance (default 1e-8)
        "max_iter": max_iter,  # Increase iteration limit
        "acceptable_iter": 10,  # Stop after 10 "good enough" iters
        # Output control
        "print_level": 5,
        "print_frequency_iter": 10,
    }

    for key in options.keys():
        nlp.add_option(key, options[key])

    x_opt, info = nlp.solve(x0)

    path_opt = problem.unpack_path(x_opt)

    return path_opt, problem.path_history, info


# %%
max_polygons = 10  # 7 (1111), 10 (1337), 5 (1001)
max_vertices = 20
bounding_box = (-20.0, -20, 20, 20)
# Distance between vertices will be at least 2*first element, and at most
# 2*second element of radius_limits:
radius_limits = (2, 6)
q0 = np.array([-3.0, -12]).reshape(-1, 1)
q0 = np.array([-8.5, -16]).reshape(-1, 1)  # 1001
q0 = np.array([-17.0, -17]).reshape(-1, 1)  # 1337, 1111
qd = np.array([18, 15]).reshape(-1, 1)
# qd = np.array([11.0, 15]).reshape(-1, 1)
n_points = 100
h = 0.01
r = 0.1
zeta = 0.5
alpha = np.log(2) / 0.4
min_path = True
max_attempts = 500
seed = 1337  # 1001, 1337, 1111
min_area = None
radius = None
num_vertices = None

obstacles = Polytope.random_set(
    n_polytopes=max_polygons,
    intersect_polytopes=False,
    q0=q0,
    qd=qd,
    max_vertices=max_vertices,
    radius_lim=radius_limits,
    bbox=bounding_box,
    seed=seed,
    min_area=min_area,
    max_attempts=max_attempts,
    radius=radius,
    num_vertices=num_vertices,
)

lambda_ = np.linspace(0, 1, n_points)
init_path = (1 - lambda_) * q0 + lambda_ * qd
path = init_path.copy()
dists = [-100]
iter_ = 0
path_hist = [init_path.copy()]
max_iters = 1200
kind = "in"
kind = "out"
kind = None

# bounding_box = (-20.0, -20, 20, 20.11) # 1337 plotting related
# fig = go.Figure()
print(f"Creating level sets for {len(obstacles)} obstacles.")
fig = create_level_sets(
    obstacles,
    r=r,
    h=h,
    kind="both",
    bbox=bounding_box,
    n_points=20,
    n_contours=40,
    add_reference=False,
    test=None,
    rescale=True,
)
print("Level sets created.")
# for obstacle in obstacles:
#     add_polygon(fig, obstacle.A, obstacle.b, add_reference=False)

# obstacles = [obstacles[0]]

while np.any(np.array(dists) < 0.0):
    if iter_ >= max_iters:
        print(f"reached max iterations: {max_iters}")
        break

    path, dist, grad = deform_path(
        path, obstacles, kind=kind, h=h, r=r, min_path=min_path, zeta=zeta, alpha=alpha
    )
    path_hist.append(path.copy())
    dists = dist
    if iter_ % 10 == 0:
        print(f"iteration {iter_}: min dist = {np.min(dists)}")

    iter_ += 1

print(f"deformation completed in {iter_} iterations with min dist = {np.min(dists)}")
add_path(fig, path_hist, num_paths=6, base_color="black")
fig.update_layout(width=1200, height=800)
fig.show()

# fig.write_image(f"path_seed_{seed}_maxpoly_{max_polygons}.pdf")


# [jump]

# %%
n_polyhedra = 12
max_vertices = 15
bounding_box = (-20.0, -20, -20, 20, 20, 20)
# Distance between vertices will be at least 2*first element, and at most
# 2*second element of radius_limits:
radius_limits = (2, 10)
q0 = np.array([-1.0, -15, 10.0]).reshape(-1, 1)
qd = np.array([18.0, 18, -18.0]).reshape(-1, 1)
# qd = np.array([11.0, 15]).reshape(-1, 1)
n_points = 100
h = 0.01
r = 0.1
zeta = 0.5 / 2
alpha = np.log(2) / 0.2
min_path = True
max_attempts = 500
seed = 1001  # 1001, 69 cool, 42 NICE post mods, 100 is cool
min_volume = 4 / 3 * np.pi * (3**3)  # at least radius 2
radius = None
num_vertices = None

obstacles = Polytope.random_set_polyhedra(
    n_polyhedra=n_polyhedra,
    intersect_polyhedra=False,
    q0=q0,
    qd=qd,
    max_vertices=max_vertices,
    radius_lim=radius_limits,
    bbox=bounding_box,
    seed=seed,
    dim=3,
    min_volume=min_volume,
    max_attempts=max_attempts,
    radius=radius,
    num_vertices=num_vertices,
)

# Test with simple cube
# obstacles = [
#     Polytope(
#         A=np.array([
#             [1, 0, 0],
#             [-1, 0, 0],
#             [0, 1, 0],
#             [0, -1, 0],
#             [0, 0, 1],
#             [0, 0, -1],
#         ]),
#         b=np.array([5, 5, 5, 5, 5, 5]).reshape(-1, 1),
#     )
# ]

lambda_ = np.linspace(0, 1, n_points)
init_path = (1 - lambda_) * q0 + lambda_ * qd
path = init_path.copy()
dists = [-100]
iter_ = 0
path_hist = [init_path.copy()]
max_iters = 200
kind = "in"
kind = "out"
kind = None

# bounding_box = (-20.0, -20, 20, 20.11) # 1337 plotting related
fig = go.Figure()
for obstacle in obstacles:
    add_polyhedron(fig, obstacle.A, obstacle.b, add_reference=False)

# obstacles = [obstacles[0]]
while np.any(np.array(dists) < 0.0):
    if iter_ >= max_iters:
        print(f"reached max iterations: {max_iters}")
        break

    path, dist, grad = deform_path(
        path, obstacles, kind=kind, h=h, r=r, min_path=min_path, zeta=zeta, alpha=alpha
    )
    path_hist.append(path.copy())
    dists = dist
    if iter_ % 10 == 0:
        print(f"iteration {iter_}: min dist = {np.min(dists)}")

    iter_ += 1

print(f"deformation completed in {iter_} iterations with min dist = {np.min(dists)}")
add_path3d(fig, path_hist, num_paths=6, base_color="black")
fig.update_layout(width=1200, height=800)
fig.show()

# %%
# Scipy optimization approach
# optimized_path, history = deform_path_optimize(
#     path,
#     obstacles,
#     kind=kind,
#     h=h,
#     r=r,
#     min_path=min_path,
#     zeta=zeta,
#     alpha=alpha,
#     method=None,
#     options = {"maxiter": 2500, "ftol": 1e-3, "gtol": 1e-3, "disp": True}
# )
# print(history["message"])
# add_path(fig, history["paths"], num_paths=6, base_color="black")
# fig.update_layout(width=1200, height=800)
# fig.show()
#
