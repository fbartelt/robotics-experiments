import numpy as np
import plotly.graph_objects as go
import plotly.colors as pc
import cyipopt

# from distances import signed_dist2convex, phi, smooth_min
from smoothfunctions import signedDist2Convex, smoothMinListWithGradient, smoothMinList, phi
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
    for i, path_ in enumerate(paths2add):
        path = path_.T  # Transpose to (2 x N)
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


class OptimalPathProblem:
    """
    Optimal path deformation problem using IPOPT.
    min sum_{i=1}^{N} smoothsat(d(O, p_i)) + sum_{i=1}^{N-1} zeta/2 * ||p_{i+1} - p_{i}||^2
    s.t. p_1 = q0
         p_N = qd

    where smoothsat is a smooth saturation function, d(O, p_i) is the signed distance from point p_i to the set of obstacles O.

    """

    def __init__(
        self,
        obstacles,
        init_path,
        zeta,
        alpha,
        min_path,
        r,
        h,
        kind=None,
        delta=0,
    ):
        self.init_path = init_path
        self.N = init_path.shape[0]  # Number of path points
        self.n = init_path.shape[1]  # Dimension of the space
        self.q0 = init_path[0].reshape(-1, 1)
        self.qd = init_path[-1].reshape(-1, 1)
        self.obstacles = obstacles
        self.n_obstacles = len(obstacles)
        self.n_variables = self.N * self.n
        # p1 = q0, pN = qd, ||p_{i+1} - p_{i}|| ≤ delta for i=1..N-1
        self.m_constraints = 2 * self.n  # + (self.N - 1)
        # Distance-related parameters
        self.h = h
        self.r = r
        self.kind = kind
        self.test = kind if kind in ["in", "out"] else ""
        # Functionals parameters
        self.alpha = alpha
        self.zeta = zeta
        self.min_path = min_path
        self.delta = delta  # Currently unused
        # History of paths
        self.path_history = []
        self._last_x = None  # For IPOPT intermediate callback

    def pack_path(self, path):
        return path.flatten()

    def unpack_path(self, x):
        return x.reshape((self.N, self.n))

    def objective(self, x):
        self._last_x = x.copy()
        path = self.unpack_path(x)
        total_cost = 0.0

        # Obstacle avoidance cost
        for i in range(self.N):
            p_i = path[i].reshape(-1, 1)
            dists = np.zeros(self.n_obstacles)
            for j, obs in enumerate(self.obstacles):
                dist_ij, _ = signedDist2Convex(
                    p_i,
                    obs.A,
                    obs.b.reshape(-1, 1),
                    r=self.r,
                    eps=self.h,
                    test=self.test,
                )
                dists[j] = dist_ij
            smooth_min_dist = smoothMinList(dists, self.r)
            # Smooth saturation (avoids cheating by circunventing the map)
            exponent = np.clip(-self.alpha * smooth_min_dist, -100, 100)
            sat_dist = (-1 / self.alpha) * np.log(
                0.5 * (1 + np.exp(exponent))
            )  # Smooth saturation

            total_cost += -sat_dist  # Minimize negative sat_dist to maximize distance
            # Path length cost
            # if self.min_path and i < self.N - 1:
            #     p_next = path[:, i + 1].reshape(-1, 1)
            #     total_cost += (self.zeta / 2) * np.linalg.norm(p_next - p_i) ** 2

        return total_cost

    def gradient(self, x):
        path = self.unpack_path(x)
        grad = np.zeros_like(path)  # N x n

        for i in range(self.N):
            p_i = path[i].reshape(-1, 1)
            dists = np.zeros(self.n_obstacles)
            grads = np.zeros((self.n, self.n_obstacles))

            for j, obs in enumerate(self.obstacles):
                dist_ij, grad_ij = signedDist2Convex(
                    p_i,
                    obs.A,
                    obs.b.reshape(-1, 1),
                    r=self.r,
                    eps=self.h,
                    test=self.test,
                )
                dists[j] = dist_ij
                grads[:, j] = grad_ij.flatten()

            # Smooth min grad is shaped (1 x n_obstacles)
            smooth_min_dist, smooth_min_grad = smoothMinListWithGradient(dists, self.r)
            # Smooth saturation gradient
            exponent = np.clip(self.alpha * smooth_min_dist, -100, 100)
            grad_sat = 1 / (1 + np.exp(exponent))
            # sat_dist = (-1 / self.alpha) * np.log(
            #     0.5 * (1 + np.exp(-self.alpha * smooth_min_dist))
            # )  # Smooth saturation

            grad_full = grad_sat * (grads @ smooth_min_grad.reshape(-1, 1))  # n x 1
            # Point 29 has negative distance outside obstacle in test case ??
            if i == 29:
                print(f"p_i: {p_i.ravel()}")
                aux = [ phi(b - (a @ p_i), eps=self.h) for a, b in zip(self.obstacles[0].A, self.obstacles[0].b)]
                print(f"DEBUG: point {i}, aux: {aux}")
                print(
                    f"[DEBUG]: point {i}, dists: {dists}, smooth_min_dist: {smooth_min_dist}, grad_sat: {grad_sat}, smooth_min_grad: {smooth_min_grad.flatten()}, grads: {grads.flatten()}, grad_full: {grad_full.flatten()}"
                )
            grad[i] += -grad_full.flatten()  # Minimize negative sat_dist
            # Path length gradient
            # if self.min_path and i < self.N - 1:
            #     p_next = path[:, i + 1].reshape(-1, 1)
            #     grad[:, i] += -self.zeta * (p_next - p_i).flatten()
            #     grad[:, i + 1] += self.zeta * (p_next - p_i).flatten()

        return grad.flatten()

    def constraints(self, x):
        path = self.unpack_path(x)
        # print(f"DEBUG: path shape: {path.shape}")
        # print(f"DEBUG: path[:,0]: {path[0]}, q0: {self.q0.flatten()}")
        # print(f"DEBUG: path[:,-1]: {path[-1]}, qd: {self.qd.flatten()}")
        constraints = np.zeros(self.m_constraints)
        # Equality constraints for endpoints
        # Equality constraints: p1 = q0, pN = qd (2n constraints)
        constraints[: self.n] = path[0] - self.q0.flatten()
        constraints[self.n : 2 * self.n] = path[-1] - self.qd.flatten()
        # constraints_.extend((path[:, 0].ravel() - self.q0.ravel()).tolist())  # p_1 = q0
        # constraints_.extend(
        #     (path[:, -1].ravel() - self.qd.ravel()).tolist()
        # )  # p_N = qd
        # Inequality constraints: ||p_{i+1} - p_i||^2 ≤ ζ^2 (N-1 constraints)
        # offset = 2 * self.n
        # for i in range(self.N - 1):
        #     p_i = path[:, i]
        #     p_next = path[:, i + 1]
        #     diff = p_next - p_i
        #     constraints[offset + i] = np.dot(diff, diff) - self.delta**2
        # print(f"DEBUG: constraints: {constraints}")
        return constraints

    def jacobian(self, x):
        # jac = np.zeros((self.m_constraints, self.n_variables))

        # First 2n constraints are the fixed endpoints (eq. constraints)
        # First n constraints: p1 = q0
        # for i in range(self.n):
        #     jac[i, i] = 1.0
        #
        # # Second n constraints: pN = qd
        # for i in range(self.n):
        #     jac[self.n + i, (self.N - 1) * self.n + i] = 1.0
        #
        # return jac.flatten()
        non_zeros = []

        # 1. Equality constraints (2n entries, all 1.0)
        non_zeros.extend([1.0] * (2 * self.n))

        # 2. Inequality constraints (2n non-zeros per constraint)
        # offset = 2 * self.n  # Start index for inequality constraints
        # for i in range(self.N - 1):
        #     p_i = path[:, i]
        #     p_next = path[:, i + 1]
        #     diff = p_next - p_i
        #
        #     # ∂c_i/∂p_i = -2*diff (all n dimensions)
        #     for dim in range(self.n):
        #         non_zeros.append(-2.0 * diff[dim])
        #
        #     # ∂c_i/∂p_{i+1} = 2*diff (all n dimensions)
        #     for dim in range(self.n):
        #         non_zeros.append(2.0 * diff[dim])
        # jac = np.zeros((self.m_constraints, self.n_variables))
        # row = 0
        #
        # # First 2d rows are equality consts
        # for i in range(self.n):
        #     jac[row, i] = 1.0
        #     row += 1
        #
        # for i in range(self.n):
        #     jac[row, i - self.n] = 1.0
        #     row += 1
        #
        # return jac.flatten()
        return np.array(non_zeros)

    def jacobianstructure(self):
        """Returns (row_indices, col_indices) for sparse Jacobian.
        Each pair indicates a non-zero entry.
        """
        # Full dense Jacobian: just return all indices
        # row_indices = []
        # col_indices = []
        # for row in range(self.m_constraints):
        #     for col in range(self.n_variables):
        #         row_indices.append(row)
        #         col_indices.append(col)
        # return np.array(row_indices), np.array(col_indices)
        # row_indices = []
        # col_indices = []
        #
        # # First n constraints
        # for i in range(self.n):
        #     row_indices.append(i)
        #     col_indices.append(i)
        #
        # # Last n constraints
        # for i in range(self.n):
        #     row_indices.append(self.n + i)
        #     col_indices.append((self.N - 1) * self.n + i)
        #
        # return (np.array(row_indices), np.array(col_indices))
        print(f"WE ARE INSIDE JACOBIAN STRUCTURE WITH {self.m_constraints} CONSTRAINTS")
        row_indices = []
        col_indices = []

        # 1. Equality constraints (2n non-zeros)
        # First n constraints: p1 = q0
        for i in range(self.n):
            row_indices.append(i)  # Constraint i
            col_indices.append(i)  # Variable p1[i]

        # Next n constraints: pN = qd
        for i in range(self.n):
            row_indices.append(self.n + i)  # Constraint n+i
            col_indices.append((self.N - 1) * self.n + i)  # Variable pN[i]

        # 2. Inequality constraints (2n non-zeros per constraint)
        # offset = 2 * self.n  # Start row for inequality constraints
        # for i in range(self.N - 1):
        #     # Constraint for segment (i, i+1)
        #     row = offset + i
        #
        #     # Derivatives wrt p_i (all n dimensions)
        #     for dim in range(self.n):
        #         row_indices.append(row)
        #         col_indices.append(i * self.n + dim)
        #
        #     # Derivatives wrt p_{i+1} (all n dimensions)
        #     for dim in range(self.n):
        #         row_indices.append(row)
        #         col_indices.append((i + 1) * self.n + dim)

        # print(f"DEBUG: Jacobian structure rows: {row_indices}")
        # print(f"DEBUG: Jacobian structure cols: {col_indices}")
        # row_indices = []
        # col_indices = []
        # for row in range(self.m_constraints):
        #     for col in range(self.n_variables):
        #         row_indices.append(row)
        #         col_indices.append(col)
        # print(f"DEBUG: Jacobian structure rows 2: {row_indices}")
        # print(f"DEBUG: Jacobian structure cols 2: {col_indices}")
        # return np.array(row_indices), np.array(col_indices)
        return (np.array(row_indices), np.array(col_indices))

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

        self.path_history.append(self.unpack_path(x))

        return True

    def bounds(self):
        """
        Returns bounds for constraints:
        - Equality constraints: cl = cu = 0
        - Inequality constraints: cl = -∞, cu = 0 (since ≤ 0)
        """
        cl = np.zeros(self.m_constraints)
        cu = np.zeros(self.m_constraints)

        # Equality constraints: p1 = q0, pN = qd
        # Set both bounds to 0 (equality)
        cl[: 2 * self.n] = 0.0
        cu[: 2 * self.n] = 0.0

        # Inequality constraints: ||p_{i+1} - p_i||^2 ≤ ζ^2
        # c(x) ≤ 0, so lower bound = -∞, upper bound = 0
        # cl[2 * self.n :] = -self.delta**2
        # cu[2 * self.n :] = 0.0

        return cl, cu


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

    print(
        f"{problem.n_variables} variables, {problem.m_constraints} constraints. N x d = {problem.N} x {problem.n}"
    )

    x0 = init_path.flatten()

    x_L = np.full(problem.n_variables, -np.inf)
    x_U = np.full(problem.n_variables, np.inf)

    # Constraint bounds
    c_L, c_U = problem.bounds()

    nlp = cyipopt.Problem(
        n=problem.n_variables,
        m=problem.m_constraints,
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
        "derivative_test_perturbation": 1e-3,
        "derivative_test_tol": 1e-2,
        # "derivative_test_print_all": "yes",
        # "hessian_approximation": "limited-memory",
        # 'gradient_approximation': 'finite-difference-values',
        # 'jacobian_approximation': 'finite-difference-values',
        # Configure L-BFGS parameters
        # 'limited_memory_update_type': 'bfgs',  # Standard BFGS update
        # 'limited_memory_max_history': 10,      # History size (10-50 is typical)
        # Disable second-order features
        # 'mehrotra_algorithm': 'no',           # Disable second-order correction
        # 'fast_step_computation': 'no',         # Disable advanced step calc
        # "alpha_for_y": "primal",  # More aggressive step sizing
        # "recalc_y": "no",  # Reduces computational overhead
        "mu_strategy": "adaptive",
        # 'linear_solver': 'mumps',
        # Adjust convergence criteria for first-order method
        "tol": 1e-3,  # Relax tolerance (default 1e-8)
        "max_iter": max_iter,  # Increase iteration limit
        "acceptable_iter": 10,  # Stop after 10 "good enough" iters
        # Output control
        "print_level": 5,
        "print_frequency_iter": 10,
    }

    for key in options.keys():
        nlp.add_option(key, options[key])

    print("Starting IPOPT solver...")
    x_opt, info = nlp.solve(x0)

    path_opt = problem.unpack_path(x_opt)

    return path_opt, problem.path_history, info


# %%
# Ipopt optimization approach
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
zeta = 0.5 * 1e5
alpha = np.log(2) / 1e-2
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
init_path = (1 - lambda_) * q0 + lambda_ * qd  # (2 x n_points)
init_path = init_path.T  # (n_points x 2)
path = init_path.copy()
delta = 1.5 * np.linalg.norm(path[:, 1] - path[:, 0]) ** 2
dists = [-100]
iter_ = 0
path_hist = [init_path.copy()]
max_iters = 1200
opt_max_iters = 100
kind = "in"
# kind = "out"
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

path_opt, path_hist, info = deform_path_ipopt(
    init_path,
    obstacles,
    max_iter=opt_max_iters,
    kind=kind,
    h=h,
    r=r,
    alpha=alpha,
    zeta=zeta * 0,
    min_path=False,
    delta=delta,
)

print(f"deformation completed in {iter_} iterations with min dist = {np.min(dists)}")
add_path(fig, path_hist, num_paths=6, base_color="black")
fig.update_layout(width=1200, height=800)
fig.show()

# fig.write_image(f"path_seed_{seed}_maxpoly_{max_polygons}.pdf")


# [jump]

# %%
"""debug"""
import numpy as np
import cyipopt


class SimpleIPOPTTest:
    """A simple test problem to debug cyipopt's derivative checker."""

    def __init__(self):
        self.n = 2  # dimension
        self.N = 2  # 2 points
        self.n_variables = self.n * self.N  # 4 variables
        self.m_constraints = 2 * self.n  # 4 constraints

        # Fixed points
        self.q0 = np.array([0.0, 0.0])
        self.qd = np.array([1.0, 1.0])

        print(
            f"Test problem: {self.n_variables} variables, {self.m_constraints} constraints"
        )

        # For debugging
        self.call_count = {
            "objective": 0,
            "gradient": 0,
            "constraints": 0,
            "jacobian": 0,
            "jacobianstructure": 0,
        }

    def pack_path(self, path):
        """Flatten the path - let's test BOTH orderings"""
        print(f"pack_path: Input shape {path.shape}")
        print(f"  Path:\n{path}")

        # Test C-order (row-major)
        x_c = path.flatten(order="C")
        print(f"  C-order (row-major): {x_c}")

        # Test F-order (column-major)
        x_f = path.flatten(order="F")
        print(f"  F-order (column-major): {x_f}")

        # Let's use C-order for now
        return x_c

    def unpack_path(self, x):
        """Unpack variables to path"""
        print(f"unpack_path: Input {x}")

        path_c = x.reshape((self.n, self.N), order="C")
        path_f = x.reshape((self.n, self.N), order="F")

        print(f"  C-order reshape:\n{path_c}")
        print(f"  F-order reshape:\n{path_f}")

        return path_c

    def objective(self, x):
        self.call_count["objective"] += 1
        print(f"\nobjective call #{self.call_count['objective']}")
        print(f"  x = {x}")

        # Simple quadratic: minimize sum of squares
        obj = np.sum(x**2)
        print(f"  objective = {obj}")
        return obj

    def gradient(self, x):
        self.call_count["gradient"] += 1
        print(f"\ngradient call #{self.call_count['gradient']}")

        # Gradient of sum(x^2) = 2x
        grad = 2 * x
        print(f"  gradient = {grad}")
        return grad

    def constraints(self, x):
        self.call_count["constraints"] += 1
        print(f"\nconstraints call #{self.call_count['constraints']}")

        # Unpack variables
        path = self.unpack_path(x)

        # Constraints: p1 = q0, pN = qd
        constraints = np.zeros(self.m_constraints)

        # First point constraints
        constraints[0] = path[0, 0] - self.q0[0]  # x1 - 0
        constraints[1] = path[1, 0] - self.q0[1]  # y1 - 0

        # Last point constraints (N=2, so p2)
        constraints[2] = path[0, 1] - self.qd[0]  # x2 - 1
        constraints[3] = path[1, 1] - self.qd[1]  # y2 - 1

        print(f"  constraints = {constraints}")
        return constraints

    def jacobian(self, x):
        self.call_count["jacobian"] += 1
        print(f"\njacobian call #{self.call_count['jacobian']}")

        # Jacobian is constant for these linear constraints
        # Non-zero pattern should be:
        # constraint 0 (x1): depends on variable 0 (x1)
        # constraint 1 (y1): depends on variable 1 (y1)
        # constraint 2 (x2): depends on variable 2 (x2)
        # constraint 3 (y2): depends on variable 3 (y2)

        # In sparse format: return [1.0, 1.0, 1.0, 1.0] for these 4 positions
        jac_values = np.ones(4)  # All derivatives are 1.0
        print(f"  jacobian values = {jac_values}")
        return jac_values

    def jacobianstructure(self):
        self.call_count["jacobianstructure"] += 1
        print(f"\njacobianstructure call #{self.call_count['jacobianstructure']}")

        # Sparse structure: (0,0), (1,1), (2,2), (3,3)
        row_indices = [0, 1, 2, 3]
        col_indices = [0, 1, 2, 3]

        print(f"  row_indices = {row_indices}")
        print(f"  col_indices = {col_indices}")
        return (np.array(row_indices), np.array(col_indices))

    def intermediate(self, *args):
        """IPOPT callback"""
        print(f"\nintermediate callback")
        return True


# Create the problem
problem = SimpleIPOPTTest()

# Set up bounds
lb = -np.inf * np.ones(problem.n_variables)
ub = np.inf * np.ones(problem.n_variables)

# Equality constraints: c(x) = 0
cl = np.zeros(problem.m_constraints)
cu = np.zeros(problem.m_constraints)

print("\n" + "=" * 60)
print("SETTING UP IPOPT PROBLEM")
print("=" * 60)

# Create IPOPT problem
nlp = cyipopt.Problem(
    n=problem.n_variables,
    m=problem.m_constraints,
    problem_obj=problem,
    lb=lb,
    ub=ub,
    cl=cl,
    cu=cu,
)

# Set options for debugging
nlp.add_option("derivative_test", "first-order")
nlp.add_option("derivative_test_tol", 1e-4)
nlp.add_option("print_level", 5)
nlp.add_option("max_iter", 0)  # Don't solve, just test derivatives
nlp.add_option("tol", 1e-8)

# Initial guess
init_path = np.array([[0.5, 1.5], [2.5, 3.5]])  # 2D, 2 points
init_path.flatten()
x0 = problem.pack_path(init_path)
problem.unpack_path(x0)  # Debug unpacking

print("\n" + "=" * 60)
print("RUNNING DERIVATIVE TEST")
print("=" * 60)

# This will run the derivative test but not solve
try:
    x, info = nlp.solve(x0)
except Exception as e:
    print(f"\nException during derivative test: {e}")

print("\n" + "=" * 60)
print("FINAL CALL COUNTS")
print("=" * 60)
for method, count in problem.call_count.items():
    print(f"{method}: {count} calls")

# Let's also do a manual finite difference check
print("\n" + "=" * 60)
print("MANUAL FINITE DIFFERENCE CHECK")
print("=" * 60)

epsilon = 1e-6

# Check constraints at x0
c0 = problem.constraints(x0)
print(f"\nConstraints at x0: {c0}")

# Compute finite difference Jacobian
J_fd = np.zeros((problem.m_constraints, problem.n_variables))
for j in range(problem.n_variables):
    x_pert = x0.copy()
    x_pert[j] += epsilon
    c_pert = problem.constraints(x_pert)
    J_fd[:, j] = (c_pert - c0) / epsilon

print(f"\nFinite difference Jacobian (full 4x4):")
print(J_fd)

# Compare with analytic Jacobian
rows, cols = problem.jacobianstructure()
jac_vals = problem.jacobian(x0)
J_analytic = np.zeros((problem.m_constraints, problem.n_variables))
for r, c, v in zip(rows, cols, jac_vals):
    J_analytic[r, c] = v

print(f"\nAnalytic Jacobian (from sparse):")
print(J_analytic)

print(f"\nDifference (max): {np.max(np.abs(J_analytic - J_fd))}")
