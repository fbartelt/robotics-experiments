import time
import numpy as np
import plotly.graph_objects as go
import plotly.colors as pc
import cyipopt
from itertools import combinations
from scipy.spatial import HalfspaceIntersection, ConvexHull, Delaunay
from scipy.optimize import linprog
from cyipopt import Problem
from numba import njit

# To display in notebook
from IPython.display import HTML, display
import webbrowser
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.colors import LinearSegmentedColormap

# from distances import signed_dist2convex, phi, smooth_min
from smoothfunctions import (
    signedDist2Convex,
    smoothMinListWithGradient,
    smoothMinList,
    phi,
    ESDF_CGAL,
)
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
    for i, path_ in enumerate(paths2add):
        path = path_.T  # Transpose to (3 x N)
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


class OptimalPathProblemESDF:
    """
    Optimal path deformation problem using IPOPT.
    min sum_{i=1}^{N} smoothsat(ESDF(O, p_i)) + sum_{i=1}^{N-1} zeta/2 * ||p_{i+1} - p_{i}||^2
    s.t. p_1 = q0
         p_N = qd

    where smoothsat is a smooth saturation function

    """

    def __init__(
        self,
        obstacles,
        init_path,
        zeta,
        alpha,
        min_path,
        delta=0,
        *args,
        **kwargs,
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
        self.m_constraints = 2 * self.n + (self.N - 1)
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
                time_0 = time.time()
                dist_ij, grad, closest_pt = ESDF_CGAL(p_i, obs.A, obs.b.reshape(-1, 1))
                time_1 = time.time()
                # print(f"ESDF_CGAL time: {time_1 - time_0:.6f} seconds")
                if np.isinf(dist_ij):
                    print(f"Dist={dist_ij} for point {p_i.ravel()}")
                    print(f"Closest point: {closest_pt.ravel()}")
                    print(
                        f"Expected dist: {np.linalg.norm(p_i.ravel() - closest_pt.ravel())}"
                    )
                # if dist_ij < 0 or np.isnan(dist_ij) or np.isinf(dist_ij):
                #     print(f"Point {i}, Obstacle {j}, dist = {dist_ij}")
                dists[j] = dist_ij
            min_dist = min(dists)
            # Smooth saturation (avoids cheating by circunventing the map)
            exponent = np.clip(-self.alpha * min_dist, -100, 100)
            sat_dist = (-1 / self.alpha) * np.log(
                0.5 * (1 + np.exp(exponent))
            )  # Smooth saturation
            sat_dist = self.alpha * min_dist if min_dist < 0 else min_dist
            if (
                np.isnan(sat_dist)
                or np.isinf(sat_dist)
                or np.isnan(min_dist)
                or np.isinf(min_dist)
            ):
                print(
                    f"DEBUG: NaN or Inf detected at point {i}: smooth_min_dist={min_dist}, sat_dist={sat_dist}"
                )
                raise ValueError("NaN or Inf detected in objective computation.")

            total_cost += -sat_dist  # Minimize negative sat_dist to maximize distance
            # Path length cost
            if self.min_path and i < self.N - 1:
                p_next = path[i + 1].reshape(-1, 1)
                total_cost += (self.zeta / 2) * np.linalg.norm(p_next - p_i) ** 4

        return total_cost

    def gradient(self, x):
        path = self.unpack_path(x)
        grad = np.zeros_like(path)  # N x n

        for i in range(self.N):
            p_i = path[i].reshape(-1, 1)
            dists = np.zeros(self.n_obstacles)
            grads = np.zeros((self.n, self.n_obstacles))

            for j, obs in enumerate(self.obstacles):
                dist_ij, grad_ij, _ = ESDF_CGAL(p_i, obs.A, obs.b.reshape(-1, 1))
                dists[j] = dist_ij
                grads[:, j] = grad_ij.flatten()

            # Smooth min grad is shaped (1 x n_obstacles)
            id_min = np.argmin(dists)
            min_dist = min(dists)
            min_grad = grads[:, id_min]
            if np.isnan(min_grad).any():
                min_grad = np.zeros_like(min_grad)
            # Smooth saturation gradient
            exponent = np.clip(self.alpha * min_dist, -100, 100)
            grad_sat = 1 / (1 + np.exp(exponent))
            grad_sat = self.alpha if min_dist < 0 else 1.0
            if (
                np.isnan(grad_sat)
                or np.isinf(grad_sat)
                or np.isnan(min_grad).any()
                or np.isinf(min_grad).any()
            ):
                print(
                    f"DEBUG: NaN or Inf detected at point {i}: smooth_min_dist={min_dist}, grad_sat={grad_sat}, smooth_min_grad={min_grad}"
                )
            grad_full = grad_sat * min_grad  # n x 1

            grad[i] += -grad_full.flatten()  # Minimize negative sat_dist
            # Path length gradient
            if self.min_path and i < self.N - 1:
                p_next = path[i + 1].reshape(-1, 1)
                seg_norm = np.linalg.norm(p_next - p_i) ** 2
                # 1/2 ||p - p_i||^4 ==> 2 * ||p - p_i||^2 * (p - p_i)
                grad[i] += -2 * self.zeta * (p_next - p_i).flatten() * seg_norm
                grad[i + 1] += 2 * self.zeta * (p_next - p_i).flatten() * seg_norm

        return grad.flatten()

    def constraints(self, x):
        path = self.unpack_path(x)
        constraints = np.zeros(self.m_constraints)
        # Equality constraints: p1 = q0, pN = qd (2n constraints)
        constraints[: self.n] = path[0] - self.q0.flatten()
        constraints[self.n : 2 * self.n] = path[-1] - self.qd.flatten()

        return constraints

    def jacobian(self, x):
        path = self.unpack_path(x)
        jac = np.zeros((self.m_constraints, self.n_variables))
        row = 0
        # First 2d rows are equality consts
        for i in range(self.n):
            jac[row, i] = 1.0
            row += 1

        for i in range(self.n):
            jac[row, i - self.n] = 1.0
            row += 1

        return jac.flatten()

    def jacobianstructure(self):
        """Returns (row_indices, col_indices) for sparse Jacobian.
        Each pair indicates a non-zero entry.
        """
        # Full dense Jacobian: just return all indices
        row_indices = []
        col_indices = []
        for row in range(self.m_constraints):
            for col in range(self.n_variables):
                row_indices.append(row)
                col_indices.append(col)
        return np.array(row_indices), np.array(col_indices)

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

        return cl, cu


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
        self.m_constraints = 2 * self.n + (self.N - 1)
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
            sat_dist = self.alpha * smooth_min_dist if smooth_min_dist < 0 else smooth_min_dist
            if (
                np.isnan(sat_dist)
                or np.isinf(sat_dist)
                or np.isnan(smooth_min_dist)
                or np.isinf(smooth_min_dist)
            ):
                print(
                    f"DEBUG: NaN or Inf detected at point {i}: smooth_min_dist={smooth_min_dist}, sat_dist={sat_dist}"
                )

            total_cost += -sat_dist  # Minimize negative sat_dist to maximize distance
            # Path length cost
            if self.min_path and i < self.N - 1:
                p_next = path[i + 1].reshape(-1, 1)
                total_cost += (self.zeta / 2) * np.linalg.norm(p_next - p_i) ** 4

            # Curvature cost
            # if i < self.N - 2:
            #     # Discrete approximation of
            #     p_ip1 = path[i + 1].reshape(-1, 1)
            #     p_ip2 = path[i + 2].reshape(-1, 1)

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
            grad_sat = self.alpha if smooth_min_dist < 0 else 1.0
            if (
                np.isnan(grad_sat)
                or np.isinf(grad_sat)
                or np.isnan(smooth_min_grad).any()
                or np.isinf(smooth_min_grad).any()
            ):
                print(
                    f"DEBUG: NaN or Inf detected at point {i}: smooth_min_dist={smooth_min_dist}, grad_sat={grad_sat}, smooth_min_grad={smooth_min_grad}"
                )
            # sat_dist = (-1 / self.alpha) * np.log(
            #     0.5 * (1 + np.exp(-self.alpha * smooth_min_dist))
            # )  # Smooth saturation

            grad_full = grad_sat * (grads @ smooth_min_grad.reshape(-1, 1))  # n x 1
            # Point 29 has negative distance outside obstacle in test case ??
            # if i == 29:
            #     print(f"p_i: {p_i.ravel()}")
            #     aux = [ phi(b - (a @ p_i), eps=self.h) for a, b in zip(self.obstacles[0].A, self.obstacles[0].b)]
            #     print(f"DEBUG: point {i}, aux: {aux}")
            #     print(
            #         f"[DEBUG]: point {i}, dists: {dists}, smooth_min_dist: {smooth_min_dist}, grad_sat: {grad_sat}, smooth_min_grad: {smooth_min_grad.flatten()}, grads: {grads.flatten()}, grad_full: {grad_full.flatten()}"
            #     )
            grad[i] += -grad_full.flatten()  # Minimize negative sat_dist
            # Path length gradient
            if self.min_path and i < self.N - 1:
                p_next = path[i + 1].reshape(-1, 1)
                seg_norm = np.linalg.norm(p_next - p_i) ** 2
                grad[i] += -2 * self.zeta * (p_next - p_i).flatten() * seg_norm
                grad[i + 1] += 2 * self.zeta * (p_next - p_i).flatten() * seg_norm

        return grad.flatten()

    def constraints(self, x):
        path = self.unpack_path(x)
        constraints = np.zeros(self.m_constraints)
        # Equality constraints: p1 = q0, pN = qd (2n constraints)
        constraints[: self.n] = path[0] - self.q0.flatten()
        constraints[self.n : 2 * self.n] = path[-1] - self.qd.flatten()

        # Inequality constraints: ||p_{i+1} - p_i||^2 ≤ ζ^2 (N-1 constraints)
        # offset = 2 * self.n
        # for i in range(self.N - 1):
        #     p_i = path[i]
        #     p_next = path[i + 1]
        #     diff = p_next - p_i
        #     constraints[offset + i] = np.dot(diff, diff)

        return constraints

    def jacobian(self, x):
        path = self.unpack_path(x)
        jac = np.zeros((self.m_constraints, self.n_variables))
        row = 0
        # First 2d rows are equality consts
        for i in range(self.n):
            jac[row, i] = 1.0
            row += 1

        for i in range(self.n):
            jac[row, i - self.n] = 1.0
            row += 1

        # for i in range(self.N - 1):
        #     idx_i = i * self.n
        #     idx_ip1 = (i + 1) * self.n
        #
        #     diff = path[i + 1] - path[i]
        #
        #     grad_i = -2 * diff
        #     grad_ip1 = -grad_i
        #     jac[row, idx_i : idx_i + self.n] = grad_i
        #     jac[row, idx_ip1 : idx_ip1 + self.n] = grad_ip1
        #     row += 1

        return jac.flatten()

    def jacobianstructure(self):
        """Returns (row_indices, col_indices) for sparse Jacobian.
        Each pair indicates a non-zero entry.
        """
        # Full dense Jacobian: just return all indices
        row_indices = []
        col_indices = []
        for row in range(self.m_constraints):
            for col in range(self.n_variables):
                row_indices.append(row)
                col_indices.append(col)
        return np.array(row_indices), np.array(col_indices)

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
        # c(x) >= 0, so lower bound = 0, upper bound = zeta^2
        # cu[2 * self.n :] = self.delta**2

        return cl, cu


def deform_path_ipopt(
    init_path,
    obstacles,
    method="ours",
    max_iter=200,
    ipopt_options=None,
    verbose=True,
    **kwargs,
):
    if method.lower() == "esdf":
        problem = OptimalPathProblemESDF(
            init_path=init_path,
            obstacles=obstacles,
            **kwargs,
        )
    else:
        problem = OptimalPathProblem(
            init_path=init_path,
            obstacles=obstacles,
            **kwargs,
        )

    if verbose:
        method_str = "ESDF-based" if method.lower() == "esdf" else "SmoothSDF-based"
        msg = f"Deforming path using IPOPT ({method_str} formulation)..."
        msg += f"\nNumber of path points: {problem.N}, Dimension: {problem.n}"
        msg += f"\nNumber of obstacles: {len(obstacles)}"
        msg += f"\nOptimization variables: {problem.n_variables}, constraints: {problem.m_constraints}"
        print(msg)

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

    if ipopt_options is None:
        print_lvl = 5 if verbose else 0
        options = {
            "mu_strategy": "adaptive",
            "tol": 1e-6,  # Relax tolerance (default 1e-8)
            "max_iter": max_iter,  # Increase iteration limit
            "acceptable_iter": 10,  # Stop after 10 "good enough" iters
            # Output control
            "print_level": print_lvl,
            "print_frequency_iter": 10,
        }
    else:
        options = ipopt_options

    for key in options.keys():
        nlp.add_option(key, options[key])

    if verbose:
        print("Starting IPOPT solver...")
    x_opt, info = nlp.solve(x0)

    path_opt = problem.unpack_path(x_opt)

    return path_opt, problem.path_history, info


def find_strictly_feasible_point(A, b):
    """
    Solve an LP to find a strictly feasible point x such that A x < b
    """
    m, n = A.shape
    # Objective: maximize δ (slack)
    c = np.zeros(n + 1)
    c[-1] = -1  # Maximize δ ⇒ minimize -δ

    # Constraints: A x + δ ||A_i|| ≤ b_i
    norms = np.linalg.norm(A, axis=1)
    A_lp = np.hstack((A, norms[:, None]))
    bounds = [(None, None)] * n + [(0, None)]  # δ ≥ 0

    res = linprog(c, A_ub=A_lp, b_ub=b, bounds=bounds, method="highs")
    if res.success:
        return res.x[:-1]  # Return x (ignore δ)
    else:
        raise ValueError("Could not find a strictly feasible point.")


def get_polytope_vertices_opt(A, b, tol=1e-6):
    n_dim = A.shape[1]  # Dimension of the polytope (e.g., 2 for 2D)
    vertices = []

    # Generate direction vectors (all combinations of ±1 in each dimension)
    directions = []
    for signs in combinations([-1, 1] * n_dim, n_dim):
        directions.append(np.array(signs))
    directions = np.unique(directions, axis=0)  # Remove duplicates

    # Solve LP for each direction
    for c in directions:
        res = linprog(
            c=-c,  # Maximize c^T x (linprog minimizes, so we negate)
            A_ub=A,
            b_ub=b,
            bounds=(None, None),  # No bounds beyond Ax ≤ b
            method="highs",  # Uses the HiGHS solver
        )
        if res.success:
            vertex = np.round(res.x, int(-np.log10(tol)))
            if not any(np.allclose(vertex, v, atol=tol) for v in vertices):
                vertices.append(vertex)
    interior_point = find_strictly_feasible_point(A, b)
    halfspaces = np.hstack((A, -b[:, None]))
    hs = HalfspaceIntersection(halfspaces, interior_point)
    reconstructed_vertices = hs.intersections

    # Use ConvexHull to order them
    hull = ConvexHull(reconstructed_vertices)
    vertices = reconstructed_vertices[hull.vertices]

    return np.array(vertices)


def add_polygon_plt(ax, A, b, alpha=0.5, color=(0.64, 0.62, 0.61)):
    """Add polygon to matplotlib axes"""
    vertices = get_polytope_vertices_opt(A, b)
    # print(vertices)
    hull = ConvexHull(vertices)
    poly_vertices = vertices[hull.vertices]
    ax.fill(
        poly_vertices[:, 0],
        poly_vertices[:, 1],
        color=color,
        alpha=alpha,
        edgecolor=color,
        linewidth=1,
    )


def animate_deformation_matplotlib(
    path_list,
    init_path,
    obstacles,
    q0,
    qd,
    p1,
    p2,
    distances,
    frame_delay=200,  # ms between frames
):
    # Create static figure and axes
    fig, ax = plt.subplots(figsize=(10, 8))
    n_iters = len(path_list)
    # Precompute all paths
    paths = [init_path.copy()]
    paths.extend(path_list)

    # Setup static elements
    # Create RdBu-like colormap
    cmap = LinearSegmentedColormap.from_list("RdBu", ["#2166ac", "#f7f7f7", "#b2182b"])
    contour = ax.contourf(p1, p2, distances, levels=20, cmap=cmap, alpha=1.0)
    cbar = plt.colorbar(contour, ax=ax)
    cbar.set_label("Distance")

    # Add polygons
    # for A, b in obstacles:
    for obs in obstacles:
        A, b = obs.A, obs.b
        add_polygon_plt(ax, A, b)

    ax.plot(q0[0, 0], q0[1, 0], "gx", markersize=10, label="q0")
    ax.plot(qd[0, 0], qd[1, 0], "b*", markersize=10, label="qd")

    # Initialize path line
    (path_line,) = ax.plot([], [], "ok-", linewidth=2)

    # Add legend and set limits
    ax.legend()
    ax.set_xlim(np.min(p1), np.max(p1))
    ax.set_ylim(np.min(p2), np.max(p2))
    ax.set_aspect("equal", "box")
    ax.set_title("Path Deformation Animation")
    ax.grid(True)

    # Animation update function
    def update(frame):
        path = paths[frame]
        # path_line.set_data(path[0, :], path[1, :])
        path_line.set_data(path[:, 0], path[:, 1])
        ax.set_title(f"Deformation Step: {frame}/{n_iters}")
        return (path_line,)

    # Create animation
    ani = FuncAnimation(
        ax.figure, update, frames=n_iters, interval=frame_delay, blit=True
    )

    return ani

def show_animation(animation, filename="anim.html"):
    path = Path(filename).absolute()
    animation.save(path, writer="html")
    webbrowser.open(f"file://{path}")
