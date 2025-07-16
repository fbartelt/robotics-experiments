# %%
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import PillowWriter
from scipy.optimize import brute, linprog
from scipy.spatial import ConvexHull, HalfspaceIntersection

class Curve:
    def __init__(self, h0_func, dh0ds_func, obstacles, zeta=1.0, s_samples=100, r=0.8, alpha=np.log(2) / 0.2):
        """
        h0_func: function h0(s) -> R^n
        grad_phi_func: function grad_phi(h: np.ndarray[M, n], t: float) -> np.ndarray[M, n]
        gamma: relaxation constant
        s_samples: number of discretization points in s ∈ [0, 1]
        """
        self.zeta = zeta

        # Discretize s ∈ [0, 1]
        self.s = np.linspace(0, 1, s_samples)

        # Store h0(s) as array [M, n]
        self.h0_func = h0_func
        self.dh0ds_func = dh0ds_func
        self.h0 = np.array([h0_func(si) for si in self.s])

        # Initialize current curve as a copy of h0
        self.h = self.h0.copy()
        self.obstacles = obstacles
        self.r = r
        self.alpha = alpha

    def step(self, dt, t, obstacles=None):
        """
        Perform one time step of explicit Euler integration
        """
        if obstacles is None:
            obstacles = self.obstacles
        # Compute gradient of phi at current positions
        # dist, grad = signed_dist(self.h, obstacles=obstacles, t=t, r=self.r, alpha=self.alpha)
        #
        # # Euler update: h <- h + dt * ( -gamma (h - h0) + grad_phi )
        # self.h += dt * (-self.zeta * (self.h - self.h0) + grad)
        for i, s in enumerate(self.s):
            index = self.get_index_from_s(s)
            h0s = self.h0[index]
            hs = self.h[index]
            grad = self.deformation(hs, obstacles, t, r=self.r, alpha=self.alpha)
            self.h[index] = hs + dt * (-self.zeta * (hs - h0s) + 10*grad)

    def eval(self, s, t=0, obstacles=None):
        """
        Evaluate the curve at point s
        """
        if obstacles is None:
            obstacles = self.obstacles
        index = self.get_index_from_s(s)
        return self.h[index]

    def deformation(self, point, obstacles=None, t=0.0, r=None, alpha=None):
        if r is None:
            r = self.r
        if alpha is None:
            alpha = self.alpha
        if obstacles is None:
            obstacles = self.obstacles

        if not isinstance(obstacles, list):
            obstacles = [obstacles]

        dist, grad = signed_dist(point, obstacles, t, r, alpha)
        threshold = (1/alpha) * np.log(2)
        if dist > threshold:
            grad = grad * 0.0
        if dist >= 0.0:
            gain, _ = smooth_sat(0.1/(abs(dist) + 1e-3), np.log(2)/(10.0 * 1e2))
        else:
            gain = abs(dist) * 1e4 * 10.0
        return gain * grad.ravel()

    def dh_ds(self, s, t, obstacles, delta=1e-3):
        """
        Approximate ∂h/∂s at a specific s and t using directional derivative of the deformation
        deformation_func: function that takes (point, obstacles, t, alpha) and returns deformation vector
        """
        # h0(s)
        index = self.get_index_from_s(s)
        point = self.h0[index]

        # dh0/ds analytically
        dh0_ds = self.dh0ds_func(s)

        if t <= 0.0:
            return dh0_ds

        # Directional numerical derivative of deformation
        dir_vec = dh0_ds / (np.linalg.norm(dh0_ds) + 1e-6)  # normalize to avoid scale issues

        deform_plus = self.deformation(point + delta * dir_vec, obstacles, t, r=self.r, alpha=self.alpha).ravel()
        deform_minus = self.deformation(point - delta * dir_vec, obstacles, t, r=self.r, alpha=self.alpha).ravel()
        ddef = (deform_plus - deform_minus) / (2 * delta)

        return dh0_ds + ddef

    def dh_dt(self, s, t, obstacles, dt=1e-3):
        """
        Approximate ∂h/∂t at a specific s and t using directional derivative of the deformation
        """
        # h0(s)
        index = self.get_index_from_s(s)
        h0s = self.h0[index]
        curr_hs = self.h[index]

        # dh0/dt analytically
        grad = self.deformation(curr_hs, obstacles, t, r=self.r, alpha=self.alpha)
        next_hs = curr_hs + dt * (-self.zeta * (curr_hs - h0s) + 10*grad)
        prev_hs = curr_hs - dt * (-self.zeta * (curr_hs - h0s) + 10*grad)
        dh_dt = (next_hs - prev_hs) / (2 * dt)
        return dh_dt


    def get_index_from_s(self, s):
        """
        Given a scalar s ∈ [0, 1], return the closest index in the discretized s array.
        """
        idx = np.argmin(np.abs(self.s - s))
        return idx

    def get_curve(self):
        """
        Return current discretized curve
        """
        return self.h

    def get_params(self):
        return self.s

def h0(s):
    return np.array([np.cos(2 * np.pi * s), np.sin(2 * np.pi * s), 0.0])


# def h(s, t, obstacles):
#     point = h0(s)
#     if t > 0.0:
#         deform = deformation(point, obstacles, t, r=0.8, alpha=np.log(2) / 1.0).ravel()
#         # print(f"Time: {t}, deformation: {deform.ravel()}")
#         return point + deform
#     else:
#         return point
#
#
def dh0_ds(s):
    return np.array(
        [-2 * np.pi * np.sin(2 * np.pi * s), 2 * np.pi * np.cos(2 * np.pi * s), 0.0]
    )


def find_s_star(q, curve):
    params = curve.get_params()
    min_val, min_idx = np.inf, -1
    points = curve.get_curve()

    for s in params:
        idx = curve.get_index_from_s(s)
        h_s = points[idx]
        val = np.linalg.norm(q - h_s)
        if val < min_val:
            min_val = val
            min_idx = idx
    return params[min_idx]  # Return the s corresponding to the closest point


def plot_things():
    traj, xi_ns, xi_ts, xis, curves_hist, obstacle_hist = simulate()
    steps = len(curves_hist)

    fig, ax = plt.subplots()
    ax.set_aspect("equal")
    ax.set_xlim(-3, 3)
    ax.set_ylim(-3, 3)
    ax.grid(True)

    # Initialize the curve and point
    (curve_line,) = ax.plot([], [], "b-", lw=2, label="Curve")
    (point_dot,) = ax.plot([], [], "ro", label="Robot")
    (obstacle_patch,) = ax.plot([], [], "k--", lw=1.5, label="Obstacle")

    def init():
        curve_line.set_data([], [])
        point_dot.set_data([], [])
        obstacle_patch.set_data([], [])
        return curve_line, point_dot

    def update(frame):
        curve = np.array(curves_hist[frame])
        point = traj[frame].ravel()

        curve_line.set_data(curve[:, 0], curve[:, 1])
        point_dot.set_data([point[0]], [point[1]])
        obstacle = np.array(obstacle_hist[frame])
        obstacle = np.vstack([obstacle, obstacle[0]])  # close the polygon
        obstacle_patch.set_data(obstacle[:, 0], obstacle[:, 1])
        return curve_line, point_dot, obstacle_patch

    ani = animation.FuncAnimation(
        fig, update, frames=steps, init_func=init, blit=True, interval=20, repeat=False
    )

    ax.legend()
    ani.save("animation.mp4", writer="ffmpeg", fps=30)


def simulate():
    q0 = np.array([2.0, 0.0, 0.0])
    k_n = 2.0
    dt = 1e-2
    steps = 500
    A0 = np.array(
        [
            [1, 0],  # x ≤ 1
            [-1, 0],  # -x ≤ 1 → x ≥ -1
            [0, 1],  # y ≤ 1
            [0, -1],  # -y ≤ 1 → y ≥ -1
        ]
    )
    b0 = np.array([0.1, 0.1, 0.1, 0.1])
    b0 = b0 + (A0 @ np.array([-2.0, -1.0]).reshape(-1, 1)).ravel()
    vel_vec = 3.0 * np.array([0.5, 0.0])  # Velocity vector for the obstacle
    obstacle = MovingPolytope(A0, b0, v=vel_vec)
    obstacles = [obstacle]

    zv = q0.copy() * 0.0
    xi_ns, xi_ts, xis = [zv], [zv], [zv]
    q = q0.copy()
    trajectory = [q.copy()]
    n_samples = 500
    curves_hist = [[h0(s) for s in np.linspace(0, 1, n_samples)]]
    vertices = get_polytope_vertices(A0, b0)
    obstacle_hist = [vertices.copy()]

    curve = Curve(
        h0_func=h0,
        dh0ds_func=dh0_ds,
        obstacles=obstacles,
        zeta=5.0,
        s_samples=n_samples,
        r=0.8,
        alpha=np.log(2) / 0.5,
    )

    for i in range(steps):
        if i % 50 == 0:
            print(f"Step {i}/{steps}, Current Position: {q}")
            # check if curve changed
            ht0 = np.array(curves_hist[0])
            htn = np.array(curves_hist[-1])
            print(np.any(ht0 != htn))
        time = i * dt
        curve.step(dt, time, obstacles=obstacles)
        s_star = find_s_star(q, curve)
        h_curr = curve.get_curve()

        h_s = curve.eval(s_star, t=time)
        curves_hist.append(h_curr.copy())
        A_, b_ = obstacles[0](t=time)
        vertices = get_polytope_vertices(A_, b_)
        obstacle_hist.append(vertices.copy())
        distance = np.linalg.norm(q - h_s)
        kn = k_n * np.tanh(distance)
        kt = k_n * np.tanh(1 - distance)
        normal = -kn * (q - h_s) / (np.linalg.norm(q - h_s) + 1e-6)
        tangent = curve.dh_ds(s_star, t=time, obstacles=obstacles)
        tangent = kt * tangent / (np.linalg.norm(tangent) + 1e-6)
        ff = curve.dh_dt(s_star, t=time, obstacles=obstacles, dt=dt)
        dq = normal + tangent + ff
        xi_ns.append(normal.copy())
        xi_ts.append(tangent.copy())
        xis.append(dq.copy())
        q = q + 5 * dq * dt
        trajectory.append(q.copy())

    return trajectory, xi_ns, xi_ts, xis, curves_hist, obstacle_hist


def guidance_vf(point, t, obstacles, k_n=2.0):
    point = np.array(point, dtype=float)
    s_star = find_s_star(point, t, obstacles)
    h_s = h(s_star, t, obstacles)
    distance = np.linalg.norm(point - h_s)
    kn = k_n * np.tanh(distance)
    kt = k_n * np.tanh(1 - distance)
    normal = -kn * (point - h_s) / (np.linalg.norm(point - h_s) + 1e-6)
    tangent = dh_ds(s_star)
    tangent = kt * tangent / (np.linalg.norm(tangent) + 1e-6)
    normal = np.array(normal, dtype=float)
    tangent = np.array(tangent, dtype=float)
    return normal + tangent


def phi(sigma, gamma=0.1):
    if sigma < 0.0:
        return 0.0
    else:
        return (sigma**3) / (2 * (sigma + gamma))


def dphi_dsigma(sigma, gamma=0.1):
    if sigma < 0.0:
        return 0.0
    else:
        return ((sigma**2) * (2 * sigma + 3 * gamma)) / (2 * (sigma + gamma) ** 2)


def outer_distance(point, obstacles, t):
    point = np.array(point, dtype=float).reshape(-1, 1)
    if not isinstance(obstacles, list):
        obstacles = [obstacles]

    dists, grads = [], []

    for obstacle in obstacles:
        A, b = obstacle(t=t)
        A = np.hstack(
            (A.copy(), np.zeros((A.shape[0], 1)))
        )  # Add a column for the point
        sigmas = [(ai @ point - bi).item() for ai, bi in zip(A, b)]
        itens = [phi(sigma) for sigma in sigmas]
        ditens = [dphi_dsigma(sigma) for sigma in sigmas]
        ditens = [ds * ai for ds, ai in zip(ditens, A)]
        sum_ = sum(itens)
        dsum_ = sum(ditens)
        dists.append(sum_ / len(itens))
        grads.append(dsum_ / len(itens))

    imin = np.argmin(dists)
    dist, grad = dists[imin], grads[imin]

    return dist, grad.reshape(1, -1)


def inner_distance(point, obstacles, t, r=0.1):
    point = np.array(point, dtype=float).reshape(-1, 1)
    if not isinstance(obstacles, list):
        obstacles = [obstacles]

    dists, grads = [], []

    for obstacle in obstacles:
        A, b = obstacle(t=t)
        A = np.hstack(
            (A.copy(), np.zeros((A.shape[0], 1)))
        )  # Add a column for the point
        sigmas = [(bi - ai @ point).item() for ai, bi in zip(A, b)]
        items, ditens = [], []
        for i, sigma in enumerate(sigmas):
            val = phi(sigma)
            ai = A[i]
            if val > 1e-6:
                item = val ** (-1 / r)
                ditem = (-1 / r) * (val ** (-(1 + r) / r)) * dphi_dsigma(sigma) * (-ai)
            else:
                item = 1e-6 ** (-1 / r)
                ditem = 0.0
            items.append(item)
            ditens.append(ditem)
        # itens = [phi(sigma) ** (-1/r) for sigma in sigmas]
        # ditens = [
        #     (-1/r) * (phi(sigma) ** (-(1+r)/r)) * dphi_dsigma(sigma)
        #     * (-ai) for sigma, ai in zip(sigmas, A)
        # ]
        sum_ = sum(items)
        dist_ = (sum_ / len(items)) ** (-r)
        first_chain = -r * (dist_ / sum_)
        grad_ = first_chain * sum(ditens)
        dists.append(dist_)
        grads.append(grad_)

    imin = np.argmax(dists)
    dist, grad = dists[imin], grads[imin]

    return -dist, -grad.reshape(1, -1)


def smooth_sat(x, alpha=np.log(2) / 0.2):
    sat = -1 / alpha * np.log(np.exp(-alpha * x) / 2 + 0.5)
    dsat = 1.0 / (np.exp(alpha * x) + 1.0)
    return sat, dsat


def signed_dist(point, obstacles, t, r=0.1, alpha=np.log(2) / 0.2):
    point = np.array(point, dtype=float).reshape(-1, 1)
    outer_dist, outer_grad = outer_distance(point, obstacles, t)
    inner_dist, inner_grad = inner_distance(point, obstacles, t, r)
    # print(outer_dist, outer_grad.ravel())
    dist = outer_dist + inner_dist
    grad = outer_grad + inner_grad
    # dist, dsat = smooth_sat(dist, alpha)
    # grad = dsat * grad
    # print(f"Sat: {dist}, Grad: {grad.ravel()}")
    return dist, grad.reshape(1, -1)


def deformation(point, obstacles, t, r=0.1, alpha=np.log(2) / 0.2):
    if not isinstance(obstacles, list):
        obstacles = [obstacles]
    # A_list, b_list = [], []
    #
    # for obstacle in obstacles:
    #     A, b = obstacle.A, obstacle.b
    #     A_list.append(A)
    #     b_list.append(b)
    #
    # constraints = [(A, b) for A, b in zip(A_list, b_list)]
    dist, grad = signed_dist(point, obstacles, t, r, alpha)
    if dist > 0.4:
        grad = grad * 0.0
    if dist >= 0.0:
        gain, _ = smooth_sat(0.1/(abs(dist) + 1e-3), np.log(2)/10.0)
    else:
        gain = abs(dist) * 1e4
    return gain * grad.ravel()


class MovingPolytope:
    def __init__(self, A0, b0, v=None, fun_A=None, fun_b=None):
        self.A0 = A0
        self.b0 = b0.ravel()
        self.curr_time = 0.0
        if v is None:
            self.v = np.ones(A0.shape[1])
        else:
            self.v = np.array(v, dtype=float).reshape(-1, 1)
        if fun_A is None:
            self.A = self._default_A
        else:
            self.A = fun_A
        if fun_b is None:
            self.b = self._default_b
        else:
            self.b = fun_b

    def __call__(self, t=None, u=None):
        if t is None:
            t = self.curr_time
        if u is None:
            u = self.v
        return self.A(t), self.b(t)

    def _default_A(self, t, v=None):
        return self.A0

    def _default_b(self, t, v=None):
        if v is None:
            v = self.v
        return self.b0 + (self.A0 @ v * t).ravel()

    def move(self, d):
        A = self.A0
        # x0 = x + d => Ax = A(x0 - d) <= b => Ax0 <= b + Ad
        b = self.b0 + self.A0 @ d
        self.A = A
        self.b = b


def get_polytope_vertices(A, b):
    m, n = A.shape
    # Objective: maximize δ (slack)
    c = np.zeros(n + 1)
    c[-1] = -1  # Maximize δ ⇒ minimize -δ

    # Constraints: A x + δ ||A_i|| ≤ b_i
    norms = np.linalg.norm(A, axis=1)
    A_lp = np.hstack((A, norms[:, None]))
    bounds = [(None, None)] * n + [(0, None)]  # δ ≥ 0

    res = linprog(c, A_ub=A_lp, b_ub=b, bounds=bounds, method="highs")
    interior_point = res.x[:-1]  # Exclude δ
    halfspaces = np.hstack((A, -b[:, None]))
    hs = HalfspaceIntersection(halfspaces, interior_point)
    reconstructed_vertices = hs.intersections

    # Use ConvexHull to order them
    hull = ConvexHull(reconstructed_vertices)
    ordered_vertices = reconstructed_vertices[hull.vertices]
    return ordered_vertices


plot_things()
