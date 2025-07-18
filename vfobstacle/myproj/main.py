import numpy as np
from manim import *
from scipy.optimize import brute, linprog
from scipy.spatial import ConvexHull, HalfspaceIntersection


class Curve:
    def __init__(
        self,
        h0_func,
        dh0ds_func,
        obstacles,
        zeta=1.0,
        s_samples=100,
        r=0.8,
        alpha=np.log(2) / 0.2,
        gamma=0.1,
        eta=1.0,
        beta=1.0,
    ):
        """
        h0_func: function h0(s) -> R^n
        grad_phi_func: function grad_phi(h: np.ndarray[M, n], t: float) -> np.ndarray[M, n]
        gamma: relaxation constant
        s_samples: number of discretization points in s ∈ [0, 1]
        """
        self.zeta = zeta

        # Discretize s ∈ [0, 1]
        self.s = np.linspace(0, 1, s_samples)
        self.n_samples = s_samples
        self.delta_s = self.s[1] - self.s[0]

        # Store h0(s) as array [M, n]
        self.h0_func = h0_func
        self.dh0ds_func = dh0ds_func
        self.h0 = np.array([h0_func(si) for si in self.s])

        # Initialize current curve as a copy of h0
        self.h = self.h0.copy()
        self.obstacles = obstacles
        self.r = r
        self.alpha = alpha
        self.gamma = gamma
        self.eta = eta
        self.beta = beta

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
        delta_s = self.s[1] - self.s[0]  # Assuming uniform spacing
        for i, s in enumerate(self.s):
            index = self.get_index_from_s(s)
            dh_dts = self.dh_dt(s, t, obstacles, dt=dt)
            self.h[index] += dt * dh_dts
            # h0s = self.h0[index]
            # hs = self.h[index]
            # grad = self.deformation(hs, obstacles, t, r=self.r, alpha=self.alpha, gamma=self.gamma)
            # # Add path lenght constraint
            # prev_grad = hs - self.h[(index - 1) % self.n_samples]
            # next_grad = hs - self.h[(index + 1) % self.n_samples]
            # orig_prev_grad = h0s - self.h0[(index - 1) % self.n_samples]
            # orig_next_grad = h0s - self.h0[(index + 1) % self.n_samples]
            # orig_const_grad = (orig_prev_grad + orig_next_grad) / (2.0 * (delta_s)**2)
            # const_grad = (prev_grad + next_grad) / (2.0 * (delta_s)**2)
            # len_grad = const_grad - orig_const_grad
            # self.h[index] = hs + dt * (-self.zeta * (hs - h0s) + self.eta * grad - self.beta * len_grad)

    def eval(self, s, t=0, obstacles=None):
        """
        Evaluate the curve at point s
        """
        if obstacles is None:
            obstacles = self.obstacles
        index = self.get_index_from_s(s)
        return self.h[index]

    def deformation(self, point, obstacles=None, t=0.0, r=None, alpha=None, gamma=None):
        if r is None:
            r = self.r
        if alpha is None:
            alpha = self.alpha
        if gamma is None:
            gamma = self.gamma
        if obstacles is None:
            obstacles = self.obstacles

        if not isinstance(obstacles, list):
            obstacles = [obstacles]

        dist, grad = signed_dist(point, obstacles, t, r=r, alpha=alpha, gamma=gamma)
        threshold = (1 / alpha) * np.log(2)
        if dist > threshold:
            grad = grad * 0.0
        if dist >= 0.0:
            gain, _ = smooth_sat(1.0 / (abs(dist) + 1e-3), np.log(2) / (10.0))
        else:
            gain = np.sqrt(abs(dist)) * 1e4
            grad = grad / (
                np.linalg.norm(grad) + 1e-6
            )  # Normalize grad to avoid scale issues
            # print(f"Dist: {dist}, Gain: {gain}, Grad: {grad.ravel()}")
        return gain * grad.ravel()

    def dh_ds(self, s, t, obstacles, delta=None):
        """
        Approximate ∂h/∂s at a specific s and t using directional derivative of the deformation
        deformation_func: function that takes (point, obstacles, t, alpha) and returns deformation vector
        """
        # h0(s)
        if delta is None:
            delta = self.s[1] - self.s[0]  # Assuming uniform spacing
        index = self.get_index_from_s(s)
        point = self.h[index]

        # dh0/ds analytically
        dh0_ds = self.dh0ds_func(s)

        if t <= 0.0:
            return dh0_ds

        next_point = self.h[(index + 1) % self.n_samples]
        prev_point = self.h[(index - 1) % self.n_samples]
        # Numerical derivative of the curve
        dh_ds = (next_point - prev_point) / (2 * delta)

        # # Directional numerical derivative of deformation
        # dir_vec = dh0_ds / (
        #     np.linalg.norm(dh0_ds) + 1e-6
        # )  # normalize to avoid scale issues
        #
        # deform_plus = self.deformation(
        #     point + delta * dir_vec, obstacles, t, r=self.r, alpha=self.alpha
        # ).ravel()
        # deform_minus = self.deformation(
        #     point - delta * dir_vec, obstacles, t, r=self.r, alpha=self.alpha
        # ).ravel()
        # ddef = (deform_plus - deform_minus) / (2 * delta)
        #
        return dh_ds

    def dh_dt(self, s, t, obstacles=None, dt=1e-3):
        """
        Approximate ∂h/∂t at a specific s and t using directional derivative of the deformation
        """
        # h0(s)
        if obstacles is None:
            obstacles = self.obstacles
        index = self.get_index_from_s(s)
        h0s = self.h0[index]
        hs = self.h[index]
        grad = self.deformation(
            hs, obstacles, t, r=self.r, alpha=self.alpha, gamma=self.gamma
        )
        # Add path lenght constraint
        prev_grad = hs - self.h[(index - 1) % self.n_samples]
        next_grad = hs - self.h[(index + 1) % self.n_samples]
        orig_prev_grad = h0s - self.h0[(index - 1) % self.n_samples]
        orig_next_grad = h0s - self.h0[(index + 1) % self.n_samples]
        orig_const_grad = (orig_prev_grad + orig_next_grad) / (
            2.0 * (self.delta_s) ** 2
        )
        const_grad = (prev_grad + next_grad) / (2.0 * (self.delta_s) ** 2)
        len_grad = const_grad - orig_const_grad
        dh_dt = -self.zeta * (hs - h0s) + self.eta * grad - self.beta * len_grad

        # curr_hs = self.h[index]
        #
        # # grad = self.deformation(curr_hs, obstacles, t, r=self.r, alpha=self.alpha)
        # # next_hs = curr_hs + dt * (-self.zeta * (curr_hs - h0s) + 10 * grad)
        # # prev_hs = curr_hs - dt * (-self.zeta * (curr_hs - h0s) + 10 * grad)
        # # dh_dt = (next_hs - prev_hs) / (2 * dt)
        # next_deform = self.deformation(
        #     curr_hs, obstacles, t + dt, r=self.r, alpha=self.alpha, gamma=self.gamma
        # )
        # prev_deform = self.deformation(
        #     curr_hs, obstacles, t - dt, r=self.r, alpha=self.alpha, gamma=self.gamma
        # )
        # next_h = curr_hs + next_deform
        # prev_h = curr_hs + prev_deform
        # dh_dt = (next_h - prev_h) / (2 * dt)
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


class GVF(Scene):
    def construct(self):
        q0 = np.array([2.0, 0.0, 0.0])
        k_n = 2.0
        dt = 5e-3
        T = 15.0  # Total time
        steps = int(T / dt)
        t_tracker = ValueTracker(0)

        A0 = np.array(
            [
                [1, 0],  # x ≤ 1
                [-1, 0],  # -x ≤ 1 → x ≥ -1
                [0, 1],  # y ≤ 1
                [0, -1],  # -y ≤ 1 → y ≥ -1
            ]
        )
        b0 = np.array([0.1, 0.1, 0.1, 0.1])

        A0 = np.array(
            [
                [1.0, 0.0],  # x ≤ 1.2
                [-1.0, 0.0],  # x ≥ -0.8 → -x ≤ 0.8
                [0.0, 1.0],  # y ≤ 1.0
                [0.0, -1.0],  # y ≥ -0.5 → -y ≤ 0.5
                [0.5, 1.0],  # 0.5x + y ≤ 1.5
                [-1.0, 0.5],  # -x + 0.5y ≤ 1.0
                [1.0, -1.0],  # x - y ≤ 0.8
            ]
        )

        b0 = np.array(
            [
                1.2,  # x ≤ 1.2
                0.8,  # x ≥ -0.8
                1.0,  # y ≤ 1.0
                0.5,  # y ≥ -0.5
                1.5,  # 0.5x + y ≤ 1.5
                1.0,  # -x + 0.5y ≤ 1.0
                0.8,  # x - y ≤ 0.8
            ]
        )
        b0 /= 2.0  # Scale down the constraints
        b0 = b0 + (A0 @ np.array([-1.0, -1.0]).reshape(-1, 1)).ravel()
        vel_vec = 0 * 0.75 * np.array([0.5, 0.5])  # Velocity vector for the obstacle
        obstacle = MovingPolytope(A0, b0, v=vel_vec)
        vertices = get_polytope_vertices(A0, b0)
        vertices = np.hstack(
            (vertices, np.zeros((vertices.shape[0], 1)))
        )  # Add z=0 for 2D
        curveobj = Curve(
            h0,
            dh0_ds,
            obstacles=[obstacle],
            zeta=0.2,
            s_samples=100,
            r=0.8,
            alpha=np.log(2) / 0.4,
            gamma=0.1,
            eta=0.2,
            beta=5e-3,
        )
        print(f"delta_s: {curveobj.s[1] - curveobj.s[0]}")
        # print(vertices, *vertices)
        square = Polygon(*vertices, color=PINK, fill_opacity=0.5, stroke_width=1.0)

        def update_obstacle(mob):
            time = int(t_tracker.get_value() * (steps - 1)) * dt
            obstacle.curr_time = time
            v_ = np.hstack((obstacle.v.ravel(), np.zeros(1)))  # Add z=0 for 2D
            A_, b_ = obstacle(t=time)
            vertices = get_polytope_vertices(A_, b_)
            vertices = np.hstack((vertices, np.zeros((vertices.shape[0], 1))))
            mob.become(Polygon(*vertices, color=PINK, fill_opacity=0.5))
            # mob.move_to(mob.get_center() + (v_ * time).ravel())

        square.add_updater(update_obstacle)

        def aut_trajectory(q0, k_n=k_n, dt=dt):
            time = int(t_tracker.get_value() * (steps - 1)) * dt
            trajectory, xi_ns, xi_ts, xis, ffs, curve_hist = simulate_trajectory(
                q0,
                t=time,
                obstacles=obstacle,
                curve=curveobj,
                k_n=k_n,
                dt=dt,
                steps=steps,
            )
            return trajectory, xi_ns, xi_ts, xis, ffs, curve_hist

        trajectory, xi_ns, xi_ts, xis, ffs, curve_hist = aut_trajectory(
            q0, k_n=k_n, dt=dt
        )
        print("Created trajectory")

        # Draw the curve
        curve = VMobject(color=BLUE)
        curve.set_points_smoothly(curve_hist[0])  # or .set_points_as_corners()

        def update_curve(mob):
            index = int(t_tracker.get_value() * (len(curve_hist) - 1))
            new_points = curve_hist[index]
            new_curve = VMobject(color=BLUE)
            new_curve.set_points_smoothly(new_points)
            mob.become(new_curve)

        curve.add_updater(update_curve)

        dots_group = VGroup(
            *[Dot(point=pt, radius=0.02, color=MAROON_E) for pt in curve_hist[0]]
        )

        def update_dots(mob):
            index = int(t_tracker.get_value() * (len(curve_hist) - 1))
            new_points = curve_hist[index]

            # Replace existing dots with new ones
            new_dots = VGroup(
                *[Dot(point=pt, radius=0.02, color=MAROON_E) for pt in new_points]
            )
            mob.become(new_dots)

        dots_group.add_updater(update_dots)
        # curve = ParametricFunction(h0, t_range=[0, 1], color=BLUE)
        #
        # def update_curve(mob):
        #     time = int(t_tracker.get_value() * (len(trajectory) - 1)) * dt
        #     new_curve = ParametricFunction(
        #         lambda s: h(s, time, obstacle), t_range=[0, 1], color=BLUE
        #     )
        #     mob.become(new_curve)
        #
        # curve.add_updater(update_curve)
        print("Done curve")

        # Animate the trajectory
        path = VMobject(color=RED)
        path.set_points_as_corners([trajectory[0]])
        dot = Dot(point=trajectory[0], color=YELLOW)

        def update_arrow_wrapper(arrow):
            def update_arrow(mob):
                i = int(t_tracker.get_value() * (len(trajectory) - 1))
                q = np.array(trajectory[i])
                v = np.array(arrow[i], dtype=float)
                norm_v = np.linalg.norm(v)
                if norm_v > 1e-4:
                    size = min(0.3, norm_v)
                    v = size * v / (np.linalg.norm(v) + 1e-6)  # Normalize and scale
                    mob.put_start_and_end_on(q, q + v)
                    mob.tip.set_opacity(1.0)
                    mob.set_stroke(opacity=1.0)
                else:
                    mob.put_start_and_end_on(q, q + 1 * UL)
                    mob.tip.set_opacity(0.0)
                    mob.set_stroke(opacity=0.0)

            return update_arrow

        normal_arrow = Arrow(
            trajectory[0],
            trajectory[0] + 0.2 * LEFT,
            color=PURE_RED,
            max_tip_length_to_length_ratio=0.3,
            stroke_width=4.0,
        ).add_updater(update_arrow_wrapper(xi_ns))
        tangent_arrow = Arrow(
            trajectory[0],
            trajectory[0] + 0.2 * UL,
            color=PURE_GREEN,
            max_tip_length_to_length_ratio=0.3,
        ).add_updater(update_arrow_wrapper(xi_ts))
        sum_arrow = Arrow(
            trajectory[0],
            trajectory[0] + 0.2 * (UP - 0.5 * LEFT),
            # trajectory[0],
            # trajectory[0] + 0.2 * xis[0],
            color=ORANGE,
            max_tip_length_to_length_ratio=0.3,
            stroke_width=3.0,
        ).add_updater(update_arrow_wrapper(xis))
        ff_arrow = Arrow(
            trajectory[0],
            trajectory[0] + 0.2 * DL,
            color=PURE_BLUE,
            max_tip_length_to_length_ratio=0.3,
        ).add_updater(update_arrow_wrapper(ffs))
        print("Done arrows")

        def update_path_and_dot(
            mob,
        ):
            # index = int(alpha * (len(trajectory) - 1))
            index = int(t_tracker.get_value() * (len(trajectory) - 1))
            new_path = VMobject(color=RED)
            new_path.set_points_as_corners(trajectory[: index + 1])
            path.become(new_path)
            dot.move_to(trajectory[index])
            return mob

        # vector_field = ArrowVectorField(
        #     lambda x: guidance_vf(
        #         x,
        #         t=int(t_tracker.get_value() * (len(trajectory) - 1)) * dt,
        #         obstacles=obstacle,
        #         k_n=k_n,
        #     ),
        #     x_range=[-5, 5, 0.2],
        #     y_range=[-5, 5, 0.2],
        #     length_func=lambda x: 0.1 * np.linalg.norm(x),
        #     color=WHITE,
        #     opacity=0.3,
        # )

        def vf_updater(mob):
            t = int(t_tracker.get_value() * (len(trajectory) - 1)) * dt
            vf_new = ArrowVectorField(
                lambda x: guidance_vf(
                    x,
                    t=int(t_tracker.get_value() * (len(trajectory) - 1)) * dt,
                    obstacles=obstacle,
                    k_n=k_n,
                ),
                x_range=[-5, 5, 0.2],
                y_range=[-5, 5, 0.2],
                length_func=lambda x: 0.1 * np.linalg.norm(x),
                color=WHITE,
                opacity=0.3,
            )
            mob.become(vf_new)

        # vector_field.add_updater(vf_updater)

        dot.add_updater(update_path_and_dot)
        print("Done vf, dot and path")
        trail = TracedPath(dot.get_center, stroke_color=YELLOW, dissipating_time=3.0)
        plane = NumberPlane()

        print("Starting animation")
        self.play(Create(plane, run_time=1, lag_ratio=0.05))
        self.play(
            Create(curve, run_time=1, lag_ratio=0.05),
            DrawBorderThenFill(dot, run_time=1, lag_ratio=0.05),
            DrawBorderThenFill(square, run_time=1, lag_ratio=0.05),
            Create(dots_group, run_time=1, lag_ratio=0.05),
        )
        # self.wait(1)

        # self.play(Create(vector_field, run_time=2, lag_ratio=0.05))
        # self.wait()

        # self.add(normal_arrow, tangent_arrow, sum_arrow, ff_arrow, trail)
        self.play(
            GrowArrow(normal_arrow, run_time=2, lag_ratio=0.05),
            GrowArrow(tangent_arrow, run_time=2, lag_ratio=0.05),
            GrowArrow(sum_arrow, run_time=2, lag_ratio=0.05),
            GrowArrow(ff_arrow, run_time=2, lag_ratio=0.05),
            Create(trail, run_time=2, lag_ratio=0.05),
        )
        self.play(t_tracker.animate.set_value(1.0), run_time=15, rate_func=linear)
        self.wait(1)


def h0(s):
    return np.array([np.cos(2 * np.pi * s), np.sin(2 * np.pi * s), 0.0])


def h(s, t, obstacles):
    point = h0(s)
    if t > 0.0:
        deform = deformation(point, obstacles, t, r=0.8, alpha=np.log(2) / 1.0).ravel()
        # print(f"Time: {t}, deformation: {deform.ravel()}")
        return point + deform
    else:
        return point


def dh0_ds(s):
    dh0 = np.array(
        [-2 * np.pi * np.sin(2 * np.pi * s), 2 * np.pi * np.cos(2 * np.pi * s), 0.0]
    )
    return dh0


def dh_ds(s, t, obstacles):
    point = h0(s)
    dh0ds = dh0_ds(s)

    if t > 0.0:
        # Numerical derivative of deformation
        delta = 1e-3
        deform_plus = deformation(
            point + delta * dh0ds, obstacles, t, alpha=np.log(2) / 1.0
        ).ravel()
        deform_minus = deformation(
            point - delta * dh0ds, obstacles, t, alpha=np.log(2) / 1.0
        ).ravel()
        ddef = (deform_plus - deform_minus) / (2 * delta)
        return dh0ds + ddef
    else:
        return dh0ds


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


def simulate_trajectory(q0, t, obstacles, curve, k_n=2.0, k_t=1.0, dt=0.01, steps=500):
    zv = q0.copy() * 0.0
    xi_ns, xi_ts, xis, ffs = [zv], [zv], [zv], [zv]
    q = q0.copy()
    trajectory = [q.copy()]
    curve_hist = [curve.h0.copy()]
    for i in range(steps):
        time = i * dt
        curve.step(dt, time, obstacles=obstacles)
        s_star = find_s_star(q, curve)
        hs_full = curve.get_curve()
        h_s = hs_full[curve.get_index_from_s(s_star)]
        # s_star = find_s_star(q, time, obstacles)
        # h_s = h(s_star, time, obstacles)
        distance = np.linalg.norm(q - h_s)
        kn = k_n * np.tanh(distance)
        kt = k_n * np.tanh(1 - distance)
        normal = -kn * (q - h_s) / (np.linalg.norm(q - h_s) + 1e-6)
        tangent = curve.dh_ds(s_star, time, obstacles, delta=1e-3)
        # tangent = dh_ds(s_star, time, obstacles)
        norm_tan = tangent / (np.linalg.norm(tangent) + 1e-6)
        tangent = kt * norm_tan
        ff = curve.dh_dt(s_star, time, obstacles, dt=dt / 10)
        # Null space projection (remove tangent component)
        ff = (np.eye(len(ff)) - np.outer(norm_tan, norm_tan)) @ ff
        # ff = ff / (np.linalg.norm(ff) + 1e-6)  # Normalize to avoid scale issues
        # print(f"Debug: {ff.ravel()}")
        dq = 1.5 * (normal + tangent) + ff
        xi_ns.append(normal.copy())
        xi_ts.append(tangent.copy())
        xis.append(dq.copy())
        ffs.append(ff.copy())

        q = q + dq * dt
        trajectory.append(q.copy())
        curve_hist.append(hs_full.copy())

    return trajectory, xi_ns, xi_ts, xis, ffs, curve_hist


def guidance_vf(point, t, obstacles, k_n=2.0):
    point = np.array(point, dtype=float)
    s_star = find_s_star(point, t, obstacles)
    h_s = h(s_star, t, obstacles)
    distance = np.linalg.norm(point - h_s)
    kn = k_n * np.tanh(distance)
    kt = k_n * np.tanh(1 - distance)
    normal = -kn * (point - h_s) / (np.linalg.norm(point - h_s) + 1e-6)
    tangent = dh_ds(s_star, t, obstacles)
    tangent = kt * tangent / (np.linalg.norm(tangent) + 1e-6)
    normal = np.array(normal, dtype=float)
    tangent = np.array(tangent, dtype=float)
    return normal + tangent


def phi(sigma, gamma=0.1):
    if sigma < 0.0:
        return 0.0
    else:
        try:
            return (sigma**3) / (2 * (sigma + gamma))
        except OverflowError:
            print(f"OverflowError in phi with sigma={sigma}, gamma={gamma}")
            raise OverflowError("Overflow in phi calculation")


def dphi_dsigma(sigma, gamma=0.1):
    if sigma < 0.0:
        return 0.0
    else:
        return ((sigma**2) * (2 * sigma + 3 * gamma)) / (2 * (sigma + gamma) ** 2)


def outer_distance(point, obstacles, t, gamma=0.1):
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
        # print(f"Sigmas: {sigmas}")
        itens = [phi(sigma, gamma=gamma) for sigma in sigmas]
        ditens = [dphi_dsigma(sigma, gamma=gamma) for sigma in sigmas]
        ditens = [ds * ai for ds, ai in zip(ditens, A)]
        sum_ = sum(itens)
        dsum_ = sum(ditens)
        dists.append(sum_ / len(itens))
        grads.append(dsum_ / len(itens))

    imin = np.argmin(dists)
    dist, grad = dists[imin], grads[imin]

    return dist, grad.reshape(1, -1)


def inner_distance(point, obstacles, t, r=0.1, gamma=0.1):
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
            val = phi(sigma, gamma=gamma)
            ai = A[i]
            if val > 1e-6:
                item = val ** (-1 / r)
                ditem = (
                    (-1 / r)
                    * (val ** (-(1 + r) / r))
                    * dphi_dsigma(sigma, gamma=gamma)
                    * (-ai)
                )
            else:
                item = 1e-6 ** (-1 / r)
                ditem = np.zeros_like(ai)  # Avoid division by zero
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


def signed_dist(point, obstacles, t, r=0.1, alpha=np.log(2) / 0.2, gamma=0.1):
    point = np.array(point, dtype=float).reshape(-1, 1)
    outer_dist, outer_grad = outer_distance(point, obstacles, t, gamma=gamma)
    inner_dist, inner_grad = inner_distance(point, obstacles, t, r=r, gamma=gamma)
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
        gain, _ = smooth_sat(0.1 / (abs(dist) + 1e-3), np.log(2) / 10.0)
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
