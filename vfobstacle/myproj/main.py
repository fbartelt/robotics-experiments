import numpy as np
from manim import *
from scipy.optimize import brute, linprog
from scipy.spatial import ConvexHull, HalfspaceIntersection


class DefaultTemplate(Scene):
    def construct(self):
        circle = Circle()  # create a circle
        circle.set_fill(PINK, opacity=0.5)  # set color and transparency

        square = Square()  # create a square
        square.flip(RIGHT)  # flip horizontally
        square.rotate(-3 * TAU / 8)  # rotate a certain amount

        self.play(Create(square))  # animate the creation of the square
        self.wait(1)
        self.play(Transform(square, circle))  # interpolate the square into the circle
        self.wait(1)
        self.play(FadeOut(square))  # fade out animation
        self.wait(1)


class CreatingMobjects(Scene):
    def construct(self):
        circle = Circle()
        self.add(circle)
        self.wait(1)
        # self.remove(circle)
        # self.wait(1)


class VectorArrow(Scene):
    def construct(self):
        dot = Dot(ORIGIN)
        arrow = Arrow(ORIGIN, [2, 2, 0], buff=0)
        numberplane = NumberPlane()
        origin_text = Text("(0, 0)").next_to(dot, DOWN)
        tip_text = Text("(2, 2)").next_to(arrow.get_end(), RIGHT)
        self.add(numberplane, dot, arrow, origin_text, tip_text)
        self.wait(1)


class GVF(Scene):
    def construct(self):
        q0 = np.array([2.0, 0.0, 0.0])
        k_n = 2.0
        dt = 1e-2
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
        b0 = b0 + (A0 @ np.array([-1.1, -1.0]).reshape(-1, 1)).ravel()
        vel_vec = 1.0 * np.array([0.0, 0.5])  # Velocity vector for the obstacle
        obstacle = MovingPolytope(A0, b0, v=vel_vec)
        vertices = get_polytope_vertices(A0, b0)
        vertices = np.hstack((vertices, np.zeros((vertices.shape[0], 1))))  # Add z=0 for 2D
        # print(vertices, *vertices)
        square = Polygon(*vertices, color=PINK, fill_opacity=0.5, stroke_width=1.0)
        def update_obstacle(mob):
            time = int(t_tracker.get_value() * (500 - 1)) * dt
            obstacle.curr_time = time
            v_ = np.hstack((obstacle.v.ravel(), np.zeros(1)))  # Add z=0 for 2D
            A_, b_ = obstacle(t=time)
            vertices = get_polytope_vertices(A_, b_)
            vertices = np.hstack((vertices, np.zeros((vertices.shape[0], 1))))
            mob.become(Polygon(*vertices, color=PINK, fill_opacity=0.5))
            # mob.move_to(mob.get_center() + (v_ * time).ravel())
        square.add_updater(update_obstacle)

        def aut_trajectory(q0, k_n=k_n, dt=dt):
            time = int(t_tracker.get_value() * (500 - 1)) * dt
            trajectory, xi_ns, xi_ts, xis = simulate_trajectory(
                q0, t=time, obstacles=obstacle, k_n=k_n, dt=dt
            )
            return trajectory, xi_ns, xi_ts, xis

        trajectory, xi_ns, xi_ts, xis = aut_trajectory(q0, k_n=k_n, dt=dt)
        print("Created trajectory")

        # Draw the curve
        curve = ParametricFunction(h0, t_range=[0, 1], color=BLUE)
        def update_curve(mob):
            time = int(t_tracker.get_value() * (len(trajectory) - 1)) * dt
            new_curve = ParametricFunction(
                lambda s: h(s, time, obstacle), t_range=[0, 1], color=BLUE
            )
            mob.become(new_curve)
        curve.add_updater(update_curve)
        print("Done curve")

        # Animate the trajectory
        path = VMobject(color=RED)
        path.set_points_as_corners([trajectory[0]])
        dot = Dot(point=trajectory[0], color=YELLOW)

        def update_normal_arrow(mob):
            i = int(t_tracker.get_value() * (len(trajectory) - 1))
            q = np.array(trajectory[i])
            v = 0.2 * np.array(xi_ns[i], dtype=float)
            # print("normal", v.shape, q.shape)
            # print(f"norm: {np.linalg.norm(v)}")
            if np.linalg.norm(v) > 1e-2:
                mob.put_start_and_end_on(q, q + v)
                mob.tip.set_opacity(1.0)
                mob.set(opacity=1.0)
            else:
                mob.tip.set_opacity(0.0)
                mob.set(opacity=0.0)

        def update_tangent_arrow(mob):
            i = int(t_tracker.get_value() * (len(trajectory) - 1))
            q = np.array(trajectory[i])
            v = 0.2 * np.array(xi_ts[i], dtype=float)
            # print("tangent", v.shape, q.shape)
            if np.linalg.norm(v) > 1e-2:
                mob.put_start_and_end_on(q, q + v)
                mob.tip.set_opacity(1.0)
                mob.set(opacity=1.0)
            else:
                mob.tip.set_opacity(0.0)
                mob.set(opacity=0.0)

        def update_sum_arrow(mob):
            i = int(t_tracker.get_value() * (len(trajectory) - 1))
            q = np.array(trajectory[i])
            v = 0.2 * np.array(xis[i], dtype=float)
            # print("sum", v.shape, q.shape)
            if np.linalg.norm(v) > 1e-2:
                mob.put_start_and_end_on(q, q + v)
                mob.tip.set_opacity(1.0)
                mob.set(opacity=1.0)
            else:
                mob.tip.set_opacity(0.0)
                mob.set(opacity=0.0)

        normal_arrow = (
            Arrow(
                ORIGIN, ORIGIN + RIGHT, color=ORANGE, max_tip_length_to_length_ratio=0.1
            )
            .add_updater(update_normal_arrow)
            .update()
        )
        tangent_arrow = (
            Arrow(
                ORIGIN, ORIGIN + RIGHT, color=GREEN, max_tip_length_to_length_ratio=0.1
            )
            .add_updater(update_tangent_arrow)
            .update()
        )
        sum_arrow = (
            Arrow(
                ORIGIN, ORIGIN + RIGHT, color=TEAL_E, max_tip_length_to_length_ratio=0.1
            )
            .add_updater(update_sum_arrow)
            .update()
        )
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

        vector_field = ArrowVectorField(
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

        vector_field.add_updater(vf_updater)

        dot.add_updater(update_path_and_dot)
        print("Done vf, dot and path")
        trail = TracedPath(dot.get_center, stroke_color=YELLOW, dissipating_time=3.0)
        plane = NumberPlane()

        print("Starting animation")
        self.play(Create(plane, run_time=1, lag_ratio=0.05))
        self.play(
            Create(curve, run_time=2, lag_ratio=0.05),
            DrawBorderThenFill(dot, run_time=2, lag_ratio=0.05),
        )
        self.wait(1)

        self.play(Create(vector_field, run_time=2, lag_ratio=0.05))
        self.wait()

        self.add(normal_arrow, tangent_arrow, sum_arrow, square, trail)
        self.play(t_tracker.animate.set_value(1.0), run_time=6, rate_func=linear)
        self.wait()

def h0(s):
    return np.array([np.cos(2 * np.pi * s), np.sin(2 * np.pi * s), 0.0])

def h(s, t, obstacles):
    point = h0(s)
    if t > 0.0:
        deform = deformation(point, obstacles, t, r=0.8, alpha=np.log(2)/1.0).ravel()
        # print(f"Time: {t}, deformation: {deform.ravel()}")
        return point + deform
    else:
        return point


def dh_ds(s, t, obstacles):
    point = h0(s)
    dh0_ds = np.array(
        [-2 * np.pi * np.sin(2 * np.pi * s), 2 * np.pi * np.cos(2 * np.pi * s), 0.0]
    )
    if t > 0.0:
        # Numerical derivative of deformation
        delta = 1e-3
        deform_plus = deformation(point + delta * dh0_ds, obstacles, t, alpha=np.log(2)/1.0).ravel()
        deform_minus = deformation(point - delta * dh0_ds, obstacles, t, alpha=np.log(2)/1.0).ravel()
        ddef = (deform_plus - deform_minus) / (2 * delta)
        return dh0_ds + ddef
    else:
        return dh0_ds

def find_s_star(q, t, obstacles):
    def objective(s):
        return np.linalg.norm(q - h(s[0], t, obstacles)) ** 2

    result = brute(objective, ranges=[(0, 1)], Ns=200, full_output=True, finish=None)
    return result[0]


def simulate_trajectory(q0, t, obstacles, k_n=2.0, k_t=1.0, dt=0.01, steps=500):
    zv = q0.copy() * 0.0
    xi_ns, xi_ts, xis = [zv], [zv], [zv]
    q = q0.copy()
    trajectory = [q.copy()]
    for i in range(steps):
        time = i * dt
        s_star = find_s_star(q, time, obstacles)
        h_s = h(s_star, time, obstacles)
        distance = np.linalg.norm(q - h_s)
        kn = k_n * np.tanh(distance)
        kt = k_n * np.tanh(1 - distance)
        normal = -kn * (q - h_s) / (np.linalg.norm(q - h_s) + 1e-6)
        tangent = dh_ds(s_star, time, obstacles)
        tangent = kt * tangent / (np.linalg.norm(tangent) + 1e-6)
        dq = normal + tangent
        xi_ns.append(normal.copy())
        xi_ts.append(tangent.copy())
        xis.append(dq.copy())

        q = q + 3 * dq * dt
        trajectory.append(q.copy())
    return trajectory, xi_ns, xi_ts, xis


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
            if val > 1e-3:
                item = val ** (-1 / r)
                ditem = (-1 / r) * (val ** (-(1 + r) / r)) * dphi_dsigma(sigma) * (-ai)
            else:
                item = 1e-3 ** (-1 / r)
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


def smooth_sat(x, alpha=np.log(2)/0.2):
    sat = -1 / alpha * np.log(np.exp(-alpha * x) / 2 + 0.5)
    dsat = 1.0 / (np.exp(alpha * x) + 1.0)
    return sat, dsat


def signed_dist(point, obstacles, t, r=0.1, alpha=np.log(2)/0.2):
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


def deformation(point, obstacles, t, r=0.1, alpha=np.log(2)/0.2):
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

    res = linprog(c, A_ub=A_lp, b_ub=b, bounds=bounds, method='highs')
    interior_point = res.x[:-1]  # Exclude δ
    halfspaces = np.hstack((A, -b[:, None]))
    hs = HalfspaceIntersection(halfspaces, interior_point)
    reconstructed_vertices = hs.intersections

    # Use ConvexHull to order them
    hull = ConvexHull(reconstructed_vertices)
    ordered_vertices = reconstructed_vertices[hull.vertices]
    return ordered_vertices

