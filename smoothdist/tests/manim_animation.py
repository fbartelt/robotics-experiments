# %%
from manim import *
import numpy as np


class SquareDistanceScene(Scene):
    def construct(self):
        # ── load data ─────────────────────────────────────────────
        data = np.load("anim_data.npz")
        t_array = data["t"]  # (N_frames,)
        V_square_history = data["V_square_history"]  # (N_frames, 4, 3)
        hdsdf = data["distances"]  # Holder distance
        esdf = data["euclidean_distances"]  # Euclidean signed distance
        V_pentagon = data["V_pentagon"]  # (M, 3)

        # For varying curve
        epsilons = data["eps_keys"]
        hdsdf_vs_eps = data["eps_values"]
        gammas = data["gamma_keys"]
        hdsdf_vs_gamma = data["gamma_values"]

        N = len(t_array)
        T_total = t_array[-1]
        print(f"Loaded {N} frames, T = {T_total:.2f} s")

        # ── scale & vertical shift for better visibility ──────────
        scale = 2.0
        y_shift = 2.5  # move everything up by 2.5 units

        V_square_history = V_square_history * scale
        V_square_history[..., 1] += y_shift

        V_pentagon = V_pentagon * scale
        V_pentagon[:, 1] += y_shift

        # ── static pentagon (3D points, z=0) ──────────────────────
        pentagon = Polygon(*V_pentagon, color=GREEN, stroke_width=3)
        pentagon.set_fill(GREEN, opacity=0.15)
        self.add(pentagon)

        # ── moving square ─────────────────────────────────────────
        # Start with first frame, preserve style throughout
        square = Polygon(*V_square_history[0], color=RED, stroke_width=3)
        square.set_fill(RED, opacity=0.2)

        def update_square(mob):
            idx = int(round(time_tracker.get_value() / T_total * (N - 1)))
            idx = max(0, min(N - 1, idx))
            # Directly update vertex positions – no style loss
            # mob.set_points_as_corners(V_square_history[idx])
            mob.set_points_as_corners(
                np.vstack([V_square_history[idx], V_square_history[idx][0]])
            )

        square.add_updater(update_square)
        self.add(square)

        # ── graph axes (wider, bottom half) ───────────────────────
        all_hdsdf = [hdsdf]  # list of all HD‑SDF arrays we'll ever show
        if "hdsdf_vs_eps" in data:
            all_hdsdf.extend(list(data["hdsdf_vs_eps"]))
        # (add hdsdf_vs_gamma similarly if you store that)
        hdy_min = min(np.min(arr) for arr in all_hdsdf)
        hdy_max = max(np.max(arr) for arr in all_hdsdf)
        y_min = min(hdy_min, esdf.min())
        y_max = max(hdy_max, esdf.max())
        y_margin = 0.05 * (y_max - y_min) or 0.1
        axes = Axes(
            x_range=[0, T_total, 0.5],
            y_range=[y_min - y_margin, y_max + y_margin, 0.1],
            x_length=config.frame_width * 0.9,  # fill most of the width
            y_length=4,
            axis_config={"include_numbers": True, "font_size": 20},
            tips=False,
        )
        axes.shift(DOWN * 1.5)  # keep it in the lower half
        axes_labels = axes.get_axis_labels("Time (s)", "Distance")
        self.add(axes, axes_labels)

        # ── dynamic graph lines (initial animation) ───────────────
        hd_line = VMobject()
        euc_line = VMobject()

        def update_graph():
            idx = int(round(time_tracker.get_value() / T_total * (N - 1)))
            idx = max(0, min(N - 1, idx))

            t_slice = t_array[: idx + 1]
            h_slice = hdsdf[: idx + 1]
            e_slice = esdf[: idx + 1]

            # HD‑SDF
            line_h = axes.plot_line_graph(
                t_slice,
                h_slice,
                line_color=BLUE,
                add_vertex_dots=False,
                stroke_width=3,
            )
            hd_line.become(line_h)

            # SDF
            line_e = axes.plot_line_graph(
                t_slice,
                e_slice,
                line_color=YELLOW,
                add_vertex_dots=False,
                stroke_width=3,
            )
            euc_line.become(line_e)

        # attach updaters (function, not VGroup)
        hd_line.add_updater(lambda m: update_graph())
        euc_line.add_updater(lambda m: update_graph())

        self.add(hd_line, euc_line)
        # ── dynamic graph lines ───────────────────────────────────
        # graph_group = VGroup()
        #
        # def update_graph(group):
        #     idx = int(round(time_tracker.get_value() / T_total * (N - 1)))
        #     idx = max(0, min(N - 1, idx))
        #
        #     group.submobjects = []  # clear previous lines
        #
        #     t_slice = t_array[: idx + 1]
        #     h_slice = hdsdf[: idx + 1]
        #     e_slice = esdf[: idx + 1]
        #
        #     line_h = axes.plot_line_graph(
        #         t_slice,
        #         h_slice,
        #         line_color=BLUE,
        #         add_vertex_dots=False,
        #         stroke_width=3,
        #     )
        #     line_e = axes.plot_line_graph(
        #         t_slice,
        #         e_slice,
        #         line_color=YELLOW,
        #         add_vertex_dots=False,
        #         stroke_width=3,
        #     )
        #     group.add(line_h, line_e)

        # graph_group.add_updater(update_graph)
        # self.add(graph_group)

        # ── legend ────────────────────────────────────────────────
        legend_h = Tex("HD-SDF", color=BLUE).scale(0.5)
        legend_e = Tex("SDF", color=YELLOW).scale(0.5)
        legend_h.next_to(axes, RIGHT, buff=-2).shift(UP * 0.5)
        legend_e.next_to(legend_h, DOWN, buff=0.15)
        self.add(legend_h, legend_e)

        # ── time tracker ──────────────────────────────────────────
        time_tracker = ValueTracker(0)

        # ── play ─────────────────────────────────────────────────
        self.play(
            time_tracker.animate.set_value(T_total),
            # run_time=T_total,
            run_time=10,
            rate_func=linear,
        )
        self.wait(1)

        # ── phase 2: morph HD‑SDF while epsilon changes ──────────
        # Remove the old HD‑SDF updater and keep the SDF static
        hd_line.clear_updaters()
        euc_line.clear_updaters()
        # The SDF line is already the full curve; we can leave it as is.

        # ValueTracker for epsilon (will go from first to last value)
        eps_tracker = ValueTracker(epsilons[0])
        epsilon_label = DecimalNumber(
            epsilons[0], num_decimal_places=3, font_size=24, color=BLUE
        )
        epsilon_label.next_to(axes, UP, buff=0.3)
        self.add(epsilon_label)

        # Updater that interpolates the HD‑SDF curve
        def update_hd_param(mob):
            eps = eps_tracker.get_value()
            # Clamp to range
            if eps <= epsilons[0]:
                interp_curve = hdsdf_vs_eps[0]
            elif eps >= epsilons[-1]:
                interp_curve = hdsdf_vs_eps[-1]
            else:
                # find index of the two nearest precomputed curves
                i = np.searchsorted(epsilons, eps) - 1
                i = max(0, min(i, len(epsilons)-2))
                t = (eps - epsilons[i]) / (epsilons[i+1] - epsilons[i])
                interp_curve = (1-t)*hdsdf_vs_eps[i] + t*hdsdf_vs_eps[i+1]
            # Convert to Manim line
            new_line = axes.plot_line_graph(
                t_array, interp_curve,
                line_color=BLUE, add_vertex_dots=False, stroke_width=3,
            )
            mob.become(new_line)
            # Update label
            epsilon_label.set_value(eps)

        hd_line.add_updater(update_hd_param)

        # Animate epsilon from its start to end (or a custom range)
        self.play(
            eps_tracker.animate.set_value(epsilons[-1]),
            run_time=6,
            rate_func=linear,
        )
        self.wait(1)
