# %%
import pickle
from manim import *
from textwrap import wrap
import numpy as np

premable = r"""
\usepackage{amsmath}
\usepackage{amssymb}
\usepackage{graphicx}
\usepackage{calc}
\makeatletter
\newcommand*{\bigboxplus}{%
  \DOTSB
  \mathop{\vphantom{\bigoplus}\mathpalette\matt@bigboxplus\relax}%
  \slimits@
}
\newcommand\matt@bigboxplus[2]{%
  \vcenter{\m@th\hbox{\resizebox{\widthof{$#1\bigoplus$}}{!}{$\boxplus$}}}%
}
\makeatother
"""
# Tell Manim to use this preamble for all TeX

config["tex_template"] = TexTemplate(
    documentclass=r"\documentclass[preview, varwidth=600px]{standalone}",
    preamble=premable,
)
CHAR_LIM = 120

esdf_data = "./experiment_data/mode_0/video_7_data/data.pickle"
hdsdf_data = "./experiment_data/mode_1/video_1_data/data.pickle"

with open(esdf_data, "rb") as f:
    esdf = pickle.load(f)

with open(hdsdf_data, "rb") as f:
    hdsdf = pickle.load(f)



def create_subtitle_box(text_lines):
    """Return (subtitle_rect, subtitle_text).
    text_lines: list of strings, e.g. ["Hello world", "Math: a^2+b^2=c^2"]
    Plain text lines are wrapped in \\text{} so they stay upright;
    LaTeX commands can be used directly.
    """
    # ── Background rectangle ─────────────────────────────────
    subtitle_rect = RoundedRectangle(
        corner_radius=0.2,
        width=config.frame_width - 0.5,
        height=1.0,
        fill_color="#3A3A3A",
        fill_opacity=0.6,
        stroke_width=0,
    )
    subtitle_rect.to_edge(DOWN, buff=0.2)

    # ── Build a single MathTex with line breaks ──────────────
    processed_lines = []
    for line in text_lines:
        # If the line already contains LaTeX (starts with \ or contains $),
        # leave it as is; otherwise wrap in \text{}.
        if line.startswith("\\") or "$" in line:
            processed_lines.append(line)
        else:
            processed_lines.append(r"\text{" + line + "}")

    full_tex = r" \\ ".join(processed_lines)
    full_tex = r" \\ ".join(text_lines)

    subtitle_text = Tex(
        full_tex,
        font_size=24,  # no scaling needed – just pick a visible size
        color=WHITE,
        tex_template=config["tex_template"],  # use your custom preamble if needed
    )
    subtitle_text.move_to(subtitle_rect)

    return subtitle_rect, subtitle_text


class SquareDistanceScene(MovingCameraScene):
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
        # y_shift = -2.0  # move everything up by 2.5 units

        V_square_history = V_square_history * scale
        V_square_history[..., 1] += y_shift

        V_pentagon = V_pentagon * scale
        V_pentagon[:, 1] += y_shift

        # ── static pentagon (3D points, z=0) ──────────────────────
        pentagon = Polygon(*V_pentagon, color=GREEN, stroke_width=3)
        pentagon.set_fill(GREEN, opacity=0.15)

        # ── moving square ─────────────────────────────────────────
        # Start with first frame, preserve style throughout
        square = Polygon(*V_square_history[0], color=RED, stroke_width=3)
        square.set_fill(RED, opacity=0.2)

        def update_square(mob):
            idx = int(round(time_tracker.get_value() / T_total * (N - 1)))
            idx = max(0, min(N - 1, idx))
            # Directly update vertex positions – no style loss
            mob.set_points_as_corners(
                np.vstack([V_square_history[idx], V_square_history[idx][0]])
            )

        square.add_updater(update_square)

        # ── graph axes (wider, bottom half) ───────────────────────
        all_hdsdf = [hdsdf]  # list of all HD‑SDF arrays we'll ever show
        if "hdsdf_vs_eps" in data:
            all_hdsdf.extend(list(data["hdsdf_vs_eps"]))
        hdy_min = min(np.min(arr) for arr in all_hdsdf)
        hdy_max = max(np.max(arr) for arr in all_hdsdf)
        y_min = min(hdy_min, esdf.min())
        y_max = max(hdy_max, esdf.max())
        y_margin = 0.05 * (y_max - y_min) or 0.1
        axes = Axes(
            x_range=[0, T_total, 0.5],
            y_range=[y_min - y_margin, y_max + y_margin, 0.1],
            x_length=config.frame_width * 0.8,  # fill most of the width
            y_length=4,
            axis_config={"include_numbers": True, "font_size": 20},
            tips=False,
        )
        axes.shift(DOWN * 0.5)  # keep it in the lower half
        # axes.shift(UP * 1.5)  # keep it in the upper half

        # Manually placed labels – Distance on top of y‑axis, Time at right of x‑axis
        dist_label = Tex("Distance", font_size=28)
        dist_label.next_to(axes.y_axis.get_top(), RIGHT, aligned_edge=LEFT, buff=0.35)

        time_label = Tex("Time (s)", font_size=28)
        time_label.next_to(
            axes.x_axis.get_right(), RIGHT + UP * 0.01, aligned_edge=LEFT, buff=0.1
        )

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

        # ── legend ────────────────────────────────────────────────
        legend_h = Tex("HD-SDF", color=BLUE).scale(0.5)
        legend_e = Tex("SDF", color=YELLOW).scale(0.5)
        legend = VGroup(legend_h, legend_e).arrange(DOWN, aligned_edge=RIGHT, buff=0.1)
        legend.next_to(
            axes.coords_to_point(axes.x_range[1], axes.y_range[1]),  # top‑right corner
            DL,
            buff=0.25,
        )

        # ── time tracker ──────────────────────────────────────────
        time_tracker = ValueTracker(0)

        # ── play ─────────────────────────────────────────────────
        # Bring everything in gracefully
        # Add silently – no animation

        par = (
            "The Euclidean SDF presents non‑differentiable points: 1) when the faces are parallel"
            " and no collision occurs; 2) when collision occurs and separating normals are not"
            " unique. Our HD‑SDF presents none of these issues."
        )

        subtitle_rect, subtitle_text = create_subtitle_box(
            wrap(par, CHAR_LIM, subsequent_indent=" ")
        )
        # self.add(subtitle_rect, subtitle_text)
        subtitle_text.set_opacity(0)
        subtitle_rect.set_opacity(0)
        subtitle_rect.set_z_index(-1)
        self.add(subtitle_text, subtitle_rect)
        self.play(
            subtitle_rect.animate.set_opacity(1),
            subtitle_text.animate.set_opacity(1),
            run_time=0.5,
        )
        # self.play(
        #     FadeIn(subtitle_text),
        #     FadeIn(subtitle_rect),
        #     run_time=0.5,
        # )
        self.play(
            DrawBorderThenFill(pentagon),
            DrawBorderThenFill(square),
            Create(axes),
            Create(dist_label),
            Create(time_label),
            Create(legend),
            Create(hd_line),
            Create(euc_line),
            run_time=2,
        )

        self.play(
            time_tracker.animate.set_value(T_total),
            # run_time=T_total,
            run_time=10,
            rate_func=linear,
        )
        self.wait(1)

        # ---------------------------------------------------------
        # ── phase 2: morph HD‑SDF (ε, then γ) ────────────────────
        # ---------------------------------------------------------
        # Remove Phase‑1 updaters
        hd_line.clear_updaters()
        euc_line.clear_updaters()

        # Distance equation with coloured epsilon / gamma
        eq = MathTex(
            r"D_{\gamma,\epsilon}(\mathcal{A},\mathcal{B}) \triangleq ",
            r"\Phi_{",
            r"\gamma",
            r",",
            r"\epsilon",
            r"}\left(",
            r"\bigboxplus_{n\in\tilde{\mathcal{N}}(\mathcal{A}, \mathcal{B})}",
            r"\Phi_{",
            r"\gamma",
            r",",
            r"\epsilon",
            r"}\left(",
            r"\bigoplus_{\substack{a\in \mathcal{V}(\mathcal{A})\\b\in \mathcal{V}(\mathcal{B})}}",
            r"n^\top(a-b)\right)\right)",
            tex_template=config["tex_template"],
        )
        eq.set_color_by_tex(r"\gamma", GREEN)  # all \gamma in purple
        eq.set_color_by_tex(r"\epsilon", RED)  # all \epsilon in blue
        eq.scale(0.7)
        eq.move_to(pentagon).shift(UP * 0.4)  # where the shapes used to be

        # ── Remove shapes, show equation ───────────────────────
        self.play(Uncreate(pentagon), Uncreate(square), run_time=1)
        self.remove(subtitle_text)
        par = (
            r"Differentiability of the HD-SDF depends on shaping function $\Phi$."
            r"A larger $\epsilon$ makes the function smoother, but it "
            "also increases the number of false collisions. "
            r"A smaller $\epsilon$ yields a more accurate distance, yet "
            "the derivative magnitude near zero becomes larger"
        )
        subtitle_rect, subtitle_text = create_subtitle_box(
            wrap(par, CHAR_LIM, subsequent_indent=" ")
        )
        self.add(subtitle_text)
        # self.wait(0.5)

        # ---- extended epsilon data (include default ε = 1e-3) ----
        epsilons_loaded = epsilons  # original sweep (1e-4 … 1e-2)
        hdsdf_vs_eps_loaded = hdsdf_vs_eps  # shape (P, N_frames)
        default_eps = 1e-3

        # ── Parameter labels (centered above the graph) ─────────
        label_y = axes.y_axis.get_top()[1]  # same height as Distance label
        graph_center_x = axes.get_center()[0]  # horizontal centre of axes

        # ε label
        eps_prefix = MathTex(r"\epsilon", r" = ", font_size=28)
        eps_prefix.set_color_by_tex(r"\epsilon", RED)  # colour the letter only
        eps_value_mob = DecimalNumber(default_eps, num_decimal_places=5, font_size=24)
        eps_label_group = VGroup(eps_prefix, eps_value_mob).arrange(RIGHT, buff=0.1)
        eps_label_group.move_to([graph_center_x * 0.8, label_y, 0])

        # γ label (will be updated later)
        gamma_prefix = Tex(r"$\gamma = $", font_size=28)
        gamma_prefix.set_color(GREEN)  # use the same green as in the equation
        gamma_value_mob = Integer(2, font_size=24)
        gamma_label_group = VGroup(gamma_prefix, gamma_value_mob).arrange(
            RIGHT, buff=0.1
        )
        gamma_label_group.next_to(eps_label_group, RIGHT, buff=0.1)

        self.play(
            Write(eq), Create(eps_label_group), Create(gamma_label_group), run_time=3
        )
        self.wait(0.5)
        # it will appear after ε finishes – we'll add it later

        # Insert default curve into the sorted array
        eps_ext = np.append(epsilons_loaded, default_eps)
        hd_ext = np.vstack([hdsdf_vs_eps_loaded, hdsdf[np.newaxis, :]])
        sort_idx = np.argsort(eps_ext)
        eps_ext = eps_ext[sort_idx]
        hd_ext = hd_ext[sort_idx]

        eps_tracker = ValueTracker(default_eps)
        # epsilon_label = DecimalNumber(
        #     default_eps, num_decimal_places=5, font_size=24, color=BLUE
        # )
        # epsilon_label.next_to(axes, UP, buff=0.3)
        # self.add(epsilon_label)

        def update_hd_epsilon(mob):
            eps = eps_tracker.get_value()
            if eps <= eps_ext[0]:
                curve = hd_ext[0]
            elif eps >= eps_ext[-1]:
                curve = hd_ext[-1]
            else:
                i = np.searchsorted(eps_ext, eps) - 1
                i = max(0, min(i, len(eps_ext) - 2))
                t = (eps - eps_ext[i]) / (eps_ext[i + 1] - eps_ext[i])
                curve = (1 - t) * hd_ext[i] + t * hd_ext[i + 1]
            new_line = axes.plot_line_graph(
                t_array,
                curve,
                line_color=BLUE,
                add_vertex_dots=False,
                stroke_width=3,
            )
            mob.become(new_line)
            eps_value_mob.set_value(eps)
            # epsilon_label.set_value(eps)

        hd_line.add_updater(update_hd_epsilon)

        # Highlight epsilon
        # eps_rect = SurroundingRectangle(eq[1], color=RED, buff=0.1)
        # eps_rect2 = SurroundingRectangle(eq[5], color=RED, buff=0.1)
        # To be safe, we can highlight all occurrences
        self.play(Indicate(eps_label_group), Indicate(eq[1]), Indicate(eq[7]))
        # self.play(Create(eps_rect), Create(eps_rect2), run_time=1)
        # self.wait(0.5)
        # self.play(FadeOut(eps_rect), FadeOut(eps_rect2))
        # Then proceed with epsilon sweep...

        # --- ε sequence: 1e-3 → 1e-4 → 1e-2 → 1e-3 ---
        self.play(eps_tracker.animate.set_value(eps_ext[-1]), run_time=5)
        self.play(eps_tracker.animate.set_value(eps_ext[0]), run_time=5)
        self.wait(0.5)
        self.play(eps_tracker.animate.set_value(default_eps), run_time=5)
        self.wait(0.5)

        # Remove ε label, add γ label
        # self.remove(epsilon_label)
        gamma_tracker = ValueTracker(2)  # default γ = 2
        gamma_label = Integer(2, font_size=24, color=PURPLE)
        gamma_label.next_to(axes, UP, buff=0.3)
        # Swap to γ label
        # Ensure the gamma_value_mob starts at 2 (it already does)
        # self.add(gamma_label)

        def update_hd_gamma(mob):
            gam = gamma_tracker.get_value()
            if gam <= gammas[0]:
                curve = hdsdf_vs_gamma[0]
            elif gam >= gammas[-1]:
                curve = hdsdf_vs_gamma[-1]
            else:
                i = np.searchsorted(gammas, gam) - 1
                i = max(0, min(i, len(gammas) - 2))
                t = (gam - gammas[i]) / (gammas[i + 1] - gammas[i])
                curve = (1 - t) * hdsdf_vs_gamma[i] + t * hdsdf_vs_gamma[i + 1]
            new_line = axes.plot_line_graph(
                t_array,
                curve,
                line_color=BLUE,
                add_vertex_dots=False,
                stroke_width=3,
            )
            mob.become(new_line)
            gamma_value_mob.set_value(int(round(gam)))
            # gamma_label.set_value(int(round(gam)))  # show integer

        hd_line.clear_updaters()
        hd_line.add_updater(update_hd_gamma)

        self.remove(subtitle_text)
        par = (
            r"While $\gamma$ controls how many times differentiable the distance is, "
            r"it also controls how well the Hölder minimum approximates the true minimum. "
            r"In turn, this increases the size of the neighborhood that is mapped to $0$."
        )
        subtitle_rect, subtitle_text = create_subtitle_box(
            wrap(par, CHAR_LIM, subsequent_indent=" ")
        )
        self.add(subtitle_text)

        # self.play(Create(gam_rect_group), run_time=1)
        self.play(
            Indicate(gamma_label_group),
            Indicate(eq[1]),
            Indicate(eq[6]),
            Indicate(eq[7]),
            Indicate(eq[12]),
        )
        # --- γ sequence: 2 → 1 → 10 ---
        self.play(gamma_tracker.animate.set_value(1), run_time=2)
        self.play(gamma_tracker.animate.set_value(5), run_time=6)
        self.wait(1)

        # ---------------------------------------------------------
        # ── phase 3: Real‑world experiment (split‑screen) ────────
        # ---------------------------------------------------------
        # 1.  Clean up previous phase (shapes, graph, parameter labels)
        self.play(
            FadeOut(axes),
            FadeOut(hd_line),
            FadeOut(euc_line),
            FadeOut(dist_label),
            FadeOut(time_label),
            FadeOut(eps_label_group),
            FadeOut(gamma_label_group),
            FadeOut(legend),
            FadeOut(eq)
        )
        self.remove(subtitle_text, subtitle_rect)

        # 3.  Build the optimization problem (common formulation)
        eq_font_size = 30
        opt = MathTex(
            r"\min_{u} &\left\|\frac{\partial r}{\partial q}(q)u + Kr(q)\right\|^2 + \lambda\|u\|^2 \\"
            r"\text{s.t.:} & \\&\begin{aligned}"
            r"\frac{\partial \Delta_{ij}^{\text{obs}}}{\partial q}(q) \, u &\geq -\eta_{\text{obs}}\bigl(\Delta_{ij}^{\text{obs}}(q) - \delta_{\text{obs}}\bigr) \\"
            r"\frac{\partial \Delta_{ij}^{\text{auto}}}{\partial q}(q) \, u &\geq -\eta_{\text{auto}}\bigl(\Delta_{ij}^{\text{auto}}(q) - \delta_{\text{auto}}\bigr) \\"
            r"u &\ge -\eta_{\text{joint}}(q - q_{\text{min}}) \\"
            r"-u &\ge -\eta_{\text{joint}}(q_{\text{max}} - q) \\"
            r"\end{aligned}",
            tex_template=config["tex_template"],
            font_size=eq_font_size,
        )
        opt.to_corner(UL, buff=0.3)  # left side of the screen
        self.play(Write(opt), run_time=3)

        # 4.  Create two “case” rectangles on the right
        case_rect_w = config.frame_width * 0.45
        case_rect_h = config.frame_height * 0.35
        euc_rect = Rectangle(
            width=case_rect_w,
            height=case_rect_h,
            color=GREY,
            stroke_width=1,
            fill_opacity=0.1,
        )
        hd_rect = Rectangle(
            width=case_rect_w,
            height=case_rect_h,
            color=GREY,
            stroke_width=1,
            fill_opacity=0.1,
        )
        # Stack them vertically on the right side
        VGroup(euc_rect, hd_rect).arrange(DOWN, buff=0.5)
        VGroup(euc_rect, hd_rect).to_edge(RIGHT, buff=0.3).shift(UP * 0.5)
        euc_label = Tex("$\Delta=$ Euclidean distance", font_size=24).next_to(euc_rect, UP, buff=0.1)
        hd_label = Tex("$\Delta=$ HD‑SDF", font_size=24).next_to(hd_rect, UP, buff=0.1)

        self.play(
            Create(euc_rect),
            Create(hd_rect),
            Write(euc_label),
            Write(hd_label),
            run_time=2,
        )

        # 5.  Populate common numeric values (λ, K, η…)
        opt_numeric = MathTex(
            r"\min_{u} &\left\|\frac{\partial r}{\partial q}(q)u + 0.4\,I\,r(q)\right\|^2 + 0.01\|u\|^2 \\",
            r"\text{s.t.:} & \\",
            r"&\begin{aligned}",
            r"\frac{\partial \Delta_{ij}^{\text{obs}}}{\partial q}(q) \, u &\geq -0.5\bigl(\Delta_{ij}^{\text{obs}}(q) - \delta_{\text{obs}}\bigr) \\",
            r"\frac{\partial \Delta_{ij}^{\text{auto}}}{\partial q}(q) \, u &\geq -0.5\bigl(\Delta_{ij}^{\text{auto}}(q) - \delta_{\text{auto}}\bigr) \\",
            r"u &\ge -0.5(q - q_{\text{min}}) \\",
            r"-u &\ge -0.5(q_{\text{max}} - q)",
            r"\end{aligned}",
            tex_template=config["tex_template"],
            font_size=eq_font_size,
        )
        opt_numeric.move_to(opt)
        self.play(TransformMatchingTex(opt, opt_numeric), run_time=3)
        self.wait(0.5)

        # 6.  Show the Euclidean‑case specific parameters
        euc_params = MathTex(
            r"\delta_{\text{obs}} &= 0.05\\ \delta_{\text{auto}} &= 0.05", font_size=24
        )
        euc_params.move_to(euc_rect.get_center() + UP * 0.3)
        self.play(Write(euc_params), run_time=1.5)
        self.wait(0.5)

        # ── subtitle: “First, the Euclidean baseline” ────────────
        sub_rect, sub_text = create_subtitle_box(
            ["Euclidean baseline: positive safety margins, real obstacle geometry."]
        )
        self.add(sub_rect, sub_text)

        # 7.  Zoom into the Euclidean case rectangle
        # Save current camera state
        orig_frame_center = self.camera.frame.get_center()
        orig_frame_width = self.camera.frame.get_width()
        target_area = euc_rect
        self.play(
            self.camera.frame.animate.scale(0.45).move_to(target_area.get_center()),
            run_time=1.5,
        )
        self.wait(0.3)

        # Remove the subtitle (it's now off‑screen) and add a new one
        self.remove(sub_rect, sub_text)
        sub_rect_zoom, sub_text_zoom = create_subtitle_box(
            ["Experiment: robot avoids obstacles using Euclidean SDF."]
        )
        self.add(sub_rect_zoom, sub_text_zoom)


        # ----------------------- LOAD DATa --------------------------
        hdsdf_time_ = hdsdf["timestamp"]
        t0 = hdsdf_time_[0]
        hdsdf_time = np.array([t - t0 for t in hdsdf_time_])
        hdsdf_u = np.array(hdsdf["hist_dq"]).reshape(-1, 7)

        esdf_time_ = esdf["timestamp"]
        t0 = esdf_time_[0]
        esdf_time = np.array([t - t0 for t in esdf_time_])
        esdf_u = np.array(esdf["hist_dq"]).reshape(-1, 7)

        # ── Placeholder for experiment video ────────────────────
        video_ph = Rectangle(
            width=case_rect_w * 0.8,
            height=case_rect_h * 0.8,
            color=GREY,
            stroke_width=1,
            fill_opacity=0.2,
        )
        video_ph_label = Text("Experiment\nvideo", font_size=20, color=GREY)
        video_ph_label.move_to(video_ph)
        video_ph.move_to(target_area.get_center()).shift(UP * 0.1)

        # ── Placeholder control graph (growing line) ────────────
        graph_placeholder_axes = Axes(
            x_range=[0, 5, 1],
            y_range=[-1, 1, 0.5],
            x_length=case_rect_w * 0.6,
            y_length=case_rect_h * 0.3,
            tips=False,
        ).next_to(video_ph, DOWN, buff=0.2)

        # Dummy control curve (you will replace with real data)
        dummy_t = np.linspace(0, 5, 100)
        dummy_ctrl = np.sin(dummy_t) * 0.5
        ctrl_line = axes.plot_line_graph(
            dummy_t[:1],
            dummy_ctrl[:1],
            line_color=BLUE,
            add_vertex_dots=False,
            stroke_width=2,
        )
        graph_placeholder_axes.add(ctrl_line)

        def update_ctrl(mob):
            # gradually reveal the whole curve
            mob.clear_updaters()
            mob.become(
                axes.plot_line_graph(
                    dummy_t,
                    dummy_ctrl,
                    line_color=BLUE,
                    add_vertex_dots=False,
                    stroke_width=2,
                )
            )

        ctrl_line.add_updater(update_ctrl)

        self.play(
            Create(video_ph),
            Write(video_ph_label),
            Create(graph_placeholder_axes),
            run_time=2,
        )
        self.wait(5)  # simulate the experiment duration
        self.play(FadeOut(video_ph), FadeOut(video_ph_label))

        # 8.  Zoom back out to the split screen
        self.play(
            self.camera.frame.animate.scale(1 / 0.45).move_to(orig_frame_center),
            run_time=1.5,
        )
        self.remove(sub_rect_zoom, sub_text_zoom)
        # Restore original subtitle area? We'll add new ones later.
        # Remove the placeholder graph axes as they are no longer needed
        self.remove(graph_placeholder_axes, ctrl_line)

        # 9.  Now populate the HD‑SDF case parameters
        hd_params = MathTex(
            r"\delta_{\text{obs}} = -0.01,\; \delta_{\text{auto}} = -0.01",
            r"\text{(expanded obstacles)}",
            font_size=24,
        )
        hd_params.move_to(hd_rect.get_center() + UP * 0.3)
        self.play(Write(hd_params), run_time=1.5)

        sub_rect2, sub_text2 = create_subtitle_box(
            ["HD‑SDF: negative safety margins, virtually expanded obstacles."]
        )
        self.add(sub_rect2, sub_text2)

        # 10. Zoom into HD‑SDF case
        self.play(
            self.camera.frame.animate.scale(0.45).move_to(hd_rect.get_center()),
            run_time=1.5,
        )
        self.remove(sub_rect2, sub_text2)
        sub_rect_zoom2, sub_text_zoom2 = create_subtitle_box(
            ["Experiment: robot avoids obstacles using HD‑SDF."]
        )
        self.add(sub_rect_zoom2, sub_text_zoom2)

        # Placeholder video and graph (similar to above)
        video_ph2 = video_ph.copy().move_to(hd_rect.get_center()).shift(UP * 0.1)
        video_ph_label2 = video_ph_label.copy().move_to(video_ph2)
        graph_placeholder_axes2 = graph_placeholder_axes.copy().next_to(
            video_ph2, DOWN, buff=0.2
        )
        dummy_ctrl2 = np.sin(dummy_t) * 0.8  # slightly different for contrast
        ctrl_line2 = axes.plot_line_graph(
            dummy_t[:1],
            dummy_ctrl2[:1],
            line_color=RED,
            add_vertex_dots=False,
            stroke_width=2,
        )
        graph_placeholder_axes2.add(ctrl_line2)

        def update_ctrl2(mob):
            mob.clear_updaters()
            mob.become(
                axes.plot_line_graph(
                    dummy_t,
                    dummy_ctrl2,
                    line_color=RED,
                    add_vertex_dots=False,
                    stroke_width=2,
                )
            )

        ctrl_line2.add_updater(update_ctrl2)

        self.play(
            Create(video_ph2),
            Write(video_ph_label2),
            Create(graph_placeholder_axes2),
            run_time=2,
        )
        self.wait(5)
        self.play(FadeOut(video_ph2), FadeOut(video_ph_label2))

        # 11. Zoom out again
        self.play(
            self.camera.frame.animate.scale(1 / 0.45).move_to(orig_frame_center),
            run_time=1.5,
        )
        self.remove(sub_rect_zoom2, sub_text_zoom2)
        self.remove(graph_placeholder_axes2, ctrl_line2)

        # 12. Final comparison: remove the optimization equation,
        #     expand both control graphs to fill the screen side‑by‑side
        self.play(FadeOut(opt_numeric), FadeOut(eq))
        # Remove case rectangles and labels
        self.play(
            FadeOut(euc_rect),
            FadeOut(hd_rect),
            FadeOut(euc_label),
            FadeOut(hd_label),
            FadeOut(euc_params),
            FadeOut(hd_params),
        )

        # Create two large axes: one for Euclidean, one for HD‑SDF
        big_axes_euc = Axes(
            x_range=[0, 5, 1],
            y_range=[-1, 1, 0.5],
            x_length=config.frame_width * 0.45,
            y_length=config.frame_height * 0.8,
            tips=False,
        ).to_edge(LEFT, buff=0.5)
        big_axes_hd = Axes(
            x_range=[0, 5, 1],
            y_range=[-1, 1, 0.5],
            x_length=config.frame_width * 0.45,
            y_length=config.frame_height * 0.8,
            tips=False,
        ).to_edge(RIGHT, buff=0.5)

        euc_curve = axes.plot_line_graph(
            dummy_t, dummy_ctrl, line_color=BLUE, add_vertex_dots=False, stroke_width=3
        )
        hd_curve = axes.plot_line_graph(
            dummy_t, dummy_ctrl2, line_color=RED, add_vertex_dots=False, stroke_width=3
        )
        euc_label_final = Tex("Euclidean", font_size=24).next_to(
            big_axes_euc, UP, buff=0.1
        )
        hd_label_final = Tex("HD‑SDF", font_size=24).next_to(big_axes_hd, UP, buff=0.1)

        self.play(
            Create(big_axes_euc),
            Create(big_axes_hd),
            Write(euc_label_final),
            Write(hd_label_final),
            run_time=2,
        )
        self.play(Create(euc_curve), Create(hd_curve), run_time=3)
        self.wait(2)

        # Keep the final comparison visible for a moment

        # # ---------------------------------------------------------
        # # ── phase 3: morph HD‑SDF (ε, then γ) ────────────────────
        # # ---------------------------------------------------------
        #
        # self.remove(subtitle_text)
        # par = (
        #     "While the non-differentiable points of the Euclidean distance "
        #     "might seem harmless, a real-world experiment demonstrates "
        #     "why that's not the case."
        # )
        # subtitle_rect, subtitle_text = create_subtitle_box(
        #     wrap(par, CHAR_LIM, subsequent_indent=" ")
        # )
        # self.add(subtitle_text)
        #
        # # Fade out graph, shapes, parameter labels – keep only the distance equation
        # self.play(
        #     FadeOut(axes),
        #     FadeOut(hd_line),
        #     FadeOut(euc_line),
        #     FadeOut(dist_label),
        #     FadeOut(time_label),
        #     FadeOut(eps_label_group),
        #     FadeOut(gamma_label_group),
        # )
        # # Move distance eq to top-left
        # self.play(eq.animate.scale(0.7).to_corner(UR))
        #
        # eq_font_size = 32
        # opt = MathTex(
        #     r"\min_{u} &\left\|\frac{\partial r}{\partial q}(q)u + Kr(q)\right\|^2 + \lambda\|u\|^2 \\"
        #     r"\text{s.t.:} & \\&\begin{aligned}"
        #     r"\frac{\partial \Delta_{ij}^{\text{obs}}}{\partial q}(q) \, u &\geq -\eta_{\text{obs}}\bigl(\Delta_{ij}^{\text{obs}}(q) - \delta_{\text{obs}}\bigr) \\"
        #     r"\frac{\partial \Delta_{ij}^{\text{auto}}}{\partial q}(q) \, u &\geq -\eta_{\text{auto}}\bigl(\Delta_{ij}^{\text{auto}}(q) - \delta_{\text{auto}}\bigr) \\"
        #     r"u &\ge -\eta_{\text{joint}}(q - q_{\text{min}}) \\"
        #     r"-u &\ge -\eta_{\text{joint}}(q_{\text{max}} - q) \\"
        #     r"\end{aligned}",
        #     tex_template=config["tex_template"],
        #     font_size=eq_font_size,
        # )
        # self.play(Write(opt), run_time=1)
        # delta_parts = VGroup()
        # for part in opt:
        #     if "Delta" in part.get_tex_string():
        #         delta_parts.add(part)
        # if delta_parts:
        #     self.play(Circumscribe(delta_parts, color=YELLOW, fade_out=True))
        # # Draw an arrow from delta_parts to the distance equation
        # arrow = Arrow(delta_parts.get_center(), eq.get_center(), color=YELLOW)
        # self.play(Create(arrow))
        #
        # opt_numeric = MathTex(
        #     r"\min_{u} &\left\|\frac{\partial r}{\partial q}(q)u + 0.4Ir(q)\right\|^2 + 0.01\|u\|^2 \\"
        #     r"\text{s.t.:} & \\&\begin{aligned}"
        #     r"\frac{\partial \Delta_{ij}^{\text{obs}}}{\partial q}(q) \, u &\geq -0.5\bigl(\Delta_{ij}^{\text{obs}}(q) - \delta_{\text{obs}}\bigr) \\"
        #     r"\frac{\partial \Delta_{ij}^{\text{auto}}}{\partial q}(q) \, u &\geq -0.5\bigl(\Delta_{ij}^{\text{auto}}(q) - \delta_{\text{auto}}\bigr) \\"
        #     r"u &\ge -0.5(q - q_{\text{min}}) \\"
        #     r"-u &\ge -0.5(q_{\text{max}} - q) \\"
        #     r"\end{aligned}",
        #     tex_template=config["tex_template"],
        #     font_size=eq_font_size,
        # )
        # opt_numeric.move_to(opt)
        # self.play(TransformMatchingShapes(opt, opt_numeric), run_time=3)
