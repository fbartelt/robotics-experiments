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

base_path = "/home/fbartelt/Documents/Projetos/robotics-experiments/smoothdist"
esdf_data = f"{base_path}/experiment/data/mode_0/video_7_data/data.pickle"
hdsdf_data = f"{base_path}/experiment/data/mode_1/video_1_data/data.pickle"

with open(esdf_data, "rb") as f:
    esdf = pickle.load(f)

with open(hdsdf_data, "rb") as f:
    hdsdf = pickle.load(f)

# ----------------------- Real control data (Euclidean) -----------------------
hdsdf_time_ = hdsdf["timestamp"]
t0 = hdsdf_time_[0]
hdsdf_time = np.array([t - t0 for t in hdsdf_time_])
hdsdf_u = np.array(hdsdf["hist_dq"]).reshape(-1, 7)

esdf_time_ = esdf["timestamp"]
t0 = esdf_time_[0]
esdf_time = np.array([t - t0 for t in esdf_time_])
esdf_u = np.array(esdf["hist_dq"]).reshape(-1, 7)


SUBTITLE_HEIGHT = 1.0


class Remove(Animation):
    """Instant removal of a mobject from the scene (no visual effect)."""

    def __init__(self, mobject, **kwargs):
        super().__init__(mobject, **kwargs)

    def interpolate(self, alpha):
        pass

    def finish(self):
        super().finish()
        if hasattr(self.mobject, "scene") and self.mobject.scene is not None:
            self.mobject.scene.remove(self.mobject)


def create_subtitle_box(
    text_lines,
    corner_radius=0.2,
    width=config.frame_width - 0.5,
    height=SUBTITLE_HEIGHT,
    fill_color="#3A3A3A",
    fill_opacity=0.6,
    stroke_width=0,
    font_size=24,
    text_color=WHITE,
):
    """Return (subtitle_rect, subtitle_text).
    text_lines: list of strings, e.g. ["Hello world", "Math: a^2+b^2=c^2"]
    Plain text lines are wrapped in \\text{} so they stay upright;
    LaTeX commands can be used directly.
    """
    # ── Background rectangle ─────────────────────────────────
    subtitle_rect = RoundedRectangle(
        corner_radius=corner_radius,
        width=width,
        height=height,
        fill_color=fill_color,
        fill_opacity=fill_opacity,
        stroke_width=stroke_width,
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
        font_size=font_size,  # no scaling needed – just pick a visible size
        color=text_color,
        tex_template=config["tex_template"],  # use your custom preamble if needed
    )
    subtitle_text.move_to(subtitle_rect)

    return subtitle_rect, subtitle_text


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
            "The Euclidean SDF exhibits points of non‑differentiability"
            ": 1) when faces are parallel, and 2) when separating normals are not"
            " unique. Our HD‑SDF is differentiable everywhere."
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
        self.wait(2)

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
        par = r"Differentiability of the HD-SDF depends on the shaping function $\Phi$."
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
            Write(eq), Create(eps_label_group), Create(gamma_label_group), run_time=4
        )
        self.wait(3)
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
        # self.play(Create(eps_rect), Create(eps_rect2), run_time=1)
        # self.wait(0.5)
        # self.play(FadeOut(eps_rect), FadeOut(eps_rect2))
        # Then proceed with epsilon sweep...

        # --- ε sequence: 1e-3 → 1e-4 → 1e-2 → 1e-3 ---
        self.remove(subtitle_text)
        par = (
            r"A larger $\epsilon$ makes the function smoother, but it "
            "also increases the number of false collisions."
        )
        subtitle_rect, subtitle_text = create_subtitle_box(
            wrap(par, CHAR_LIM, subsequent_indent=" ")
        )
        self.add(subtitle_text)

        self.play(Indicate(eps_label_group), Indicate(eq[1]), Indicate(eq[7]))
        self.play(eps_tracker.animate.set_value(eps_ext[-1]), run_time=6)
        self.wait(2)

        self.remove(subtitle_text)
        par = (
            r"A smaller $\epsilon$ yields a more accurate distance, yet "
            "the derivative magnitude near zero becomes larger."
        )
        subtitle_rect, subtitle_text = create_subtitle_box(
            wrap(par, CHAR_LIM, subsequent_indent=" ")
        )
        self.add(subtitle_text)

        self.play(eps_tracker.animate.set_value(eps_ext[0]), run_time=6)
        self.wait(3)

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
            r"The parameter $\gamma$ controls how many times differentiable "
            "the distance is, and also how well the Hölder minimum approximates "
            "the true minimum. In turn, this enlarges the neighborhood that is mapped to zero."
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
        self.wait(2)
        self.play(gamma_tracker.animate.set_value(5), run_time=8)
        self.wait(3)

        self.remove(subtitle_text)
        par = (
            "While the aforementioned non-differentiable points might "
            "seem harmless, the real experiment shows the contrary"
        )
        subtitle_rect, subtitle_text = create_subtitle_box(
            wrap(par, CHAR_LIM, subsequent_indent=" ")
        )
        self.add(subtitle_text)
        self.wait(6)

        # 1.  Clean up previous phase (shapes, graph, parameter labels)
        self.play(
            FadeOut(axes),
            Uncreate(hd_line),
            Uncreate(euc_line),
            Unwrite(dist_label),
            Unwrite(time_label),
            Unwrite(eps_label_group),
            FadeOut(gamma_label_group),
            Unwrite(legend),
            Unwrite(eq),
            run_time=1,
        )
        self.remove(subtitle_text, subtitle_rect)
        self.wait(0.5)


class ExperimentScene(MovingCameraScene):
    def construct(self):
        # ---------------------------------------------------------
        # ── phase 3: Real‑world experiment (split‑screen) ────────
        # ---------------------------------------------------------
        # 3.  Build the optimization problem (common formulation)
        par = (
            "We formulate pose regulation as a quadratic program with "
            r"control barrier function (CBF) constraints, where $\Delta$ "
            "is either the Euclidean distance or the HD‑SDF."
        )
        sub_rect, sub_text = create_subtitle_box(wrap(par, CHAR_LIM))
        sub_text.set_opacity(1)
        sub_rect.set_opacity(1)
        sub_rect.set_z_index(-1)
        sub_text.set_z_index(1)
        self.add(sub_rect, sub_text)

        eq_font_size = 30
        opt = MathTex(
            r"\min_{u} &\left\|\frac{\partial r}{\partial q}(q)u + Kr(q)\right\|^2 + \lambda\|u\|^2 \\",
            r"\text{s.t.:} & \\&\begin{aligned}",
            r"\frac{\partial \Delta_{ij}^{\text{obs}}}{\partial q}(q)  u &\geq -\eta_{\text{obs}}\bigl(\Delta_{ij}^{\text{obs}}(q) - \delta_{\text{obs}}\bigr) \\",
            r"\frac{\partial \Delta_{ij}^{\text{self}}}{\partial q}(q)  u &\geq -\eta_{\text{self}}\bigl(\Delta_{ij}^{\text{self}}(q) - \delta_{\text{self}}\bigr) \\",
            r"u &\ge -\eta_{\text{joint}}(q - q_{\text{min}}) \\",
            r"-u &\ge -\eta_{\text{joint}}(q_{\text{max}} - q) \\",
            r"\end{aligned}\\",
            r"r(q)&=\begin{bmatrix}p_{\text{eef}} - p_{\text{des}}\\"
            r"1 - x_{\text{eef}}^\top x_{\text{des}}\\"
            r"1 - y_{\text{eef}}^\top y_{\text{des}}\\"
            r"1 - z_{\text{eef}}^\top z_{\text{des}}"
            r"\end{bmatrix}",
            tex_template=config["tex_template"],
            font_size=eq_font_size,
        )
        opt.to_corner(UL, buff=0.3)  # left side of the screen

        # 4.  Create two “case” rectangles on the right
        rect_ratio = 0.45
        case_rect_w = config.frame_width * 0.5
        case_rect_h = (config.frame_height - SUBTITLE_HEIGHT) * 0.4
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
        euc_label = Tex(r"$\Delta=$ Euclidean distance", font_size=24).next_to(
            euc_rect, UP, buff=0.1
        )
        hd_label = Tex(r"$\Delta=$ HD‑SDF", font_size=24).next_to(hd_rect, UP, buff=0.1)

        self.play(
            Write(opt),
            Create(euc_rect),
            Create(hd_rect),
            Write(euc_label),
            Write(hd_label),
            run_time=6,
        )
        self.wait(6)

        self.remove(sub_text)
        par = (
            r"The distance $\Delta$ is used to prevent collisions with "
            r"obstacles and self‑collisions. Each case uses a different "
            r"safety margin $\delta$."
        )
        _, sub_text = create_subtitle_box(wrap(par, CHAR_LIM))
        self.add(sub_text)

        self.wait(4)
        self.play(
            Circumscribe(opt[2:4]),
            run_time=2,
        )
        self.wait(4)

        # 5.  Populate common numeric values (λ, K, η…)
        opt_numeric = MathTex(
            r"\min_{u} &\left\|\frac{\partial r}{\partial q}(q)u + 0.4Ir(q)\right\|^2 + 0.01\|u\|^2 \\",
            r"\text{s.t.:} & \\",
            r"&\begin{aligned}",
            r"\frac{\partial \Delta_{ij}^{\text{obs}}}{\partial q}(q)  u &\geq -0.5\bigl(\Delta_{ij}^{\text{obs}}(q) - ",
            r"\delta_{\text{obs}}",
            r"\bigr) \\",
            r"\frac{\partial \Delta_{ij}^{\text{self}}}{\partial q}(q)  u &\geq -0.5\bigl(\Delta_{ij}^{\text{self}}(q) - ",
            r"\delta_{\text{self}}",
            r"\bigr) \\",
            r"u &\ge -0.5(q - q_{\text{min}}) \\",
            r"-u &\ge -0.5(q_{\text{max}} - q)",
            r"\end{aligned}\\",
            r"r(q)&=\begin{bmatrix}p_{\text{eef}} - p_{\text{des}}\\"
            r"1 - x_{\text{eef}}^\top x_{\text{des}}\\"
            r"1 - y_{\text{eef}}^\top y_{\text{des}}\\"
            r"1 - z_{\text{eef}}^\top z_{\text{des}}\\"
            r"\end{bmatrix}",
            tex_template=config["tex_template"],
            font_size=eq_font_size,
        )

        self.remove(sub_text)
        par = (
            r"Joint limits $q_{\text{min}}, q_{\text{max}}$ are shared "
            r"between each case, as well as control gains and CBF tuning parameters."
        )
        _, sub_text = create_subtitle_box(wrap(par, CHAR_LIM))
        self.add(sub_text)

        opt_numeric.move_to(opt)
        self.wait(2)
        self.play(
            LaggedStart(
                Circumscribe(opt[4:6]),
                TransformMatchingShapes(opt, opt_numeric),
                lag_ratio=1.5,
                run_time=3,
            ),
            # TransformMatchingTex(Group(opt, variables), opt_numeric, lag_ratio=0.5),
        )
        self.wait(4)

        # 6.  Show the Euclidean‑case specific parameters
        euc_params = MathTex(
            r"\delta_{\text{obs}} &= 0.03\\ \delta_{\text{self}} &= 0.01", font_size=24
        )
        euc_params.move_to(euc_rect.get_center() + UP * 0.3)

        self.remove(sub_text)
        par = r"In the Euclidean case, the safety margins were set to these positive values."
        _, sub_text = create_subtitle_box(wrap(par, CHAR_LIM))
        self.add(sub_text)

        self.play(
            Indicate(opt_numeric[4]),
            Indicate(opt_numeric[7]),
            Write(euc_params),
            run_time=2,
        )
        self.wait(4)

        # ── subtitle: “First, the Euclidean baseline” ────────────
        self.remove(sub_text)

        # 7.  Zoom into the Euclidean case rectangle
        # Save current camera state
        orig_frame_center = self.camera.frame.get_center()
        orig_frame_width = self.camera.frame.get_width()
        orig_frame = self.camera.frame
        self.camera.frame.save_state()

        target_area = euc_rect
        self.play(
            FadeOut(euc_params),
            self.camera.frame.animate.replace(euc_rect, stretch=True),
            # self.camera.auto_zoom(euc_rect, margin=0.0, animate=True),
            run_time=1.5,
        )
        self.wait(0.3)

        # Remove the subtitle (it's now off‑screen) and add a new one
        self.remove(sub_text)
        # subtitle box sized for the zoomed‑in frame
        zoomed_width = self.camera.frame.get_width()

        sub_rect_height = 0.2
        sub_rect_font_size = 10
        sub_rect_width = zoomed_width * 0.95
        sub_rect_up_ratio = 0.01

        # ----------------- TODO
        self.remove(sub_text)
        par = "Both cases share the same setup: "
        sub_rect_zoom, sub_text_zoom = create_subtitle_box(
            wrap(par, CHAR_LIM + 16),
            corner_radius=0.05,
            width=sub_rect_width,
            height=sub_rect_height,
            font_size=sub_rect_font_size,
        )
        sub_rect_zoom.set_z_index(-1)
        sub_text_zoom.set_z_index(2)
        sub_rect_zoom.match_y(euc_rect, DOWN).match_x(euc_rect).shift(
            UP * sub_rect_up_ratio
        )
        sub_text_zoom.move_to(sub_rect_zoom)
        self.add(sub_rect_zoom, sub_text_zoom)
        self.wait(3)

        self.remove(sub_text_zoom)
        par = (
            r"The robot must reach a target pose inside a corridor "
            "of obstacles, while avoiding collisions with them and "
            "with itself."
        )
        _, sub_text_zoom = create_subtitle_box(
            wrap(par, CHAR_LIM + 16),
            corner_radius=0.05,
            width=sub_rect_width,
            height=sub_rect_height,
            font_size=sub_rect_font_size,
        )
        sub_text_zoom.set_z_index(2)
        sub_text_zoom.move_to(sub_rect_zoom)
        self.wait(9)

        self.remove(sub_text_zoom)
        par = "All robot links are modeled as box primitives."
        _, sub_text_zoom = create_subtitle_box(
            wrap(par, CHAR_LIM + 16),
            corner_radius=0.05,
            width=sub_rect_width,
            height=sub_rect_height,
            font_size=sub_rect_font_size,
        )
        sub_text_zoom.set_z_index(2)
        sub_text_zoom.move_to(sub_rect_zoom)

        self.add(sub_text_zoom)
        self.wait(4)
        # -------------------------
        # These will come in sequence during a graph animation
        self.remove(sub_text_zoom)
        par_prev = (
            "The animation is reconstructed from the experiment data "
            "and lets us view the scene from different angles."
        )
        par = (
            "Although the pose converges, the control input (joint "
            "velocities) exhibits very high frequency oscillations."
        )
        _, sub_text_zoom_prev = create_subtitle_box(
            wrap(par_prev, CHAR_LIM + 16),
            corner_radius=0.05,
            width=sub_rect_width,
            height=sub_rect_height,
            font_size=sub_rect_font_size,
        )
        _, sub_text_zoom = create_subtitle_box(
            wrap(par, CHAR_LIM + 16),
            corner_radius=0.05,
            width=sub_rect_width,
            height=sub_rect_height,
            font_size=sub_rect_font_size,
        )
        _, sub_text_ = create_subtitle_box(wrap(par_prev, CHAR_LIM))
        sub_text_zoom_prev.set_z_index(2)
        sub_text_zoom.set_z_index(2)
        # self.add(sub_rect_zoom, sub_text_zoom)

        # Graph axes (bottom half)
        y_data = np.round(esdf_u, 2)
        y_min_val = y_data.min()
        y_max_val = y_data.max()
        margin = (
            0.1 * (y_max_val - y_min_val) if (y_max_val - y_min_val) > 1e-6 else 0.1
        )
        x_min, x_max = esdf_time.min(), esdf_time.max()
        mini_graph_height = 0.9 * (case_rect_h - sub_rect_height) / 2

        graph_axes = Axes(
            x_range=[x_min, x_max, max((x_max - x_min) / 4, 0.1)],
            y_range=[
                y_min_val - margin,
                y_max_val + margin,
                max((y_max_val - y_min_val) / 5, 0.05),
            ],
            x_length=case_rect_w * 0.8,
            y_length=mini_graph_height,
            tips=False,
            axis_config={
                "include_numbers": True,
                "font_size": 12,
                "tick_size": 0.05,
                "stroke_width": 1,
            },
            x_axis_config={"decimal_number_config": {"num_decimal_places": 2}},
            y_axis_config={"decimal_number_config": {"num_decimal_places": 2}},
        )
        graph_axes.next_to(sub_rect_zoom, UP, buff=0.1, aligned_edge=DOWN)

        # 7 coloured curves (one per joint)
        colors = [RED, BLUE, GREEN, YELLOW, ORANGE, PURPLE, PINK]
        curves = VGroup()
        for j in range(7):
            curve = graph_axes.plot_line_graph(
                esdf_time,
                esdf_u[:, j],
                line_color=colors[j],
                add_vertex_dots=False,
                stroke_width=1,
            )
            curves.add(curve)

        # Legend: \dot{q}_i inside the graph area
        legend_items = VGroup()
        for j in range(7):
            item = MathTex(
                f"\\dot{{q}}_{{{j+1}}}", color=colors[j], font_size=sub_rect_font_size
            )
            legend_items.add(item)
        # Legend repositioned to top‑right corner of the small graph
        legend_items.arrange(RIGHT, buff=0.08)
        top_right_point = graph_axes.coords_to_point(esdf_time.max(), esdf_u.max())
        legend_items.next_to(top_right_point, UL, buff=0.05)

        # Ensure everything fits inside the zoomed rectangle
        # (the axes are already inside because they were placed relative to video_ph)

        self.play(
            Create(graph_axes),
            Write(legend_items),
            run_time=1,
        )

        self.play(
            *(Create(obj, run_time=esdf_time.max()) for obj in curves),
            Succession(
                Add(sub_text_zoom_prev),
                Wait(8.0),
                ShrinkToCenter(sub_text_zoom_prev, run_time=1e-3),
                Add(sub_text_zoom),
            ),
        )

        self.remove(sub_text_zoom_prev, sub_text_zoom)
        par = (
            "In slow motion we can see that this causes vibrations, "
            "visible even in the animation. The reason is the rapidly "
            "switching witness points."
        )
        _, sub_text_zoom = create_subtitle_box(
            wrap(par, CHAR_LIM + 16),
            corner_radius=0.05,
            width=sub_rect_width,
            height=sub_rect_height,
            font_size=sub_rect_font_size,
        )
        sub_rect_zoom.match_y(euc_rect, DOWN).match_x(euc_rect).shift(
            UP * sub_rect_up_ratio
        )
        sub_text_zoom.move_to(sub_rect_zoom)
        self.add(sub_text_zoom)
        self.wait(22)

        # 7b.  Prepare the "small" version that will sit inside the rectangle
        # after we zoom out.  Create new axes / curves with dimensions
        # that match the original rectangle size.
        small_x_length = case_rect_w * 0.85
        small_y_length = case_rect_h * 0.9  # leave room for the subtitle

        rect_graph_font_size = 20
        graph_axes_small = Axes(
            x_range=graph_axes.x_range.copy(),
            y_range=graph_axes.y_range.copy(),
            x_length=small_x_length,
            y_length=small_y_length,
            tips=False,
            axis_config={"include_numbers": True, "font_size": rect_graph_font_size},
            x_axis_config={"decimal_number_config": {"num_decimal_places": 2}},
            y_axis_config={"decimal_number_config": {"num_decimal_places": 2}},
        )
        graph_axes_small.move_to(euc_rect.get_center())

        # Build the 7 curves in the small axes
        curves_small = VGroup()
        for j in range(7):
            curve_s = graph_axes_small.plot_line_graph(
                esdf_time,
                esdf_u[:, j],
                line_color=colors[j],
                add_vertex_dots=False,
                stroke_width=2,
            )
            curves_small.add(curve_s)

        # Legend repositioned to top‑right corner of the small graph
        legend_items_small = VGroup()
        for j in range(7):
            item = MathTex(
                f"\\dot{{q}}_{{{j+1}}}", color=colors[j], font_size=rect_graph_font_size
            )
            legend_items_small.add(item)
        # Legend repositioned to top‑right corner of the small graph
        legend_items_small.arrange(RIGHT, buff=0.08)
        top_right_point = graph_axes_small.coords_to_point(
            esdf_time.max(), np.max(esdf_u)
        )
        legend_items_small.next_to(top_right_point, UL, buff=0.05)

        # 7c.  Transition: morph the zoomed‑in elements into the small ones,
        # remove the zoom‑specific subtitle, then zoom the camera back.
        self.play(
            ReplacementTransform(graph_axes, graph_axes_small),
            *[ReplacementTransform(curves[j], curves_small[j]) for j in range(7)],
            ReplacementTransform(legend_items, legend_items_small),
            # legend_items.animate.move_to(legend_items),
            ShrinkToCenter(sub_text_zoom),
            ShrinkToCenter(sub_rect_zoom),
            Restore(self.camera.frame),
            run_time=1.5,
        )
        self.remove(
            graph_axes, curves, sub_text_zoom, sub_rect_zoom
        )  # old mobjects gone
        self.add(graph_axes_small, curves_small)  # keep the small ones

        # -------------------------------------------------------------
        # ---------------------- HD-SDF
        # -------------------------------------------------------------

        # 9.  Now populate the HD‑SDF case parameters
        par = r"Our HD‑SDF precisely solves this problem."
        _, sub_text = create_subtitle_box(wrap(par, CHAR_LIM))
        self.add(sub_text)
        self.wait(5.0)

        # ---------------- TODO --------------------------------------
        self.remove(sub_text)
        par = (
            r"For the HD‑SDF, we use negative safety margins. "
            r"$\gamma$ is set to $2$ for a twice differentiable distance, "
            r"and $\epsilon$ is set accordingly."
        )
        _, sub_text = create_subtitle_box(wrap(par, CHAR_LIM))
        self.add(sub_text)
        # -----------------------------------------------------------

        hd_params = MathTex(
            r"\delta_{\text{obs}} &= -5\times 10^{-3} \quad \gamma=2\\"
            r"\delta_{\text{self}} &= -1\times 10^{-4} \quad \epsilon=9\times 10^{-4}",
            font_size=24,
        )
        hd_params.move_to(hd_rect.get_center() + UP * 0.3)
        self.play(Write(hd_params), run_time=2)
        self.wait(11)  # 13 total

        target_area = hd_rect
        self.remove(sub_text)
        self.play(
            FadeOut(hd_params),
            self.camera.frame.animate.replace(hd_rect, stretch=True),
            run_time=1.5,
        )
        self.wait(0.3)

        par = (
            "The safety margins are negative because we virtually expand "
            "the obstacles, allowing the robot to penetrate them slightly."
        )
        sub_rect_zoom, sub_text_zoom = create_subtitle_box(
            wrap(par, CHAR_LIM + 8),
            corner_radius=0.05,
            width=sub_rect_width,
            height=sub_rect_height,
            font_size=sub_rect_font_size,
        )
        sub_rect_zoom.set_z_index(-1)
        sub_text_zoom.set_z_index(2)
        sub_rect_zoom.match_y(hd_rect, DOWN).match_x(hd_rect).shift(
            UP * sub_rect_up_ratio
        )
        sub_text_zoom.move_to(sub_rect_zoom)
        self.add(sub_rect_zoom, sub_text_zoom)
        self.wait(9.0)

        # Graph axes (bottom half)
        y_data = np.round(hdsdf_u, 2)
        y_min_val = y_data.min()
        y_max_val = y_data.max()
        margin = (
            0.1 * (y_max_val - y_min_val) if (y_max_val - y_min_val) > 1e-6 else 0.1
        )
        x_min, x_max = hdsdf_time.min(), hdsdf_time.max()

        graph_axes = Axes(
            x_range=[x_min, x_max, max((x_max - x_min) / 4, 0.1)],
            y_range=[
                y_min_val - margin,
                y_max_val + margin,
                max((y_max_val - y_min_val) / 5, 0.05),
            ],
            x_length=case_rect_w * 0.8,
            y_length=mini_graph_height,
            tips=False,
            axis_config={
                "include_numbers": True,
                "font_size": 12,
                "tick_size": 0.05,
                "stroke_width": 1,
            },
            x_axis_config={"decimal_number_config": {"num_decimal_places": 2}},
            y_axis_config={"decimal_number_config": {"num_decimal_places": 2}},
        )
        graph_axes.next_to(sub_rect_zoom, UP, buff=0.1, aligned_edge=DOWN)

        # 7 coloured curves (one per joint)
        colors = [RED, BLUE, GREEN, YELLOW, ORANGE, PURPLE, PINK]
        curves = VGroup()
        for j in range(7):
            curve = graph_axes.plot_line_graph(
                hdsdf_time,
                hdsdf_u[:, j],
                line_color=colors[j],
                add_vertex_dots=False,
                stroke_width=1,
            )
            curves.add(curve)

        legend_items = VGroup()
        for j in range(7):
            item = MathTex(
                f"\\dot{{q}}_{{{j+1}}}", color=colors[j], font_size=sub_rect_font_size
            )
            legend_items.add(item)

        legend_items.arrange(RIGHT, buff=0.08)
        top_right_point = graph_axes.coords_to_point(hdsdf_time.max(), hdsdf_u.max())
        legend_items.next_to(top_right_point, UL, buff=0.05)

        self.play(
            Create(graph_axes),
            Write(legend_items),
            run_time=2,
        )

        self.remove(sub_text_zoom)
        par = "With the smoother control input, no vibrations are observed."
        _, sub_text_zoom = create_subtitle_box(
            wrap(par, CHAR_LIM + 8),
            corner_radius=0.05,
            width=sub_rect_width,
            height=sub_rect_height,
            font_size=sub_rect_font_size,
        )
        sub_text_zoom.set_z_index(2)
        sub_rect_zoom.match_y(hd_rect, DOWN).match_x(hd_rect).shift(
            UP * sub_rect_up_ratio
        )
        sub_text_zoom.move_to(sub_rect_zoom)
        self.add(sub_text_zoom)

        self.play(*(Create(obj) for obj in curves), run_time=hdsdf_time.max())

        # 7b.  Prepare the "small" version that will sit inside the rectangle
        # after we zoom out.  Create new axes / curves with dimensions
        # that match the original rectangle size.
        hd_graph_axes_small = Axes(
            x_range=graph_axes.x_range.copy(),
            y_range=graph_axes.y_range.copy(),
            x_length=small_x_length,
            y_length=small_y_length,
            tips=False,
            axis_config={"include_numbers": True, "font_size": rect_graph_font_size},
            x_axis_config={"decimal_number_config": {"num_decimal_places": 2}},
            y_axis_config={"decimal_number_config": {"num_decimal_places": 2}},
        )
        hd_graph_axes_small.move_to(hd_rect.get_center())

        # Build the 7 curves in the small axes
        hd_curves_small = VGroup()
        for j in range(7):
            curve_s = hd_graph_axes_small.plot_line_graph(
                hdsdf_time,
                hdsdf_u[:, j],
                line_color=colors[j],
                add_vertex_dots=False,
                stroke_width=2,
            )
            hd_curves_small.add(curve_s)

        # Legend repositioned to top‑right corner of the small graph
        hd_legend_items_small = VGroup()
        for j in range(7):
            item = MathTex(
                f"\\dot{{q}}_{{{j+1}}}", color=colors[j], font_size=rect_graph_font_size
            )
            hd_legend_items_small.add(item)
        # Legend repositioned to top‑right corner of the small graph
        hd_legend_items_small.arrange(RIGHT, buff=0.08)
        top_right_point = hd_graph_axes_small.coords_to_point(
            hdsdf_time.max(), np.max(hdsdf_u)
        )
        hd_legend_items_small.next_to(top_right_point, UL, buff=0.05)

        # 7c.  Transition: morph the zoomed‑in elements into the small ones,
        # remove the zoom‑specific subtitle, then zoom the camera back.
        self.play(
            ReplacementTransform(graph_axes, hd_graph_axes_small),
            *[ReplacementTransform(curves[j], hd_curves_small[j]) for j in range(7)],
            ReplacementTransform(legend_items, hd_legend_items_small),
            # legend_items.animate.move_to(legend_items),
            ShrinkToCenter(sub_text_zoom),
            ShrinkToCenter(sub_rect_zoom),
            Restore(self.camera.frame),
            run_time=1.0,
        )
        self.remove(
            graph_axes, curves, sub_text_zoom, sub_rect_zoom
        )  # old mobjects gone
        self.add(hd_graph_axes_small, hd_curves_small)  # keep the small ones

        # -------------------------------------------------------------
        # ---------------------- BIG GRAPHS
        # -------------------------------------------------------------

        big_x_length = (config.frame_width - 0.5) * 0.85
        big_y_length = case_rect_h
        big_font_size = 24
        graph_axes_big = Axes(
            x_range=graph_axes.x_range.copy(),
            y_range=graph_axes.y_range.copy(),
            x_length=big_x_length,
            y_length=big_y_length,
            tips=False,
            axis_config={"include_numbers": True, "font_size": big_font_size},
            x_axis_config={"decimal_number_config": {"num_decimal_places": 2}},
            y_axis_config={"decimal_number_config": {"num_decimal_places": 2}},
        )
        graph_axes_big.move_to(ORIGIN).shift(UP * 1.3 * big_y_length / 2)
        # Build the 7 curves in the small axes
        curves_big = VGroup()
        for j in range(7):
            curve_s = graph_axes_big.plot_line_graph(
                esdf_time,
                esdf_u[:, j],
                line_color=colors[j],
                add_vertex_dots=False,
                stroke_width=2,
            )
            curves_big.add(curve_s)
        # Legend repositioned to top‑right corner of the small graph
        legend_items_big = VGroup()
        for j in range(7):
            item = MathTex(
                f"\\dot{{q}}_{{{j+1}}}", color=colors[j], font_size=big_font_size
            )
            legend_items_big.add(item)
        # Legend repositioned to top‑right corner of the small graph
        legend_items_big.arrange(RIGHT, buff=0.08)
        top_right_point = graph_axes_big.coords_to_point(
            esdf_time.max(), np.max(esdf_u)
        )
        legend_items_big.next_to(top_right_point, UL, buff=0.05)

        hd_graph_axes_big = Axes(
            x_range=graph_axes.x_range.copy(),
            y_range=graph_axes.y_range.copy(),
            x_length=big_x_length,
            y_length=big_y_length,
            tips=False,
            axis_config={"include_numbers": True, "font_size": big_font_size},
            x_axis_config={"decimal_number_config": {"num_decimal_places": 2}},
            y_axis_config={"decimal_number_config": {"num_decimal_places": 2}},
        )
        hd_graph_axes_big.move_to(ORIGIN).shift(DOWN * big_y_length / 2)
        # Build the 7 curves in the small axes
        hd_curves_big = VGroup()
        for j in range(7):
            curve_s = hd_graph_axes_big.plot_line_graph(
                hdsdf_time,
                hdsdf_u[:, j],
                line_color=colors[j],
                add_vertex_dots=False,
                stroke_width=2,
            )
            hd_curves_big.add(curve_s)
        self.remove(sub_text)
        par = (
            "Comparing both control inputs, it is clear that the "
            "Hölder Signed Distance yields much smoother results."
        )
        _, sub_text = create_subtitle_box(wrap(par, CHAR_LIM))
        self.add(sub_text)

        # Legend repositioned to top‑right corner of the small graph
        self.play(
            Unwrite(opt_numeric),
            Uncreate(euc_rect),
            Uncreate(hd_rect),
            Unwrite(euc_label),
            Unwrite(hd_label),
            ReplacementTransform(graph_axes_small, graph_axes_big),
            ReplacementTransform(hd_graph_axes_small, hd_graph_axes_big),
            *[ReplacementTransform(curves_small[j], curves_big[j]) for j in range(7)],
            *[
                ReplacementTransform(hd_curves_small[j], hd_curves_big[j])
                for j in range(7)
            ],
            ReplacementTransform(
                Group(legend_items_small, hd_legend_items_small), legend_items_big
            ),
            # legend_items.animate.move_to(legend_items),
            ShrinkToCenter(sub_rect_zoom),
            run_time=1.5,
        )
        self.wait(10)
