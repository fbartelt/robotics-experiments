# %%
import pickle
import numpy as np
import plotly.graph_objects as go

esdf_data = "./experiment_data/mode_0/video_7_data/data.pickle"
hdsdf_data = "./experiment_data/mode_1/video_1_data/data.pickle"

with open(esdf_data, "rb") as f:
    esdf = pickle.load(f)

with open(hdsdf_data, "rb") as f:
    hdsdf = pickle.load(f)


def plot_config(hist_t, hist_q, line_width=2.5):
    """For showing every single one"""
    fig = go.Figure()
    N = hist_q.shape[1]
    for i in range(N):
        fig.add_trace(
            go.Scatter(
                x=hist_t,
                y=hist_q[:, i],
                mode="lines",
                line=dict(width=line_width),
            )
        )
    return fig


def compare_input(
    hist_t_esdf,
    hist_t_hdsdf,
    hist_u_esdf,
    hist_u_hdsdf,
    line_width=2.5,
    color_hdsdf="rgb(31,119,180)",
    color_esdf="rgb(44,160,44)",
    name_esdf="Input Euclidean",
    name_hdsdf="Input HD-SDF",
    font_size=20,
):
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=hist_t_hdsdf,
            y=hist_u_hdsdf,
            mode="lines",
            line=dict(width=line_width, color=color_hdsdf),
            name=name_hdsdf,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=hist_t_esdf,
            y=hist_u_esdf,
            mode="lines",
            line=dict(width=line_width, color=color_esdf),
            name=name_esdf,
        )
    )
    # ----------------------------------------------------
    fig.update_xaxes(
        title_text="Time (s)",
        showline=True,
        linewidth=1,
        linecolor="black",
        mirror=True,
    )
    fig.update_yaxes(
        title_text="Joint Velocity (rad/s)",
        showline=True,
        linewidth=1,
        linecolor="black",
        mirror=True,
    )

    # Add a border rectangle around the entire plot area
    fig.add_shape(
        type="rect",
        xref="paper",
        yref="paper",
        x0=0,
        y0=0,
        x1=1,
        y1=1,
        line=dict(color="black", width=2),
    )

    # Layout
    fig.update_layout(
        # title="Computation time vs. complexity",
        template="plotly_white",
        hovermode="x unified",
        legend=dict(
            orientation="h",
            yanchor="top",
            y=0.99,
            xanchor="right",
            x=0.99,
            bgcolor="rgba(255,255,255,0.7)",
        ),
        font=dict(size=font_size),
        margin=dict(l=90, r=10, t=10, b=90, pad=0),
        width=1200,
        height=480,
    )
    fig.update_xaxes(
        automargin=False,
        title_standoff=15,
    )  # kills the extra space reserved for axis title ticks
    fig.update_yaxes(
        automargin=False,
        title_standoff=15,
    )
    return fig


i = -1  # Plot last joint velocity
T = 4.973
hdsdf_time_ = hdsdf["timestamp"]
t0 = hdsdf_time_[0]
hdsdf_time = np.array([t - t0 for t in hdsdf_time_])
hdsdf_dist = np.array(hdsdf["hist_dist"]).reshape(-1, 1)
hdsdf_q = np.array(hdsdf["q"]).reshape(-1, 7)
hdsdf_qdot = np.array(hdsdf["hist_dq"]).reshape(-1, 7)
hdsdf_u = hdsdf_qdot[:, i]
hdsdf_imax = np.nonzero(hdsdf_time >= T)[0][0]
# fig = plot_config(hdsdf_time, hdsdf_qdot)
# fig.show()

esdf_time_ = esdf["timestamp"]
t0 = esdf_time_[0]
esdf_time = np.array([t - t0 for t in esdf_time_])
esdf_dist = np.array(esdf["hist_dist"]).reshape(-1, 1)
esdf_q = np.array(esdf["q"]).reshape(-1, 7)
esdf_qdot = np.array(esdf["hist_dq"]).reshape(-1, 7)
esdf_u = esdf_qdot[:, i]
esdf_imax = np.nonzero(esdf_time >= T)[0][0]
# fig = plot_config(esdf_time, esdf_qdot)
# fig.show()

fig = compare_input(
    hist_t_hdsdf=hdsdf_time[:hdsdf_imax],
    hist_t_esdf=esdf_time[:esdf_imax],
    hist_u_hdsdf=hdsdf_u[:hdsdf_imax],
    hist_u_esdf=esdf_u[:esdf_imax],
    name_hdsdf="HD-SDF",
    name_esdf="Euclidean",
)
fig.show()
fig.write_image("experiment_control_input.pdf", width=1200, height=480)
