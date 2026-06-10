import pickle
import numpy as np
import plotly.graph_objects as go
from collections import defaultdict

# ----------------------------------------------------------------------
# 1.  Load the data
# ----------------------------------------------------------------------
# Option A – if you saved all four dicts together in one file:
# with open("timing_dicts.pkl", "rb") as f:
#     time_HD, time_CHD, time_SDF, time_GDF = pickle.load(f)

# Option B – if you saved them separately (adjust filenames):
# with open("time_HD.pkl", "rb") as f:   time_HD  = pickle.load(f)
# with open("time_CHD.pkl", "rb") as f:  time_CHD = pickle.load(f)
# with open("time_SDF.pkl", "rb") as f:  time_SDF = pickle.load(f)
# with open("time_GDF.pkl", "rb") as f:  time_GDF = pickle.load(f)

# --- Replace this block with your actual loading ---
# (Example using a single file)
with open("data2plot.pkl", "rb") as f:
    (
        x_vals,
        hd_sdf_gm,
        hd_sdf_lo,
        hd_sdf_hi,
        chd_sdf_gm,
        chd_sdf_lo,
        chd_sdf_hi,
        hd_gdf_gm,
        hd_gdf_lo,
        hd_gdf_hi,
        time_HD,
        time_CHD,
        time_SDF,
        time_GDF,
    ) = pickle.load(f)
# ----------------------------------------------------------------------
# 2.  Collect all times per complexity for each method
# ----------------------------------------------------------------------
# The dictionaries share the same keys
all_keys = set(time_HD.keys())
methods = ["HDSDF", "SDF", "GDF"]
time_by_complexity = defaultdict(lambda: {m: [] for m in methods})

for key in all_keys:
    complexity = key[2]  # the third element of the key tuple
    time_by_complexity[complexity]["HDSDF"].extend(time_HD[key])
    # time_by_complexity[complexity]["CHDSDF"].extend(time_CHD[key])
    time_by_complexity[complexity]["SDF"].extend(time_SDF[key])
    time_by_complexity[complexity]["GDF"].extend(time_GDF[key])

complexities = sorted(time_by_complexity.keys())


# ----------------------------------------------------------------------
# 3.  Mean and bootstrap confidence intervals
# ----------------------------------------------------------------------
def mean_and_bootstrap_ci(data, n_bootstrap=10000, alpha=0.05):
    data = np.array(data)
    mean_val = np.mean(data)
    rng = np.random.default_rng(42)
    boot_means = [
        np.mean(rng.choice(data, size=len(data), replace=True))
        for _ in range(n_bootstrap)
    ]
    ci_lo = np.percentile(boot_means, 100 * alpha / 2)
    ci_hi = np.percentile(boot_means, 100 * (1 - alpha / 2))
    return mean_val, ci_lo, ci_hi


plot_data = {m: {"x": [], "y": [], "y_lo": [], "y_hi": []} for m in methods}
for c in complexities:
    for m in methods:
        mean_t, lo_t, hi_t = mean_and_bootstrap_ci(time_by_complexity[c][m])
        plot_data[m]["x"].append(c)
        plot_data[m]["y"].append(mean_t)
        plot_data[m]["y_lo"].append(lo_t)
        plot_data[m]["y_hi"].append(hi_t)


font_size = 20
line_width = 2.5
band_opacity = 0.4


def add_trace_with_band_single(fig, x, y, y_lo, y_hi, name, color, show_legend=True):
    # Main line
    fig.add_trace(
        go.Scatter(
            name=name,
            x=x,
            y=y,
            mode="lines+markers",
            line=dict(color=color, width=line_width),
            showlegend=show_legend,
        )
    )
    # Upper bound (invisible)
    fig.add_trace(
        go.Scatter(
            x=x,
            y=y_hi,
            mode="lines",
            marker=dict(color="rgba(68,68,68,0)"),
            line=dict(width=0),
            showlegend=False,
        )
    )
    # Lower bound with fill
    fig.add_trace(
        go.Scatter(
            x=x,
            y=y_lo,
            mode="lines",
            marker=dict(color="rgba(68,68,68,0)"),
            line=dict(width=0),
            fillcolor=color.replace("rgb", "rgba").replace(")", f",{band_opacity})"),
            fill="tonexty",
            showlegend=False,
        )
    )


# Create figure
fig = go.Figure()

colors = {
    "HDSDF": "rgb(31,119,180)",  # blue
    "CHDSDF": "rgb(255,127,14)",  # orange
    "SDF": "rgb(44,160,44)",  # green
    "GDF": "rgb(148,103,189)",  # purple
    "GPU": "rgb(214,39,40)",  # red
}

for m in methods:
    if m in plot_data:  # only if data exists
        m_ = "HD-SDF" if m == "HDSDF" else m
        add_trace_with_band_single(
            fig,
            plot_data[m]["x"],
            plot_data[m]["y"],
            plot_data[m]["y_lo"],
            plot_data[m]["y_hi"],
            name=m_,
            color=colors[m],
            show_legend=True,
        )
# ----------------------------------------------------
# Optional GPU time (set to None if not available)
time_GPU = {
    ("tetra", "tetra"): 0.454422 * 1e-6,  # second
    ("tetra", "cube"): 0.766345 * 1e-6,
    ("tetra", "octa"): 0.629510 * 1e-6,
    ("tetra", "dodeca"): 2.426670 * 1e-6,
    ("tetra", "icosa"): 1.875779 * 1e-6,
    ("cube", "cube"): 1.408164 * 1e-6,
    ("cube", "octa"): 1.240184 * 1e-6,
    ("cube", "dodeca"): 4.943016 * 1e-6,
    ("cube", "icosa"): 3.458157 * 1e-6,
    ("octa", "octa"): 1.126226 * 1e-6,
    ("octa", "dodeca"): 4.085908 * 1e-6,
    ("octa", "icosa"): 2.891865 * 1e-6,
    ("dodeca", "dodeca"): 16.613784 * 1e-6,
    ("dodeca", "icosa"): 10.302578 * 1e-6,
    ("icosa", "icosa"): 7.177605 * 1e-6,
}
vertices = {"tetra": 4, "cube": 8, "octa": 6, "dodeca": 20, "icosa": 12}


def complexity(solidA, solidB):
    va, vb = vertices[solidA], vertices[solidB]
    return va * va * vb + va * vb * vb  # |VA|²|VB| + |VA||VB|²


# ----- sort GPU data by complexity -----
pairs = []
for (sA, sB), t in time_GPU.items():
    pairs.append((complexity(sA, sB), t))
pairs.sort(key=lambda x: x[0])
gpu_x = [p[0] for p in pairs]
gpu_y = [p[1] for p in pairs]

# Add GPU points to the existing figure (fig)
fig.add_trace(
    go.Scatter(
        x=gpu_x,
        y=gpu_y,
        name="HD-SDF (GPU)",
        mode="lines+markers",
        line=dict(color=colors["GPU"], width=line_width),
        showlegend=True,
    )
)
# ----------------------------------------------------
# Log axes
fig.update_xaxes(
    type="log",
    title_text=r"Complexity  $|\mathcal{V}(\mathcal{A})|^2|\mathcal{V}(\mathcal{B})| + |\mathcal{V}(\mathcal{A})||\mathcal{V}(\mathcal{B})|^2$",
    showline=True,
    linewidth=1,
    linecolor="black",
    mirror=True,
)
fig.update_yaxes(
    type="log",
    title_text="Time (seconds)",
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
        xanchor="left",
        x=0.01,
        bgcolor="rgba(255,255,255,0.7)",
    ),
    font=dict(size=font_size),
    margin=dict(l=90, r=10, t=10, b=90, pad=0),
)
fig.update_xaxes(
    automargin=False,
    title_standoff=15,
)  # kills the extra space reserved for axis title ticks
fig.update_yaxes(
    automargin=False,
    title_standoff=15,
)
# fig.show()
fig.write_image("absolute_time.pdf", width=1200, height=480)
