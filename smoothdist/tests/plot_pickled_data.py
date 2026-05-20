# %%
# This script assumes that compute_time.py was run at least once.
import sys
import os
import pickle
import itertools
import numpy as np
import uaibot as ub
import plotly.graph_objects as go
import webbrowser
from time import time, perf_counter

sys.path.insert(0, "/home/fbartelt/Projects/robotics-experiments/smoothdist")
sys.path.insert(0, "/home/fbartelt/Documents/Projetos/robotics-experiments/smoothdist/")
from euclidean_sdf import esdf, compute_vertices_and_faces
from scipy.spatial import ConvexHull
from pathlib import Path
from plotly.subplots import make_subplots
from multiprocessing import Pool, cpu_count
from tqdm import tqdm

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
    ) = pickle.load(f)

fig = go.Figure()

font_size = 20
line_width = 2.5
band_opacity = 0.4
# ----- Convenience: add a trace with its own confidence band -----
def add_trace_with_band(
    fig, x, y, y_lo, y_hi, name, color, row, col, show_legend=True
):
    """Add a main line and a filled error band to a subplot."""
    # Main line
    fig.add_trace(
        go.Scatter(
            name=name,
            x=x,
            y=y,
            mode="lines",
            line=dict(color=color, width=line_width),
            showlegend=show_legend,
        ),
        # row=row,
        # col=col,
    )
    # Upper bound (invisible line, just used for filling)
    fig.add_trace(
        go.Scatter(
            name=f"{name} upper",
            x=x,
            y=y_hi,
            mode="lines",
            marker=dict(color="rgba(68,68,68,0)"),
            line=dict(width=0),
            showlegend=False,
        ),
        # row=row,
        # col=col,
    )
    # Lower bound – fills to the upper bound
    fig.add_trace(
        go.Scatter(
            name=f"{name} lower",
            x=x,
            y=y_lo,
            mode="lines",
            marker=dict(color="rgba(68,68,68,0)"),
            line=dict(width=0),
            fillcolor=color.replace("rgb", "rgba").replace(")", f",{band_opacity})"),
            fill="tonexty",
            showlegend=False,
        ),
        # row=row,
        # col=col,
    )

# ----- Panel (a): HD‑SDF / SDF  and  CHD‑SDF / SDF -----
add_trace_with_band(
    fig,
    x_vals,
    hd_sdf_gm,
    hd_sdf_lo,
    hd_sdf_hi,
    name="HD‑SDF / SDF",
    color="rgb(31, 119, 180)",  # blue
    row=1,
    col=1,
)

add_trace_with_band(
    fig,
    x_vals,
    chd_sdf_gm,
    chd_sdf_lo,
    chd_sdf_hi,
    name="CHD‑SDF / SDF",
    color="rgb(255, 127, 14)",  # orange
    row=1,
    col=1,
)

# Parity line (y = 1)
# fig.add_trace(
#     go.Scatter(
#         x=[min(x_vals), max(x_vals)],
#         y=[1, 1],
#         mode="lines",
#         line=dict(dash="dash", color="black", width=0.8),
#         showlegend=False,
#     ),
#     row=1,
#     col=1,
# )

# ----- Panel (b): HD‑SDF / GDF -----
add_trace_with_band(
    fig,
    x_vals,
    hd_gdf_gm,
    hd_gdf_lo,
    hd_gdf_hi,
    name="HD‑SDF / GDF",
    color="rgb(44, 160, 44)",  # green
    row=2,
    col=1,
)

# Parity line (y = 1)
# fig.add_trace(
#     go.Scatter(
#         x=[min(x_vals), max(x_vals)],
#         y=[1, 1],
#         mode="lines",
#         line=dict(dash="dash", color="black", width=0.8),
#         showlegend=False,
#     ),
#     row=2,
#     col=1,
# )

# ---------------------------------------------------------------------------
# 3.  Layout
# ---------------------------------------------------------------------------
# fig.update_xaxes(
#     type="log",
#     row=1,
#     col=1,
# )

fig.update_xaxes(
    title_text="Complexity  |V<sub>A</sub>|² &#215; |V<sub>B</sub>|²",
    type="log",
    # row=2,
    # col=1,
)

fig.update_yaxes(
    title_text="Time ratio",
    type="log",  # <-- remove this line if ratios stay small
    # row=1,
    # col=1,
)
# fig.update_yaxes(
#     title_text="Time ratio (HD‑SDF / GDF)",
#     type="log",  # <-- remove if not needed
#     row=2,
#     col=1,
# )

fig.update_layout(
    # title=dict(text="Runtime comparison for Platonic‑solid pairs"),
    hovermode="x unified",
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1,
    ),
    template="plotly_white",
    font=dict(size=font_size),
    margin=dict(l=0, r=0, t=10)
)

# fig.show()
fig.write_image("runtime_comparison.pdf", width=1200, height=600)
print("Plot saved as 'runtime_comparison.pdf'.")
