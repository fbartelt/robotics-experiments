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

font_size = 20
line_width = 2.5
band_opacity = 0.4
def plot_fp_rate(tolerances, fp_rates, legend_labels, tol_zero_rate=None, title=None):
    """Plotly figure for the false‑positive curve."""
    fig = go.Figure()
    for fp, lbl, in zip(fp_curves, legend_labels):
        fig.add_trace(
            go.Scatter(
                x=tolerances,
                y=fp,
                mode="lines+markers",
                name=lbl,
                line=dict(width=2.5),
                marker=dict(size=6),
                showlegend=False,
            )
        )
    if tol_zero_rate is not None:
        fig.add_annotation(
            x=tolerances[0],
            y=tol_zero_rate,
            text=f"tol = 0: {tol_zero_rate:.2f}%",
            showarrow=True,
            arrowhead=2,
            ax=0,
            ay=-20,
        )
    return fig

file_name = "fp_eps_sweep.pkl"
with open(file_name, "rb") as f:
    data = pickle.load(f)

# data_out = {
#     "tolerances": tolerances,
#     "fp_rates": fp_rates,
#     "tol_zero_rate": tol_zero_rate,
#     "sdf_distances": sdf_distances,
#     "hd_distances": hd_distances,
# }

# fp_rates = data["fp_curves"]
# sdf_distances = np.array(data["hd_distances"])
# hd_distances = np.array(data["sdf_distances"])
# tol_zero_mask = np.array(sdf_distances) >= 0.0
# false_positives = (np.array(hd_distances) <= 0) & tol_zero_mask
# hd_distances[false_positives]
# sdf_distances[false_positives]
# tol_zero_rate = false_positives.sum() / tol_zero_mask.sum() * 100.0

tolerances = data["tolerances"]
fp_curves = data["fp_curves"]
legend_labels = data["legend_labels"]


# tolerances = np.logspace(-4, 1, 50)  # 0.0001 … 1.0
# denom_counts = [(np.array(sdf_distances) >= tol).sum() for tol in tolerances]
# print(np.array(list(zip(tolerances, denom_counts, fp_rates))))


fig = plot_fp_rate(
    tolerances, fp_curves, legend_labels,
    title=f"False‑positive rate vs. tolerance "
)
 
fig.update_xaxes(type="log", title_text="Tolerance (SDF ≥ tol)")
fig.update_yaxes(title_text="False‑positive rate (%)")#, range=[0, max(fp_rates) * 1.1])
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

fig.write_image("false_positive_rate.pdf", width=1200, height=480)
# fig.show()
#
