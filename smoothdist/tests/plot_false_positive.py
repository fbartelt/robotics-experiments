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

def plot_fp_rate(tolerances, fp_rates, tol_zero_rate=None, title=None):
    """Plotly figure for the false‑positive curve."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=tolerances, y=fp_rates,
        mode='lines+markers',
        name='False positives',
        line=dict(color='rgb(31,119,180)', width=2.5),
        marker=dict(size=6),
    ))
    if tol_zero_rate is not None:
        fig.add_annotation(
            x=tolerances[0], y=tol_zero_rate,
            text=f"tol = 0: {tol_zero_rate:.2f}%",
            showarrow=True, arrowhead=2, ax=30, ay=-20
        )
    fig.update_xaxes(type='log', title_text='Tolerance (SDF ≥ tol)')
    fig.update_yaxes(title_text='False‑positive rate (%)',
                     range=[0, max(fp_rates) * 1.1])
    fig.update_layout(
        title=title if title else 'HD‑SDF false positives vs. tolerance',
        template='plotly_white',
        hovermode='x unified',
    )
    return fig



file_name = "fp_analysis.pkl"
with open(file_name, "rb") as f:
    data = pickle.load(f)

# data_out = {
#     "tolerances": tolerances,
#     "fp_rates": fp_rates,
#     "tol_zero_rate": tol_zero_rate,
#     "sdf_distances": sdf_distances,
#     "hd_distances": hd_distances,
# }
 
fp_rates = data["fp_rates"]
sdf_distances = np.array(data["hd_distances"])
hd_distances = np.array(data["sdf_distances"])
tol_zero_mask = np.array(sdf_distances) >= 0.0
false_positives = (np.array(hd_distances) <= 0) & tol_zero_mask
hd_distances[false_positives]
sdf_distances[false_positives]
tol_zero_rate = (
    false_positives.sum()
    / tol_zero_mask.sum()
    * 100.0
)


tolerances = np.logspace(-4, 1, 50)  # 0.0001 … 1.0
denom_counts = [(np.array(sdf_distances) >= tol).sum() for tol in tolerances]
print(np.array(list(zip(tolerances, denom_counts, fp_rates))))

# fig = plot_fp_rate(tolerances, fp_rates, tol_zero_rate=tol_zero_rate,
#                    title="HD‑SDF conservatism (false positive rate)")
# fig.write_image("false_positive_rate.pdf")
# fig.show()
#

