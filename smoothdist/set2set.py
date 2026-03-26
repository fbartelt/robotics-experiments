# %%
import webbrowser
import uaibot as ub
import numpy as np
import scipy as sp
import uaibot as ub
import plotly.graph_objects as go
import plotly.express as px
import plotly.colors as pc
from polygon import Polytope
from scipy.optimize import linprog

# from uaibot_cpp_bind import expSO3, SmapSO3, SmapSE3, expSE3, ECdistance
from uaibot.simobjects.convexpolytope import ConvexPolytope
from plotly.subplots import make_subplots
from pathlib import Path
from smoothfunctions import (
    holderMeanWithGradient,
    signedDist2Convex,
    smoothMinListWithGradient,
    smoothMin2ElementsGradient,
    smoothMin2ElementsWithGradient,
    smoothMinList,
    smoothMaxList,
    smoothMaxListWithGradient,
    phi,
    ESDF_CGAL,
)

# xs = np.linspace(-0.5, 0.5, 1000)
# vals = []
# r = 1e-1
#
# for x in xs:
#     smin = smoothMinList(np.hstack([x, x**2, x**3]), r=r)
#     smin2 = smoothMinList(np.hstack([np.exp(x), 2.0]), r=r)
#     smax = smoothMaxList(np.array([smin, smin2]), r=r)
#     vals.append(smax)
#
# fig = go.Figure()
# fig.add_trace(go.Scatter(x=xs, y=vals))
# fig.show()
#
#
# # %%


# open browser with html file (neovim workaround)
def open_in_browser(filename: str):
    """
    Opens an HTML file in the system's default web browser.
    Works cross-platform (Linux, macOS, Windows).
    """
    path = Path(filename).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    # Convert to file:// URL and open
    webbrowser.open_new_tab(path.as_uri())


def pR2htm(p, R):
    H = np.eye(4)
    H[:3, :3] = R
    H[:3, -1] = p.ravel()
    return H


# Two Cubes
side = 0.1
A = np.array(
    [
        [1.0, 0, 0],
        [-1.0, 0, 0],
        [0, 1.0, 0],
        [0, -1.0, 0],
        [0, 0, 1.0],
        [0, 0, -1.0],
    ]
)
b = np.array([side, side, side, side, side, side])
dx_ = 0.5
displacement = np.array([dx_, 0.0, 0]).reshape(-1, 1)
P = Polytope(A=A, b=b)
Q = Polytope(A=A, b=b - (A @ displacement).ravel())
PmQ, _ = Polytope.minkowski_subtraction(P, Q)

ubP = ConvexPolytope(htm=np.eye(4), A=P.A, b=P.b, color="red", opacity=0.8)
ubQ = ConvexPolytope(htm=np.eye(4), A=Q.A, b=Q.b, color="blue", opacity=0.8)
# ubsum = ConvexPolytope(A=PmQ.A, b=PmQ.b, color='green')
sim = ub.Simulation()
sim.add([ubP, ubQ])

r = 1e-1
h = 1e-2

scale = 10
dt = 1e-3
T = 5.0  # s
imax = int(T / dt)
hist_dist = []
hist_gradnorm = []
hist_val = []
hist_euc_dist = []
rng = np.random.default_rng(seed=42)


def dist_set2set(P: Polytope, Q: Polytope):
    """TEST: Compute the approximate distance:
        max_{d in D} min_{vp in P, vq in Q} d^T (vp - vq)"""
    verts_P = P.vertices
    verts_Q = Q.vertices
    minkowski_verts = [vP - vQ for vP in verts_P for vQ in verts_Q]
    normals_P = [a for a in P.A]
    normals_Q = [a for a in Q.A]
    directions = normals_P + normals_Q
    inner_mins = []
    jacobian = np.zeros((len(directions), len(minkowski_verts)))
    for i, d in enumerate(directions):
        dotprods = [np.dot(d, v) for v in minkowski_verts]
        dist, grad = smoothMinListWithGradient(dotprods, r=r)
        inner_mins.append(dist)
        jacobian[i, :] = grad
    dist, grad = smoothMaxListWithGradient(inner_mins, r=r)
    grad = grad @ jacobian
    return dist, grad


for i in range(imax):
    dx = i * dt / T
    angle = 2 * np.pi * i * dt / T
    translate = np.array([dx, 0.0, 0.0]).reshape(-1, 1)
    # translate = np.array([dx, 1.5*dx, 1e-1*dx]).reshape(-1, 1)
    R = np.array(ub.Utils.rotx(angle) @ ub.Utils.roty(angle) @ ub.Utils.rotz(angle))[
        :3, :3
    ]
    # R = np.array(ub.Utils.roty(angle))[:3, :3]
    # R = np.eye(3)

    A_new = A @ R.T
    b_new = Q.b.reshape(-1, 1) + A_new @ translate
    Q_new = Polytope(A_new, b_new)
    htm = pR2htm(translate, R)
    # print(R)
    ubQ.add_ani_frame(time=i * dt, htm=htm)

    dist, grad = dist_set2set(P, Q_new)
    _, _, euclidean_dist, *_ = ubQ.compute_dist(ubP)
    hist_dist.append(dist)
    hist_euc_dist.append(euclidean_dist)
    hist_gradnorm.append(np.linalg.norm(grad))
    # hist_val.append(val)


def cfig(data):
    if isinstance(data, np.ndarray):
        data = [data]
    fig = go.Figure()
    for d in data:
        fig.add_trace(go.Scatter(y=d))
    return fig


cfig([hist_dist, hist_euc_dist]).show()
cfig([hist_gradnorm]).show()

# %%
"""Visual"""
# sim.add([ubsum])
filename = "minkowski"
sim.set_parameters(width=800, height=600, pixel_ratio=0.7)
sim.save("./", f"{filename}")
print("Saved")

# print(dt * (final_time / sim_dt) / decimation / speedup)

open_in_browser(f"./{filename}.html")
