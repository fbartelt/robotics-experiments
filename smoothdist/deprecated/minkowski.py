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
    phi,
    ESDF_CGAL,
)


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
imax = 10000 * scale
imax = 1000
hist_dist = []
hist_gradnorm = []
hist_val = []
rng = np.random.default_rng(seed=42)


def dist2minkowski(P, Q):
    PmQ, val = Polytope.minkowski_subtraction(P, Q)
    if val == 32:
        dist, grad = signedDist2Convex(
            np.zeros((3,)),
            PmQ.A,
            PmQ.b.reshape(-1, 1),
            r=r,
            eps=h,
        )
        # dist, grad, _ = ESDF_CGAL(np.zeros((3,)), PmQ.A, PmQ.b.reshape(-1, 1))
        return dist, grad, val
    else:
        print(val)
        A_, b_ = Q.A.copy(), Q.b.copy()
        eps = 1 / np.sqrt(3)
        disturb = 1e-3
        R_p = np.array(
            ub.Utils.rotx(disturb) @ ub.Utils.roty(disturb) @ ub.Utils.rotz(disturb)
        )[:3, :3]
        R_2p = np.array(
            ub.Utils.rotx(2 * disturb)
            @ ub.Utils.roty(2 * disturb)
            @ ub.Utils.rotz(2 * disturb)
        )[:3, :3]
        R_m = np.array(
            ub.Utils.rotx(-disturb) @ ub.Utils.roty(-disturb) @ ub.Utils.rotz(-disturb)
        )[:3, :3]
        R_2m = np.array(
            ub.Utils.rotx(-2 * disturb)
            @ ub.Utils.roty(-2 * disturb)
            @ ub.Utils.rotz(-2 * disturb)
        )[:3, :3]

        A_p = A_ @ R_p.T
        A_2p = A_ @ R_2p.T
        A_m = A_ @ R_m.T
        A_2m = A_ @ R_2m.T
        Q_p = Polytope(A_p, b_)
        Q_2p = Polytope(A_2p, b_)
        Q_m = Polytope(A_m, b_)
        Q_2m = Polytope(A_2m, b_)

        dist_p, grad_p, val_p = dist2minkowski(P, Q_p)
        dist_m, grad_m, val_m = dist2minkowski(P, Q_m)
        dist_2p, grad_2p, val_2p = dist2minkowski(P, Q_2p)
        dist_2m, grad_2m, val_2m = dist2minkowski(P, Q_2m)
        dist = (
            (-1 / 6 * dist_2p)
            + (2 / 3 * dist_p)
            + (2 / 3 * dist_m)
            + (-1 / 6 * dist_2m)
        )
        grad = (
            (-1 / 6 * grad_2p)
            + (2 / 3 * grad_p)
            + (2 / 3 * grad_m)
            + (-1 / 6 * grad_2m)
        )
        # grad = ((-1/6 * grad_2p) + (2/3 * grad_p) + (2/3 * grad_m) + (-1/6 * grad_2m))
        return dist, grad, val


# Move one polytope and check if distance to minkowski is smooth:
for i in range(imax):
# for i in range(2400 * scale, 2600 * scale, 1):
    dx = 1 / imax * i
    angle = 2 * np.pi * i / imax
    translate = np.array([dx, 0.0, 0.0]).reshape(-1, 1)
    # translate = -displacement
    # translate = np.zeros((3, 1))
    R = np.array(ub.Utils.rotx(angle) @ ub.Utils.roty(angle) @ ub.Utils.rotz(angle))[
        :3, :3
    ]
    R = np.array(ub.Utils.rotz(angle))[:3, :3]
    # R = np.eye(3)
    A_new = A @ R.T
    b_new = Q.b.reshape(-1, 1) + A_new @ translate
    Q_new = Polytope(A_new, b_new)
    PmQ, val = Polytope.minkowski_subtraction(P, Q_new)

    # if PmQ.A.shape[0] < 16:
    #     eps = 1e-3
    #     b_ = b_new + A_new @ np.array([eps, eps, eps])
    #     b_ = b_new + A_new @ np.array([eps, eps, eps])

    htm0 = np.array(ubQ.htm)
    htm0 = np.eye(4)
    htm = htm0 @ pR2htm(translate, R)
    ubQ.add_ani_frame(time=i * 0.01, htm=htm)
    # if i is multiple of 125:
    if val < 32:
        sim.add(
            ConvexPolytope(
                A=PmQ.A, b=PmQ.b.reshape(-1, 1), color="magenta", opacity=0.5
            )
        )
        # sim.add(ConvexPolytope(A=A_new, b=b_new, color='magenta', opacity=0.5))

    dist, grad, val = dist2minkowski(P, Q_new)
    # hist_dist.append(A.shape[0])
    # hist_gradnorm.append(val)
    # hist_dist.append(np.linalg.norm(PmQ.A))
    # hist_gradnorm.append(np.linalg.norm(PmQ.b))
    hist_dist.append(dist)
    hist_gradnorm.append(np.linalg.norm(grad))
    hist_val.append(val)


def cfig(data):
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=data))
    return fig


cfig(hist_dist).show()
cfig(hist_gradnorm).show()

# %%
"""Visual"""
# sim.add([ubsum])
filename = "minkowski"
sim.set_parameters(width=800, height=600, pixel_ratio=0.7)
sim.save("./", f"{filename}")
print("Saved")

# print(dt * (final_time / sim_dt) / decimation / speedup)

open_in_browser(f"./{filename}.html")
