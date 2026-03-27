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
displacement = np.array([-dx_, 0.0, 0]).reshape(-1, 1)
P = Polytope(A=A, b=b)
Q = Polytope(A=A, b=1.5 * b + (A @ displacement).ravel())
PmQ, _ = Polytope.minkowski_subtraction(P, Q)

# ubP = ConvexPolytope(htm=np.eye(4), A=P.A, b=P.b, color="red", opacity=0.8)
# ubQ = ConvexPolytope(htm=np.eye(4), A=Q.A, b=Q.b, color="blue", opacity=0.8)
side *= 2.0
ubP = ub.Box(width=side, height=side, depth=side, color="red", opacity=0.8)
ubQ = ub.Box(
    width=1.5 * side,
    height=1.5 * side,
    depth=1.5 * side,
    color="blue",
    opacity=0.8,
    htm=pR2htm(-displacement, np.eye(3)),
)
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

def dist_box2box(ubP, ubQ):
    wx, wz, wy = ubP.width / 2, ubP.height / 2, ubP.depth / 2
    verts_P = []
    verts_P.append(
        np.array(
            ubP.htm[:3, 0] * wx
            + ubP.htm[:3, 1] * wy
            + ubP.htm[:3, 2] * wz
            + ubP.htm[:3, -1]
        )
    )
    verts_P.append(
        np.array(
            ubP.htm[:3, 0] * wx
            + ubP.htm[:3, 1] * wy
            + ubP.htm[:3, 2] * -wz
            + ubP.htm[:3, -1]
        )
    )
    verts_P.append(
        np.array(
            ubP.htm[:3, 0] * wx
            + ubP.htm[:3, 1] * -wy
            + ubP.htm[:3, 2] * wz
            + ubP.htm[:3, -1]
        )
    )
    verts_P.append(
        np.array(
            ubP.htm[:3, 0] * wx
            + ubP.htm[:3, 1] * -wy
            + ubP.htm[:3, 2] * -wz
            + ubP.htm[:3, -1]
        )
    )
    verts_P.append(
        np.array(
            ubP.htm[:3, 0] * -wx
            + ubP.htm[:3, 1] * wy
            + ubP.htm[:3, 2] * wz
            + ubP.htm[:3, -1]
        )
    )
    verts_P.append(
        np.array(
            ubP.htm[:3, 0] * -wx
            + ubP.htm[:3, 1] * wy
            + ubP.htm[:3, 2] * -wz
            + ubP.htm[:3, -1]
        )
    )
    verts_P.append(
        np.array(
            ubP.htm[:3, 0] * -wx
            + ubP.htm[:3, 1] * -wy
            + ubP.htm[:3, 2] * wz
            + ubP.htm[:3, -1]
        )
    )
    verts_P.append(
        np.array(
            ubP.htm[:3, 0] * -wx
            + ubP.htm[:3, 1] * -wy
            + ubP.htm[:3, 2] * -wz
            + ubP.htm[:3, -1]
        )
    )
    wx, wz, wy = ubQ.width / 2, ubQ.height / 2, ubQ.depth / 2
    verts_Q = []
    verts_Q.append(
        np.array(
            ubQ.htm[:3, 0] * wx
            + ubQ.htm[:3, 1] * wy
            + ubQ.htm[:3, 2] * wz
            + ubQ.htm[:3, -1]
        )
    )
    verts_Q.append(
        np.array(
            ubQ.htm[:3, 0] * wx
            + ubQ.htm[:3, 1] * wy
            + ubQ.htm[:3, 2] * -wz
            + ubQ.htm[:3, -1]
        )
    )
    verts_Q.append(
        np.array(
            ubQ.htm[:3, 0] * wx
            + ubQ.htm[:3, 1] * -wy
            + ubQ.htm[:3, 2] * wz
            + ubQ.htm[:3, -1]
        )
    )
    verts_Q.append(
        np.array(
            ubQ.htm[:3, 0] * wx
            + ubQ.htm[:3, 1] * -wy
            + ubQ.htm[:3, 2] * -wz
            + ubQ.htm[:3, -1]
        )
    )
    verts_Q.append(
        np.array(
            ubQ.htm[:3, 0] * -wx
            + ubQ.htm[:3, 1] * wy
            + ubQ.htm[:3, 2] * wz
            + ubQ.htm[:3, -1]
        )
    )
    verts_Q.append(
        np.array(
            ubQ.htm[:3, 0] * -wx
            + ubQ.htm[:3, 1] * wy
            + ubQ.htm[:3, 2] * -wz
            + ubQ.htm[:3, -1]
        )
    )
    verts_Q.append(
        np.array(
            ubQ.htm[:3, 0] * -wx
            + ubQ.htm[:3, 1] * -wy
            + ubQ.htm[:3, 2] * wz
            + ubQ.htm[:3, -1]
        )
    )
    verts_Q.append(
        np.array(
            ubQ.htm[:3, 0] * -wx
            + ubQ.htm[:3, 1] * -wy
            + ubQ.htm[:3, 2] * -wz
            + ubQ.htm[:3, -1]
        )
    )

    minkowski_verts = [vP - vQ for vP in verts_P for vQ in verts_Q]
    normals_P = [np.array(ubP.htm[:3, i]) for i in range(3)]
    normals_P += [-np.array(ubP.htm[:3, i]) for i in range(3)]
    normals_Q = [np.array(ubQ.htm[:3, i]) for i in range(3)]
    normals_Q += [-np.array(ubQ.htm[:3, i]) for i in range(3)]

    directions = normals_P + normals_Q
    inner_mins = []
    debugs = []
    jacobian = np.zeros((len(directions), len(minkowski_verts)))
    for i, d_ in enumerate(directions):
        d = d_ / np.linalg.norm(d_)
        dotprods = [np.dot(d.ravel(), v.ravel()) for v in minkowski_verts]
        dist, grad = smoothMinListWithGradient(dotprods, r=r)
        inner_mins.append(dist)
        debugs.append(dotprods)
        jacobian[i, :] = grad
    dist, grad = smoothMaxListWithGradient(inner_mins, r=r)
    grad = grad @ jacobian
    return dist, grad, (inner_mins, debugs, verts_Q, directions)
    # return dist, grad, (inner_mins, debugs, minkowski_verts, directions)


def dist_set2set(P: Polytope, Q: Polytope):
    """TEST: Compute the approximate distance:
    max_{d in D} min_{vp in P, vq in Q} d^T (vp - vq)"""
    verts_Q = Q.vertices
    verts_P = P.vertices
    minkowski_verts = [vP - vQ for vP in verts_P for vQ in verts_Q]
    normals_P = [a for a in P.A]
    normals_Q = [a for a in Q.A]
    directions = normals_P + normals_Q
    inner_mins = []
    debugs = []
    jacobian = np.zeros((len(directions), len(minkowski_verts)))
    for i, d_ in enumerate(directions):
        d = d_ / np.linalg.norm(d_)
        dotprods = [np.dot(d, v) for v in minkowski_verts]
        dist, grad = smoothMinListWithGradient(dotprods, r=r)
        inner_mins.append(dist)
        debugs.append(dotprods)
        jacobian[i, :] = grad
    dist, grad = smoothMaxListWithGradient(inner_mins, r=r)
    grad = grad @ jacobian
    return dist, grad, (inner_mins, debugs, minkowski_verts, directions)


for i in range(imax):
# for i in [1360]:
    dx = i * dt / T
    angle = 8 * np.pi * i * dt / T
    translate = displacement.reshape(-1, 1) + np.array([dx, 0.0, 0.0]).reshape(-1, 1)
    # translate = np.array([dx, 1.5*dx, 1e-1*dx]).reshape(-1, 1)
    R = np.array(ub.Utils.rotx(angle) @ ub.Utils.roty(angle) @ ub.Utils.rotz(2*angle))[
        :3, :3
    ]
    # R = np.array(ub.Utils.roty(angle))[:3, :3]
    # R = np.eye(3)

    R2 = ub.Utils.trn([dx/2, 0, 0]) @ ub.Utils.rotx(4 * np.pi * i * dt / T)
    ubP.add_ani_frame(time=i * dt, htm=R2)

    # A_new = A @ R.T
    # b_new = Q.b.reshape(-1, 1) + A_new @ translate
    # Q_new = Polytope(A_new, b_new)
    htm = pR2htm(R @ translate, R)
    # htm = pR2htm(translate, np.eye(3)) @ pR2htm(np.zeros((3, 1)), R)
    # htm = pR2htm(np.zeros((3, 1)), R) @ np.array(ub.Utils.trn(translate.ravel()))
    htm_Q = np.array(ubQ.htm)
    # print(R)
    ubQ.add_ani_frame(time=i * dt, htm=htm)

    # PmQ, *_ = Polytope.minkowski_subtraction(P, Q_new)
    # ubPmQ = ConvexPolytope(A=PmQ.A, b=PmQ.b, color="green", opacity=0.5)
    # sim.add([ubPmQ])

    # dist, grad, (inner, debugs, verts, dirs) = dist_set2set(P, Q_new)#ubP, ubQ)
    dist, grad, (inner, debugs, verts, dirs) = dist_box2box(ubP, ubQ)#ubP, ubQ)
    idx = np.where(np.array(inner) > 0)[0]
    # for v in verts:
    #     sim.add(
    #         ub.Ball(
    #             radius=0.01, htm=pR2htm(v.reshape(-1, 1), np.eye(3)), color="purple"
    #         )
    #     )
    # bad_direction = np.array(dirs)[idx]
    # if bad_direction.ndim == 1:
    #     sim.add(
    #         ub.Arrow(
    #             origin=np.zeros((3, 1)),
    #             vector=bad_direction.reshape(-1, 1),
    #             color="orange",
    #         )
    #     )
    nearP, nearQ, euclidean_dist, *_ = ubP.compute_dist(ubQ)
    nearP = np.array(nearP).reshape(-1, 1)
    nearQ = np.array(nearQ).reshape(-1, 1)
    # print(PmQ.A @ (nearP - nearQ) - PmQ.b.reshape(-1, 1))
    # print(debug)
    hist_dist.append(dist)
    hist_euc_dist.append(euclidean_dist)
    hist_gradnorm.append(np.linalg.norm(grad))
    # hist_val.append(val)

print(np.array(debugs)[idx])
print(len(verts))


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
