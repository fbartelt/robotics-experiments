import sys, os
import numpy as np
import plotly.graph_objects as go

sys.path.append(os.path.dirname("/home/fbartelt/Documents/Projetos/robotics-experiments/smoothdist/test_nonconvexdist.py"))
from polygon import Polytope, NonConvexPolygon, add_polygon
from distances import e_s_hat, smooth_min, holder_mean


def create_level_sets(
        polygon,
        eps=1e-3,
        r=1e-1,
        h=1e-1,
        eta=1.0,
        kind='both',
        bbox=(-5, -5, 5, 5),
        n_points=100,
        n_countours=50,
        ignore=[],
        test_=False):
    fig = go.Figure()

    centroid_dists = []
    for poly in polygon.polytopes:
        A = poly.A.copy()
        b = poly.b.copy()
        add_polygon(fig, A, b)
        aux, *_ = e_s_hat(
            poly.centroid.reshape(-1, 1),
            A,
            b,
            kind=kind,
            eps=eps,
            r=r,
            h=h,
            eta=eta,
        )
        poly.dist_to_centroid = abs(aux)
        centroid_dists.append(abs(poly.dist_to_centroid))
    centroid_dist = np.max(centroid_dists)

    x_min, y_min, x_max, y_max = bbox
    p1 = np.linspace(x_min, x_max, n_points)
    p2 = np.linspace(y_min, y_max, n_points)
    P1, P2 = np.meshgrid(p1, p2)
    P = np.vstack([P1.ravel(), P2.ravel()]).T
    distances = []

    for j, pi_ in enumerate(P):
        # print(j)
        pi = pi_.copy().reshape(-1, 1)
        dists_wo_const, dists_w_const = [], []
        for i, poly in enumerate(polygon.polytopes):
            if i == -1:
                continue
            A = poly.A.copy()
            b = poly.b.copy()
            # shared = polygon.shared_boundaries[i, :] # array with lists or 0.0
            shared = polygon.shared_boundaries[i]
            dist2polyi, *_ = e_s_hat(
                pi,
                A,
                b,
                kind='in',
                eps=eps,
                r=r,
                h=h,
                eta=eta,
            )
            dists_w_const.append(dist2polyi / centroid_dist)
            dist_masks = []
            for k, list_s in enumerate(shared):
                s = np.array(list_s, dtype=int)
                if j == 0:
                    print(f"Polygon {i} shared with {k} at index {s}: {shared}")
                    print(f"poly.shared_boundaries: {polygon.shared_boundaries}")
                # Ignore rows of A with indices in s:
                if np.any(s >= 0):
                    mask = np.ones(A.shape[0], dtype=bool)
                    # Remove all -1 from s
                    s_ = s[s >= 0]
                    mask[s_] = False
                    A_ = A[mask, :]
                    b_ = b[mask]
                    dist2polyi_masked, *_ = e_s_hat(
                        pi,
                        A_,
                        b_,
                        kind="in",
                        eps=eps,
                        r=r,
                        h=h,
                        eta=eta,
                    )
                    polyk = polygon.polytopes[k]
                    A_k = polygon.polytopes[k].A.copy()
                    b_k = polygon.polytopes[k].b.copy()
                    # Theres only one shared boundary if both are convex
                    shared_ki = polygon.shared_boundaries[k][i]
                    shared_ki = np.array(shared_ki, dtype=int)
                    shared_ki = shared_ki[shared_ki >= 0]
                    mask_k = np.ones(A_k.shape[0], dtype=bool)
                    mask_k[shared_ki] = False
                    A_k_ = A_k[mask_k, :]
                    b_k_ = b_k[mask_k]
                    dist2polyk_masked, *_ = e_s_hat(
                        pi,
                        A_k_,
                        b_k_,
                        kind="in",
                        eps=eps,
                        r=r,
                        h=h,
                        eta=eta,
                    )
                    dist2polyi_masked = dist2polyi_masked / centroid_dist
                    dist2polyk_masked = dist2polyk_masked / centroid_dist
                    dist_masks.append(-holder_mean(list(map(abs, [dist2polyi_masked, dist2polyk_masked])), r=r))
                    # dist_masks.append()
            # dists_wo_const.append(smooth_min(dist_masks, r=r))
            min_mask = smooth_min(dist_masks, r=r)
            test_mask = -holder_mean(list(map(abs, dist_masks)), r=r/10)
            short_mask = -holder_mean(list(map(abs, dist_masks)) + [abs(min_mask)], r=r/10)
            dists_wo_const.append(min_mask)
        # dist_wo_const = -holder_mean(list(map(abs, dists_wo_const)), r=r)
        dist_wo_const = smooth_min(dists_wo_const, r=r)
        dist_w_const = smooth_min(dists_w_const, r=r)
        distances.append(smooth_min(dist_wo_const, dist_w_const, r=r))
        # distances.append(dist_wo_const)

    distances = np.array(distances).reshape(P1.shape)
    contour = go.Contour(
        x=p1,
        y=p2,
        z=distances,
        colorscale="RdBu",
        ncontours=n_countours,
        name="Level Sets",
    )
    fig.add_trace(contour)
    # Update layout
    fig.update_layout(
        xaxis_title="x", yaxis_title="y", showlegend=False, width=800, height=800,
        xaxis_range=[bbox[0], bbox[2]], yaxis_range=[bbox[1], bbox[3]],
        margin=dict(t=0, l=10, r=10, b=10)
    )

    return fig



n_points = 100
max_iters = 100
h = 0.1
r = 0.1
eps = 5e-2
bulge = True
min_path = True
k = 5e-1
eta=10.0
n_contours = 70
max_iters = 200
# bounding_box = (-6, -6, 6, 6)
bounding_box = (-1, -1, 5, 5)

seed = 42 # 100 is cool

# Bottom horizontal
import numpy as np
A1 = np.array([
    [1.0, 0], # x <= 2
    [-1, 0],  # x >= 0
    [0, -1],  # y >= 0
    [0, 1],   # y <= 1
])
b1 = np.array([2.0, 0, 0, 1])
# b1 = np.array([2.0, 2.0, 0, 1])
# Vertical
A2 = np.array([
    [1.0, 0],  # x <= 1
    [-1, 0],   # x >= 0
    [0, 1],    # y <= 3
    [0, -1],   # y >= 1
])
b2 = np.array([1.0, 0, 3, -1])
# Top horizontal
A3 = np.array([
    [1.0, 0], # x <= 2
    [-1, 0],  # x >= 0
    [0, -1],  # y >= 3
    [0, 1],   # y <= 4
])
b3 = np.array([2.0, 0, -3, 4])
A4 = np.array([
    [-1.0, 0],  # x >= 2
    [-1, 1],    # -x + y <= 2
    [1, 1],     # x + y <= 8
    [1, -1],    # x - y <= 2
    [-2, -1]    # 2x + y >= 7
])
b4 = np.array([-2.0, 2, 8, 2, -7])
A5 = np.array([
    [1.0, 0],  # x <= 0
    [-1, -1],  # x + y >= 1
    [-1, 1]    # -x + y <= 3
])
b5 = np.array([0.0, -1, 3])
A6 = np.array([
    [1.0, 0],   # x <= 4
    [-1.0, 0],  # x >= 2
    [0, -1.0],  # y >= 1
    [0, 1.0],   # y <= 5
])
b6 = np.array([4.0, -2.0, 1.0, 5.0])
A_list = [
    A1,
    A2,
    A3,
    A4,
    # A6,
    A5
]
b_list = [
    b1,
    b2,
    b3,
    b4,
    # b6,
    b5
]
shared_boundaries = (np.ones((len(b_list), len(b_list), 1), dtype=int) * -1).tolist()
# 3d array S[i, j, k] means that the k-th constraint of the i-th polytope
# is a common boundary with the j-th polytope.
# S[i, j] is a list that contains the indices of each constraint shared
# S[j, i] can be different from S[i, j] given that they are defined from
# each matrix A_i and A_j.
# shared_boundaries[0, 1] = 3
# shared_boundaries[1, 0] = 3
# shared_boundaries[1, 2] = 2
# shared_boundaries[2, 1] = 2
# shared_boundaries[2, 3] = 0
# shared_boundaries[3, 2] = 0
# shared_boundaries[1, 4] = 1
# shared_boundaries[4, 1] = 0
shared_boundaries[0][1] = [3]
shared_boundaries[1][0] = [3]
shared_boundaries[1][2] = [2]
shared_boundaries[2][1] = [2]
shared_boundaries[2][3] = [0] #, 2, 3]
shared_boundaries[3][2] = [0, 1, 4]
### Pentagon -> long rectangle ###
# shared_boundaries[0][3] = [0]
# shared_boundaries[3][0] = [1]
# shared_boundaries[2][3] = [0]
# shared_boundaries[3][2] = [1]
### EDIT ###
# shared_boundaries[1][4] = [1, 2, 3] # More triangular shape
shared_boundaries[1][4] = [1] # More trapezoidal shape
shared_boundaries[4][1] = [0]
# print(shared_boundaries[3])

polygon = NonConvexPolygon(A_list, b_list, shared_boundaries)
fig = create_level_sets(
    polygon,
    eps=eps,
    r=r,
    h=h,
    eta=eta,
    kind='both',
    bbox=bounding_box,
    n_points=n_points,
    n_countours=n_contours,
)
fig.show()
