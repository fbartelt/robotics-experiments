# %%
import torch
import time
import numpy as np
import uaibot as ub
import torch.nn.functional as F
from uaibot_cpp_bind import smooth_min, smooth_max
import random as rnd


# Correct
def holder_min(x, g, dim=1, eps=1e-12):
    p = g + 1.0
    m = x.min(dim=dim, keepdim=True).values

    if torch.all(m > 0):
        y = x / m.clamp_min(eps)
        out = m * y.pow(-p).sum(dim=dim, keepdim=True).pow(-1.0 / p)

    elif torch.all(m < 0):
        M = (-m).clamp_min(eps)  # since M = max(-x) = -min(x)
        y = torch.clamp(-x / M, min=0.0)
        out = -M * y.pow(p).sum(dim=dim, keepdim=True).pow(1.0 / p)

    else:
        # mixed batch case: falls back to safe elementwise selection
        y_pos = x / m.clamp_min(eps)
        pos = m * y_pos.pow(-p).sum(dim=dim, keepdim=True).pow(-1.0 / p)

        M = (-m).clamp_min(eps)
        y_neg = torch.clamp(-x / M, min=0.0)
        neg = -M * y_neg.pow(p).sum(dim=dim, keepdim=True).pow(1.0 / p)

        out = torch.where(m > 0, pos, torch.where(m < 0, neg, torch.zeros_like(m)))

    return out.squeeze(dim)


# Correct
def holder_max(x, g, dim=1, eps=1e-12):
    return -holder_min(-x, g, dim, eps)


def pairwise_cross(edges_1, edges_2, eps=1e-12):
    O1, E1, _ = edges_1.shape
    O2, E2, _ = edges_2.shape

    result = torch.cross(
        edges_1[:, None, :, None, :],  # (O1,1,E1,1,3)
        edges_2[None, :, None, :, :],  # (1,O2,1,E2,3)
        dim=-1,
    )  # (O1,O2,E1,E2,3)

    # Normalize along the last axis (the 3D vector)
    result = F.normalize(result, p=2, dim=-1, eps=eps)

    return result.reshape(O1, O2, E1 * E2, 3)  # (O1,O2,E1*E2,3)


def pairwise_difference(vertex_1, vertex_2):

    O1, V1, _ = vertex_1.shape
    O2, V2, _ = vertex_2.shape

    result = (
        vertex_1[:, None, :, None, :]  # (O1,1,V1,1,3)
        - vertex_2[None, :, None, :, :]  # (1,O2,1,V2,3)
    )  # (O1,O2,V1,V2,3)

    print(f"Vertices1: {vertex_1.shape}")
    print(f"Vertices2: {vertex_2.shape}")
    print(f"result: {result.shape}")

    return result.reshape(O1, O2, V1 * V2, 3)  # (O1,O2,V1*V2,3)


def pairwise_direction_vertex_dot(direction, vertices):
    # Save original shape, flatten outer dims
    o1, o2, N, _ = direction.shape
    _, _, V, _ = vertices.shape

    # Reshape to (o1*o2, N, 3) and (o1*o2, V, 3)
    dir_flat = direction.reshape(-1, N, 3)
    vert_flat = vertices.reshape(-1, V, 3)

    # Matrix multiply: (o1*o2, N, 3) × (o1*o2, 3, V) -> (o1*o2, N, V)
    result_flat = torch.matmul(dir_flat, vert_flat.transpose(-2, -1))

    # Reshape back to (o1, o2, N, V)
    return result_flat.reshape(o1, o2, N, V)


def phi(x, k):
    # return x**3 / (x**2 + 1e-3)
    print(f"k: {k}")
    print(f"Phi received {x.shape}: {x}")
    eps = 1e-6
    # eps = 0.0
    abs_x_pow = torch.abs(x) ** k
    res = x * abs_x_pow / (abs_x_pow + eps)
    # res = x * (abs(x)**k) / ((abs(x) ** k) + (1e-46))
    # res = torch.where(torch.isnan(res), torch.zeros_like(res), res)  # nan -> 0
    print(f"Phi computed {res.shape}: {res}")
    return res


def group_direction_vertices(x, g, eps=1e-12):
    # Group vertices
    print("minmax")
    print(f"x: {x.shape}")
    x1 = holder_min(x, g, dim=3, eps=eps)  # (O1, O2, N)
    print(f"x1: {x1.shape}")
    # x1_comp = []
    # for x_ in x[0, 0, :]:
    #     x_array = x_.cpu().detach().numpy()
    #     x1_comp.append(smooth_min(x_array.ravel(), r=g)[0])
    # x1_comp = np.array(x1_comp)
    # print(f"x1_comp: {x1_comp.shape}, {x1_comp}")

    # Group normals
    x2 = holder_max(phi(x1, g), g, dim=2, eps=eps)  # (O1, O2)
    print(f"x2: {x2.shape}")

    return phi(x2, g)


def pairwise_concat_objects(a, b):

    n1, no1, d1 = a.shape
    n2, no2, d2 = b.shape

    assert d1 == 3 and d2 == 3

    # (n1,1,no1,3) -> (n1,n2,no1,3)
    a_exp = a[:, None, :, :].expand(n1, n2, no1, 3)

    # (1,n2,no2,3) -> (n1,n2,no2,3)
    b_exp = b[None, :, :, :].expand(n1, n2, no2, 3)

    # concatenate along object dimension  (n1,n2,no1+no2,3)
    return torch.cat([a_exp, b_exp], dim=2)


def holder_dist(vertexA, edgesA, normalsA, vertexB, edgesB, normalsB, g):
    print(f"Vertices A: {vertexA.shape}")
    print(f"Vertices B: {vertexB.shape}")
    print(f"Edges A: {edgesA.shape}")
    print(f"Edges B: {edgesB.shape}")
    edges_AB = pairwise_cross(edgesA, edgesB)
    print(f"Edges shape: {edges_AB.shape}")
    edges_AB = torch.cat([edges_AB, -edges_AB], dim=2)
    print(f"Edges shape: {edges_AB.shape}")
    normals_AB = pairwise_concat_objects(normalsA, normalsB)
    vertices_AB = pairwise_difference(vertexA, vertexB)
    print(f"Paiwwaise diff: {vertices_AB.shape}")

    direction_AB = torch.cat([edges_AB, normals_AB], dim=2)

    print(f"Normals shape: {direction_AB.shape}")
    pnv = pairwise_direction_vertex_dot(direction_AB, vertices_AB)
    print(f"PNV: {pnv.shape},")

    dist = group_direction_vertices(pnv, g)
    print(f"dist: {dist}")

    return dist


def extract_comp_from_box(htm, lx, ly, lz):

    x = htm[0:3, 0]
    y = htm[0:3, 1]
    z = htm[0:3, 2]
    s = htm[0:3, 3]

    vertices = torch.tensor(
        np.hstack(
            [
                s + 0.5 * lx * x + 0.5 * ly * y + 0.5 * lz * z,
                s + 0.5 * lx * x + 0.5 * ly * y - 0.5 * lz * z,
                s + 0.5 * lx * x - 0.5 * ly * y + 0.5 * lz * z,
                s + 0.5 * lx * x - 0.5 * ly * y - 0.5 * lz * z,
                s - 0.5 * lx * x + 0.5 * ly * y + 0.5 * lz * z,
                s - 0.5 * lx * x + 0.5 * ly * y - 0.5 * lz * z,
                s - 0.5 * lx * x - 0.5 * ly * y + 0.5 * lz * z,
                s - 0.5 * lx * x - 0.5 * ly * y - 0.5 * lz * z,
            ]
        ),
        dtype=torch.float32,
    ).T.contiguous()  # (3, 8)

    # Edge direction vectors – one per unique undirected edge line,
    # keeping the C++ Box convention: 6 oriented unit vectors (±X,±Y,±Z)
    edges = torch.tensor(
        np.hstack([x, x, x, x, y, y, y, y, z, z, z, z]), dtype=torch.float32
    ).T.contiguous()  # shape (3,12)
    # edges = edges / (edges.norm(dim=0, keepdim=True) + 1e-12)   # unit length

    # Face normals: ±x, ±y, ±z (unit length)
    normals = torch.tensor(
        np.hstack([x, -x, y, -y, z, -z]), dtype=torch.float32
    ).T.contiguous()  # shape (3,6)
    # normals = normals / (normals.norm(dim=0, keepdim=True) + 1e-12)

    # edges = np.hstack([+lx * x, -lx * x, +ly * y, -ly * y, +lz * z, -lz * z]).T
    # normals = np.matrix(edges)
    print(
        f"Vertices shape {vertices.shape}, edges {edges.shape}, normals {normals.shape}"
    )

    return vertices, edges, normals


# %%
obsA = []
obsB = []
ubobj1, ubobj2 = None, None

for i in range(1):
    # htm = ub.Utils.htm_rand([-2, -2, -2], [2, 2, 2])
    htm = ub.Utils.trn([0, 0, 0.0])
    lx = rnd.uniform(0.3, 0.5)
    ly = rnd.uniform(0.3, 0.5)
    lz = rnd.uniform(0.3, 0.5)
    lx, ly, lz = 1.0, 1.0, 1.0
    ubobj1 = ub.Box(htm=htm, width=lx, depth=ly, height=lz)
    obsA.append(extract_comp_from_box(htm, lx, ly, lz))
for j in range(1):
    # htm = ub.Utils.htm_rand([-2, -2, -2], [2, 2, 2])
    # htm = ub.Utils.rotx(np.pi/2)
    htm = ub.Utils.trn([0.1, .1, .1]) @ ub.Utils.rotx(np.pi / 4) @ ub.Utils.roty(np.pi / 4) @ ub.Utils.rotz(np.pi / 4)
    lx = rnd.uniform(0.3, 0.5)
    ly = rnd.uniform(0.3, 0.5)
    lz = rnd.uniform(0.3, 0.5)
    lx, ly, lz = 1.0, 1.0, 1.0
    ubobj2 = ub.Box(htm=htm, width=lx, depth=ly, height=lz)
    obsB.append(extract_comp_from_box(htm, lx, ly, lz))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(device)


vA = torch.tensor(np.array([x[0] for x in obsA])).to(device)
eA = torch.tensor(np.array([x[1] for x in obsA])).to(device)
nA = torch.tensor(np.array([x[2] for x in obsA])).to(device)

vB = torch.tensor(np.array([x[0] for x in obsB])).to(device)
eB = torch.tensor(np.array([x[1] for x in obsB])).to(device)
nB = torch.tensor(np.array([x[2] for x in obsB])).to(device)

start = time.perf_counter()

# r = 0.1
# gamma = (1 - r) / r
gamma = 2.0
r = 1 / (gamma + 1)
# r = 0.1 = 1/(gamma + 1)
# r*gamma + r = 1
# gamma = (1 - r)/ro
dist_torch = 0.0

for i in range(1):
    dist, *_ = ubobj1.signed_distance(
        ubobj2, gamma=gamma, epsilon=1e-46, skip_gradient=True, is_conservative=False
    )
    dist_torch = holder_dist(vA, eA, nA, vB, eB, nB, gamma)

print(f"Torch distance: {dist_torch}\ndist: {dist}")

ubobj1
ubobj2
end = time.perf_counter()

print(f"Elapsed time: {(end-start)/50:.6f} seconds")

# %%
x = torch.tensor(
    [
        # first "batch" (index 0) of shape (2,3,4)
        [
            # row 0, depth 0: all positive, min=1
            [
                [1.0, 2.0, 3.0, 4.0],
                # row 0, depth 1: contains zero, min=0
                [5.0, 0.0, 6.0, 7.0],
                # row 0, depth 2: negative min, min=-2
                [8.0, -1.0, -2.0, 9.0],
            ],
            # row 1, depth 0: all positive, min=0.5
            [
                [0.5, 1.5, 2.5, 3.5],
                # row 1, depth 1: contains zero, min=0
                [0.0, 4.0, 5.0, 6.0],
                # row 1, depth 2: negative min, min=-3
                [1.0, -3.0, 2.0, 0.0],
            ],
        ]
    ],
    dtype=torch.float32,
)
x.shape
res = holder_min(x, 2, dim=3, eps=1e-12)
print(res.shape, res)
res2 = holder_max(res, 2, dim=2, eps=1e-12)
print(res2.shape, res2)
