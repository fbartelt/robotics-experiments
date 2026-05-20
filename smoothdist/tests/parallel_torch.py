# %%
import torch
import time
import numpy as np
import uaibot as ub
import random as rnd


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


def holder_max(x, g, dim=1, eps=1e-12):
    return -holder_min(-x, g, dim, eps)


def pairwise_cross(edges_1, edges_2):
    O1, E1, _ = edges_1.shape
    O2, E2, _ = edges_2.shape

    result = torch.cross(
        edges_1[:, None, :, None, :],  # (O1,1,E1,1,3)
        edges_2[None, :, None, :, :],  # (1,O2,1,E2,3)
        dim=-1,
    )  # (O1,O2,E1,E2,3)

    return result.reshape(O1, O2, E1 * E2, 3)  # (O1,O2,E1*E2,3)


def pairwise_difference(vertex_1, vertex_2):

    O1, V1, _ = vertex_1.shape
    O2, V2, _ = vertex_2.shape

    result = (
        vertex_1[:, None, :, None, :]  # (O1,1,V1,1,3)
        - vertex_2[None, :, None, :, :]  # (1,O2,1,V2,3)
    )  # (O1,O2,V1,V2,3)

    return result.reshape(O1, O2, V1 * V2, 3)  # (O1,O2,V1*V2,3)


def pairwise_direction_vertex_dot(direction, vertices):
    result = (
        direction[:, :, :, None, :]  # (O1,O2,N,1,3)
        * vertices[:, :, None, :, :]  # (O1,O2,1,V,3)
    ).sum(
        dim=-1, keepdim=True
    )  # (O1,O2,N,V,1)

    return result.squeeze(-1)  # (O1,O2,N,V)


def phi(x):
    return x**3 / (x**2 + 0.001)


def group_direction_vertices(x, g, eps=1e-12):
    # Group vertices
    x1 = holder_min(x, g, dim=3, eps=eps)  # (O1, O2, N)

    # Group normals
    x2 = holder_max(phi(x1), g, dim=2, eps=eps)  # (O1, O2)

    return phi(x2)


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
    edges_AB = pairwise_cross(edgesA, edgesB)
    normals_AB = pairwise_concat_objects(normalsA, normalsB)
    vertices_AB = pairwise_difference(vertexA, vertexB)

    direction_AB = torch.cat([edges_AB, normals_AB], dim=2)
    pnv = pairwise_direction_vertex_dot(direction_AB, vertices_AB)

    dist = group_direction_vertices(pnv, g)

    return dist


def extract_comp_from_box(htm, lx, ly, lz):

    x = htm[0:3, 0]
    y = htm[0:3, 1]
    z = htm[0:3, 2]
    s = htm[0:3, 3]

    vertices = np.hstack(
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
    ).T

    edges = np.hstack([+lx * x, -lx * x, +ly * y, -ly * y, +lz * z, -lz * z]).T

    normals = np.matrix(edges)

    return vertices, edges, normals


# %%
obsA = []
obsB = []

for i in range(300):
    htm = ub.Utils.htm_rand([-2, -2, -2], [2, 2, 2])
    lx = rnd.uniform(0.3, 0.5)
    ly = rnd.uniform(0.3, 0.5)
    lz = rnd.uniform(0.3, 0.5)
    obsA.append(extract_comp_from_box(htm, lx, ly, lz))
for j in range(320):
    htm = ub.Utils.htm_rand([-2, -2, -2], [2, 2, 2])
    lx = rnd.uniform(0.3, 0.5)
    ly = rnd.uniform(0.3, 0.5)
    lz = rnd.uniform(0.3, 0.5)
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

for i in range(50):
    holder_dist(vA, eA, nA, vB, eB, nB, 1)

end = time.perf_counter()

print(f"Elapsed time: {(end-start)/50:.6f} seconds")
