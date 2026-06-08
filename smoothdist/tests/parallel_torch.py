# %%
import torch
import time
import itertools
import numpy as np
import uaibot as ub
import torch.nn.functional as F
import random as rnd
from uaibot_cpp_bind import smooth_min, smooth_max
from scipy.spatial import ConvexHull, HalfspaceIntersection


def random_htm(
    translation_max=[1.0, 1.0, 1.0],
    translation_min=[-1.0, -1.0, -1.0],
    rotation_max=[2 * np.pi, 2 * np.pi, 2 * np.pi],
    rotation_min=[0.0, 0.0, 0.0],
    rng=None,
    seed=None,
):
    if rng is None:
        if seed is not None:
            rng = np.random.default_rng(seed)
        else:
            rng = np.random.default_rng()
    translation = rng.uniform(translation_min, translation_max)
    rotation = rng.uniform(rotation_min, rotation_max)
    R = (
        ub.Utils.rotx(rotation[0])
        @ ub.Utils.roty(rotation[1])
        @ ub.Utils.rotz(rotation[2])
    )
    htm = np.array(ub.Utils.trn(translation) @ R)
    return htm


def create_platonic_solid(n_faces, radius=1.0, *args, **kwargs):
    """
    Create a platonic solid with a given number of faces.

    Parameters
    ----------
    n_faces : int
        Number of faces. Supported: 4 (tetrahedron), 6 (cube), 8
        (octahedron), 12 (dodecahedron), 20 (icosahedron).
    radius : float, optional
        Circumscribed sphere radius. Default is 1.0.
    *args, **kwargs
        Additional arguments passed to the ConvexPolytope constructor,
        such as 'htm' for the homogeneous transformation matrix.

    Returns
    -------
    ub.ConvexPolytope
    """
    phi = (1 + np.sqrt(5)) / 2  # golden ratio

    # Canonical vertices with circumradius = 1 (local frame, center at origin)
    match n_faces:
        case 4:  # tetrahedron
            v = np.array(
                [[1.0, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]]
            ) / np.sqrt(3)

        case 6:  # cube
            v = np.array(
                [
                    [-1.0, -1, -1],
                    [-1, -1, 1],
                    [-1, 1, -1],
                    [-1, 1, 1],
                    [1, -1, -1],
                    [1, -1, 1],
                    [1, 1, -1],
                    [1, 1, 1],
                ]
            ) / np.sqrt(3)

        case 8:  # octahedron
            v = np.array(
                [[1.0, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]]
            )  # already radius 1

        case 12:  # dodecahedron
            # all 20 vertices have norm √3 – scale to 1
            verts = []
            # (±1, ±1, ±1)
            for x in (-1, 1):
                for y in (-1, 1):
                    for z in (-1, 1):
                        verts.append([x, y, z])
            # (0, ±φ, ±1/φ) and cyclic permutations
            for a in (-phi, phi):
                for b in (-1 / phi, 1 / phi):
                    verts.append([0, a, b])
                    verts.append([b, 0, a])
                    verts.append([a, b, 0])
            v = np.array(verts) / np.sqrt(3)

        case 20:  # icosahedron
            verts = []
            norm_factor = np.sqrt(1 + phi**2)
            # cyclic permutations of (0, ±1, ±φ)
            for a in (-1, 1):
                for b in (-phi, phi):
                    verts.append([0, a, b])
                    verts.append([b, 0, a])
                    verts.append([a, b, 0])
            v = np.array(verts) / norm_factor

        case _:
            raise ValueError(
                f"Unsupported number of faces: {n_faces}. "
                "Must be one of {4, 6, 8, 12, 20}."
            )

    # Apply radius scaling
    v *= radius
    # Uaibot already converts A, b to world frame using the htm, so we
    # can just compute in local frame and let it handle the transformation.

    # Compute half‑space representation (Ax ≤ b) from convex hull
    hull = ConvexHull(v)
    A = hull.equations[:, :3]  # outward normals
    b = -hull.equations[:, 3]  # right‑hand side
    eqs = hull.equations
    # Normalize so that normal is unit (already unit if from Qhull) and offset sign consistent
    # For outward normals, all offset signs should be positive? Not necessarily, but we can
    # group by the plane parameters. Better: round to some tolerance.
    unique_planes = {}
    tolerance = 1e-10
    for eq in eqs:
        # round to tolerance to group identical planes
        key = tuple(np.round(eq / np.linalg.norm(eq[:3]), decimals=10))
        if key not in unique_planes:
            unique_planes[key] = eq
    A = np.array([v[:3] for v in unique_planes.values()])
    b = -np.array([v[3] for v in unique_planes.values()])
    # Now A has one row per distinct face
    # print(f"Created {n_faces}-face solid with {len(A)} unique planes (originally {len(hull.equations)})")
    htm = kwargs.pop("htm", np.eye(4))
    ubobj = ub.ConvexPolytope(A=A, b=b, htm=np.eye(4), *args, **kwargs)
    ubobj.set_ani_frame(htm=htm)
    return ubobj, A, b


def extract_comp_from_platonic(htm, A, b, rtol=1e-5):
    """
    Extract vertices, true edge directions, and face normals from a convex polyhedron.

    Parameters
    ----------
    htm : np.ndarray (4,4)
        Homogeneous transformation matrix (world frame).
    A : np.ndarray (F,3)
        Outward unit normals in local frame.
    b : np.ndarray (F,)
        Face offsets (distance from origin to plane) in local frame.
    rtol : float
        Relative tolerance for edge‑length equality.

    Returns
    -------
    vertices : torch.Tensor (V, 3)
        Vertices in world frame.
    edges : torch.Tensor (E, 3)
        Unit direction vectors of each undirected edge in world frame.
    normals : torch.Tensor (F, 3)
        Outward unit normals of each face in world frame.
    """
    # 1. Local vertices from half‑space intersection
    halfspaces = np.hstack([A, -b.reshape(-1, 1)])
    hs = HalfspaceIntersection(halfspaces, np.zeros(3))
    local_verts = hs.intersections  # (V, 3)
    local_verts = np.unique(local_verts.round(decimals=10), axis=0)

    # 2. Find true edges by minimal positive distance (like C++ logic)
    nv = local_verts.shape[0]
    # all pairwise squared distances
    diff = local_verts[:, None, :] - local_verts[None, :, :]  # (V, V, 3)
    sq_dist = np.sum(diff * diff, axis=-1)  # (V, V)

    # Ignore zero (self) and find minimal positive squared length
    mask = sq_dist > 1e-12
    if not mask.any():
        raise ValueError("No edges found – degenerate vertices?")
    edge_len_sq = np.min(sq_dist[mask])
    tol = edge_len_sq * rtol

    # Select all unordered pairs with that length
    edge_set = set()
    for i in range(nv):
        for j in range(i + 1, nv):
            if abs(sq_dist[i, j] - edge_len_sq) < tol:
                edge_set.add((i, j))

    edges_idx = np.array(list(edge_set))  # (E, 2)
    # Edge direction vectors (unit) – shape (E, 3)
    edge_vectors = local_verts[edges_idx[:, 1]] - local_verts[edges_idx[:, 0]]
    edge_vectors /= np.linalg.norm(edge_vectors, axis=1, keepdims=True)
    local_edges = edge_vectors

    # 3. Face normals are already unit – shape (F, 3)
    local_normals = A

    # 4. Transform to world frame
    R = htm[:3, :3]
    t = htm[:3, 3]
    world_vertices = (R @ local_verts.T).T + t  # (V, 3)
    world_edges = (R @ local_edges.T).T  # (E, 3)
    world_normals = (R @ local_normals.T).T  # (F, 3)

    # 5. Return float32 tensors
    return (
        torch.tensor(world_vertices, dtype=torch.float32),
        torch.tensor(world_edges, dtype=torch.float32),
        torch.tensor(world_normals, dtype=torch.float32),
    )


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


def phi(x, k, eps=1e-3):
    abs_x_pow = torch.abs(x) ** k
    res = x * abs_x_pow / (abs_x_pow + eps)
    return res


def group_direction_vertices(x, g, eps=1e-3):
    # Group vertices
    x1 = holder_min(x, g, dim=3, eps=eps)  # (O1, O2, N)

    # Group normals
    x2 = holder_max(phi(x1, g, eps=eps), g, dim=2, eps=eps)  # (O1, O2)

    return phi(x2, g, eps=eps)


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


def holder_dist(vertexA, edgesA, normalsA, vertexB, edgesB, normalsB, g, eps=1e-3):
    edges_AB = pairwise_concat_objects(edgesA, edgesB)
    edges_AB = torch.cat([edges_AB, -edges_AB], dim=2)
    normals_AB = pairwise_concat_objects(normalsA, normalsB)
    vertices_AB = pairwise_difference(vertexA, vertexB)

    direction_AB = torch.cat([edges_AB, normals_AB], dim=2)

    pnv = pairwise_direction_vertex_dot(direction_AB, vertices_AB)

    dist = group_direction_vertices(pnv, g, eps=eps)

    return dist


def extract_comp_from_box(htm, lx, ly, lz):

    htm = np.array(htm)
    x = htm[0:3, 0].reshape(3, 1)
    y = htm[0:3, 1].reshape(3, 1)
    z = htm[0:3, 2].reshape(3, 1)
    s = htm[0:3, 3].reshape(3, 1)

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
    ).T.contiguous()  # (8, 3) after tranpose

    # Edge direction vectors – one per unique undirected edge line,
    # keeping the C++ Box convention: 6 oriented unit vectors (±X,±Y,±Z)
    edges = torch.tensor(
        np.hstack([x, x, x, x, y, y, y, y, z, z, z, z]), dtype=torch.float32
    ).T.contiguous()  # shape (12,3) after transpose

    # Face normals: ±x, ±y, ±z (unit length)
    normals = torch.tensor(
        np.hstack([x, -x, y, -y, z, -z]), dtype=torch.float32
    ).T.contiguous()  # shape (6,3) after tranpose

    # Normals and edges are already unit vectors
    return vertices, edges, normals


# %%
ubobj1, ubobj2 = None, None
t_max = [2.0, 2.0, 2.0]
t_min = ([-2.0, -2.0, -2.0],)
r_max = [2 * np.pi, 2 * np.pi, 2 * np.pi]
r_min = [0.0, 0.0, 0.0]
max_radius = 1.0
min_radius = 1e-3
gamma = 2
faces = [4, 6, 8, 12, 20]
platonic_solids = {4: "tetra", 6: "cube", 8: "octa", 12: "dodeca", 20: "icosa"}

epsilon = 1e-3

N = 100
# n_facesA = 12
# n_facesB = 12
for n_facesA, n_facesB in itertools.combinations_with_replacement(faces, 2):
    obsA = []
    obsB = []
    for i in range(N):
        rng = np.random.default_rng(seed=i)  # different seed for each pair
        htm = random_htm(t_max, t_min, r_max, r_min, rng)
        radius = rng.uniform(min_radius, max_radius)
        ubobj1, A, b = create_platonic_solid(n_facesA, radius, htm=htm)
        obsA.append(extract_comp_from_platonic(htm, A, b))

        htm = random_htm(t_max, t_min, r_max, r_min, rng)
        radius = rng.uniform(min_radius, max_radius)
        ubobj2, A, b = create_platonic_solid(n_facesB, radius, htm=htm)
        obsB.append(extract_comp_from_platonic(htm, A, b))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # print(device)

    vA = torch.tensor(np.array([x[0] for x in obsA])).to(device)
    eA = torch.tensor(np.array([x[1] for x in obsA])).to(device)
    nA = torch.tensor(np.array([x[2] for x in obsA])).to(device)

    vB = torch.tensor(np.array([x[0] for x in obsB])).to(device)
    eB = torch.tensor(np.array([x[1] for x in obsB])).to(device)
    nB = torch.tensor(np.array([x[2] for x in obsB])).to(device)

    # print(f"edges A shape: {eA.shape} edges B shape: {eB.shape}")
    # print(f"normals A shape: {nA.shape} normals B shape: {nB.shape}")

    gamma = 2.0
    dist_torch = 0.0

    # Warm-up (hide compilation & caching overhead)
    for _ in range(10):
        _ = holder_dist(vA, eA, nA, vB, eB, nB, gamma, eps=epsilon)
    torch.cuda.synchronize()

    start = time.perf_counter()
    n_repeats = 50
    for i in range(n_repeats):
        dist_torch = holder_dist(vA, eA, nA, vB, eB, nB, gamma, eps=epsilon)
    torch.cuda.synchronize()  # ensure all GPU work finished
    elapsed = time.perf_counter() - start

    avg_time_per_batch = elapsed / n_repeats
    avg_time_per_pair = avg_time_per_batch / (N * N)
    print(f"Batch size {N}: {avg_time_per_batch*1000:.6f} ms total")
    print(
        f"Per pair ({platonic_solids[n_facesA]} vs {platonic_solids[n_facesB]}): {avg_time_per_pair*1e6:.6f} µs"
    )
