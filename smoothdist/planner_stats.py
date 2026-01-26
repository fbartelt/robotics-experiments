# %%
import numpy as np
import plotly.graph_objects as go
import plotly.colors as pc
import cyipopt
import time
import pandas as pd
import uaibot as ub  # For QP solver only
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import sys

# add current path to sys.path
import os

# current_dir = os.path.dirname(os.path.abspath(__file__))
# sys.path.append(current_dir)

# from distances import signed_dist2convex, phi, smooth_min
from smoothfunctions import (
    signedDist2Convex,
    smoothMinListWithGradient,
    smoothMinList,
    phi,
    # signedEuclideanDistance
)
from scipy.optimize import minimize
from typing import List, Tuple, Optional, Callable
from polygon import (
    Polytope,
    add_polygon,
    create_level_sets,
    NonConvexPolygon,
    generate_random_polyhedron,
    generate_random_polyhedron_set,
)
from polyhedron import add_polyhedron
from path_planning import (
    deform_path_ipopt,
    add_path,
    add_path3d,
)


def check_free_path(
    path: np.ndarray, obstacles: List[Polytope], r, h, margin: float = 0.0
) -> bool:
    """
    Check if the given path is collision-free with respect to the list of obstacles.
    """
    N = path.shape[0]
    n_obstacles = len(obstacles)

    for i in range(N):
        p_i = path[i].reshape(-1, 1)
        for j, obs in enumerate(obstacles):
            dist_ij, _ = signedDist2Convex(
                p_i,
                obs.A,
                obs.b.reshape(-1, 1),
                r=r,
                eps=h,
                test="",
            )
            if dist_ij < margin:
                print(
                    f"Collision detected at point {i} with obstacle {j}. Distance: {dist_ij}"
                )
                return False  # Collision detected

    return True  # No collisions detected


def path_distance_stats(path, obstacles, r, h, alpha, test=""):
    dists = []
    for p in path:
        p = p.reshape(-1, 1).astype(np.float32)
        d_obs = []
        for obs in obstacles:
            A = obs.A.astype(np.float32)
            b = obs.b.reshape(-1, 1).astype(np.float32)
            d, _ = signedDist2Convex(p, A, b, r=r, eps=h, test=test)
            d_obs.append(d)

        smooth_min_dist = smoothMinList(d_obs, r)
        # Smooth saturation (avoids cheating by circunventing the map)
        exponent = np.clip(-alpha * smooth_min_dist, -100, 100)
        sat_dist = (-1 / alpha) * np.log(0.5 * (1 + np.exp(exponent)))
        dists.append(sat_dist)
        # dists.append(min(d_obs))
    dists = np.array(dists)
    return {
        "min_dist": float(np.min(dists)),
        "mean_dist": float(np.mean(dists)),
        "p10_dist": float(np.percentile(dists, 10)),
        "num_violations": int(np.sum(dists < 0)),
    }


# %%
# Check if path is collision free for a bunch of cases (2D)
seed = 42
rng = np.random.default_rng(seed)
max_vertices = 15
bounding_box = (-20.0, -20, 20, 20)
bounding_box_3d = (-20.0, -20, -20, 20, 20, 20)
max_checks = 2000
radius_limits = (2, 6)
max_attempts = 500
min_area = None
min_volume = None
radius = None
num_vertices = None

n_points = 200
h = 0.01
r = 0.1

opt_max_iters = 200
zeta = 0.5 * 1
alpha = np.log(2) / 5e-2
min_path = True

successes = []
records = []
save_every = 50

intersect = False  # Whether to allow intersecting obstacles


def run_single_case(i, method="ours"):
    # for i in range(max_checks):
    t0_total = time.perf_counter()
    max_polygons = rng.integers(10, 20).item()
    q0 = rng.uniform(bounding_box[0], bounding_box[2], size=(2, 1))
    qd = rng.uniform(bounding_box[0], bounding_box[2], size=(2, 1))
    aux = 1
    # ||qd - q0||^2 >= 800
    while (
        np.linalg.norm(q0 - qd) ** 2
        < 2 * ((bounding_box[2] - bounding_box[0]) / 2) ** 2
    ):
        rng2 = np.random.default_rng(seed + i + aux)
        qd = rng2.uniform(bounding_box[0], bounding_box[2], size=(2, 1))
        aux += 1
    print(f"Got q0 and qd after {aux} attempts.")
    try:
        obstacles = Polytope.random_set(
            n_polytopes=max_polygons,
            intersect_polytopes=intersect,
            q0=q0,
            qd=qd,
            max_vertices=max_vertices,
            radius_lim=radius_limits,
            bbox=bounding_box,
            seed=seed,
            min_area=min_area,
            max_attempts=max_attempts,
            radius=radius,
            num_vertices=num_vertices,
        )
    except Exception as e:
        print(f"Failed to generate obstacles: {e}. Skipping this case.")
        default_dict = {
            "run_id": i,
            "seed": seed,
            "num_obstacles": max_polygons,
            "num_path_points": n_points,
            "ipopt_info": "Generation Failed",
            "total_time": 0,
            "ipopt_time": 0,
            "success_collision_free": False,
            "min_dist": float("nan"),
            "mean_dist": float("nan"),
            "p10_dist": float("nan"),
            "num_violations": -1,
        }
        return default_dict
    print(f"Generated {len(obstacles)} obstacles.")
    lambda_ = np.linspace(0, 1, n_points)
    init_path = (1 - lambda_) * q0 + lambda_ * qd  # (2 x n_points)
    init_path = init_path.T  # (n_points x 2)
    path = init_path.copy()
    delta = 3.0 * np.linalg.norm(path[:, 1] - path[:, 0]) ** 2
    path_hist = [init_path.copy()]
    kind = None

    t0_opt = time.perf_counter()
    path_opt, path_hist, info = deform_path_ipopt(
        init_path,
        obstacles,
        method=method,
        verbose=False,
        max_iter=opt_max_iters,
        kind=kind,
        h=h,
        r=r,
        alpha=alpha,
        zeta=zeta,
        min_path=min_path,
        delta=delta,
    )
    t_opt = time.perf_counter() - t0_opt
    t_total = time.perf_counter() - t0_total

    collision_free = check_free_path(path_opt, obstacles, r=r, h=h, margin=-1e-3)
    dist_stats = path_distance_stats(
        path_opt,
        obstacles,
        r=r,
        h=h,
        alpha=alpha,
        test="" if kind is None else kind,
    )

    record = {
        "run_id": i,
        "seed": seed,
        "num_obstacles": len(obstacles),
        "num_path_points": n_points,
        "ipopt_info": info,
        # "solver_status": info.get("status", None),
        # "solver_converged": info.get("status", 0) == 0,
        # "num_iterations": info.get("iter_count", None),
        # "max_iter_reached": info.get("iter_count", 0) >= opt_max_iters,
        "total_time": t_total,
        "ipopt_time": t_opt,
        "success_collision_free": collision_free,
        **dist_stats,
    }

    return record


def run_single_case3d(i, method="ours"):
    """Run a single test case in 3D."""
    t0_total = time.perf_counter()
    max_polygons = rng.integers(20, 40).item()
    q0 = rng.uniform(bounding_box[0], bounding_box[3], size=(3, 1))
    qd = rng.uniform(bounding_box[0], bounding_box[3], size=(3, 1))
    aux = 1
    # ||qd - q0||^2 >= 800
    while (
        np.linalg.norm(q0 - qd) ** 2
        < 2 * ((bounding_box[3] - bounding_box[0]) / 2) ** 2
    ):
        rng2 = np.random.default_rng(seed + i + aux)
        qd = rng2.uniform(bounding_box[0], bounding_box[3], size=(3, 1))
        aux += 1
    print(f"Got q0 and qd after {aux} attempts.")
    try:
        obstacles = Polytope.random_set_polyhedra(
            n_polyhedra=max_polygons,
            intersect_polyhedra=intersect,
            q0=q0,
            qd=qd,
            max_vertices=max_vertices,
            radius_lim=radius_limits,
            dim=3,
            bbox=bounding_box_3d,
            seed=seed,
            min_volume=min_volume,
            max_attempts=max_attempts,
            radius=radius,
            num_vertices=num_vertices,
        )
    except Exception as e:
        print(f"Failed to generate obstacles: {e}. Skipping this case.")
        default_dict = {
            "run_id": i,
            "seed": seed,
            "num_obstacles": max_polygons,
            "num_path_points": n_points,
            "ipopt_info": "Generation Failed",
            "total_time": 0,
            "ipopt_time": 0,
            "success_collision_free": False,
            "min_dist": float("nan"),
            "mean_dist": float("nan"),
            "p10_dist": float("nan"),
            "num_violations": -1,
        }
        return default_dict
    print(f"Generated {len(obstacles)} obstacles.")
    lambda_ = np.linspace(0, 1, n_points)
    init_path = (1 - lambda_) * q0 + lambda_ * qd  # (2 x n_points)
    init_path = init_path.T  # (n_points x 2)
    path = init_path.copy()
    delta = 3.0 * np.linalg.norm(path[:, 1] - path[:, 0]) ** 2
    path_hist = [init_path.copy()]
    kind = None

    t0_opt = time.perf_counter()
    path_opt, path_hist, info = deform_path_ipopt(
        init_path,
        obstacles,
        method=method,
        verbose=False,
        max_iter=opt_max_iters,
        kind=kind,
        h=h,
        r=r,
        alpha=alpha,
        zeta=zeta,
        min_path=min_path,
        delta=delta,
    )
    t_opt = time.perf_counter() - t0_opt
    t_total = time.perf_counter() - t0_total

    collision_free = check_free_path(path_opt, obstacles, r=r, h=h, margin=-1e-3)
    dist_stats = path_distance_stats(
        path_opt,
        obstacles,
        r=r,
        h=h,
        alpha=alpha,
        test="" if kind is None else kind,
    )

    record = {
        "run_id": i,
        "seed": seed,
        "num_obstacles": len(obstacles),
        "num_path_points": n_points,
        "ipopt_info": info,
        # "solver_status": info.get("status", None),
        # "solver_converged": info.get("status", 0) == 0,
        # "num_iterations": info.get("iter_count", None),
        # "max_iter_reached": info.get("iter_count", 0) >= opt_max_iters,
        "total_time": t_total,
        "ipopt_time": t_opt,
        "success_collision_free": collision_free,
        **dist_stats,
    }

    return record


if __name__ == "__main__":
    n_workers = cpu_count()

    # Function factory to create a run_single_case with only first argument
    def run_single_case2d_ours(i):
        return run_single_case(i, method="ours")  # or "vanilla"

    def run_single_case3d_ours(i):
        return run_single_case3d(i, method="ours")  # or "vanilla"

    def run_single_case2d_esdf(i):
        return run_single_case(i, method="esdf")  # or "vanilla"

    def run_single_case3d_esdf(i):
        return run_single_case3d(i, method="esdf")  # or "vanilla"

    # Loop over each case and run multiple of a single case in parallel
    cases = {
        "2d_ours": run_single_case2d_ours,
        "3d_ours": run_single_case3d_ours,
        "2d_esdf": run_single_case2d_esdf,
        "3d_esdf": run_single_case3d_esdf,
    }

    for case_name, run_single in cases.items():
        time_now = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        print(f"[{time_now}] Starting case: {case_name} with {n_workers} workers.")

        with Pool(processes=n_workers) as pool:
            records = list(
                tqdm(
                    pool.imap_unordered(run_single, range(max_checks)),
                    total=max_checks,
                )
            )
        time_now = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        print(f"[{time_now}] Finished case: {case_name}. Starting to save results.")
        # Save final results
        df = pd.DataFrame(records)
        base_name = "planning_stats_smooth_distance"
        # Intersect will remain a global variable as it is not mandatory
        # for the experiments
        str_intersect = "_WITH_" if intersect else "_WITHOUT_"
        final_file_name = f"{base_name}_{case_name}_{str_intersect}_intersect.csv"
        df.to_csv(final_file_name, index=False)
        print("[EOR] Final results saved to planning_stats_smooth_distance.csv")
        # Print some summary statistics
        n_success = sum(r["success_collision_free"] for r in records)
        # Average min distance even if collision happened
        avg_min_dist = np.mean([r["min_dist"] for r in records])
        print(f"Total successful collision-free paths: {n_success}/{max_checks}. Average min dist: {avg_min_dist:.4f}")

        # records.append(record)
        # successes.append(collision_free)
        #
        # print(f"Check {i+1}/{max_checks}: Collision-free: {collision_free}")
        #
        # if (i + 1) % save_every == 0:
        #     print(f"Saving intermediate results at check {i+1}...")
        #     df = pd.DataFrame(records)
        #     df.to_csv("planning_stats_smooth_distance.csv", index=False)
        #     print("Saved.")


# %%
# test signedEuclideanDistance
# Square centered at 0
# A = np.array(
#     [
#         [1, 0],
#         [0, 1],
#         [-1, 0],
#         [0, -1],
#     ]
# )
# b = np.array([1, 1, 1, 1]).reshape(-1, 1)
# p = np.array([1.1, 1.0]).reshape(-1, 1)
# signedEuclideanDistance(p, A, b)
