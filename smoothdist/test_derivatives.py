# %%
import sys, os
import numpy as np
import plotly.graph_objects as go
import plotly.colors as pc

from polygon import (
    Polytope,
    NonConvexPolygon,
    add_polygon,
    get_polytope_constraints,
    create_level_sets,
)
from distances import (
    signed_dist2convex,
    id_phi,
    phi,
    smooth_min,
    signed_dist2nonconvex,
    holder_mean,
    smooth_max,
)
from polyhedron import add_polyhedron, create_isosurfaces

# %%
"""Holder mean test """
## OK ##
rng = np.random.default_rng(42)
eps = 1e-5

for i in range(10):
    vals = rng.uniform(0, 10, size=(10,))
    value, derivative = holder_mean(vals, r=0.1, compute_gradient=True)
    # Numerical derivative
    num_derivative = np.zeros_like(vals)
    for j in range(len(vals)):
        vals_plus = vals.copy()
        vals_minus = vals.copy()
        vals_plus[j] += eps
        vals_minus[j] -= eps
        value_plus = holder_mean(vals_plus, r=0.1, compute_gradient=False)
        value_minus = holder_mean(vals_minus, r=0.1, compute_gradient=False)
        num_derivative[j] = (value_plus - value_minus) / (2 * eps)
    error = np.linalg.norm(derivative - num_derivative)
    if error > 1e-3:
        print(f"!!!!Holder mean gradient check failed at iteration {i}: error = {error}")
        print("Analytical derivative:", derivative)
        print("Numerical derivative:", num_derivative)
        break
print("############# holder mean end")

""" Smoooth min test """
for i in range(10):
    vals = rng.uniform(-10., 10, size=(4,))
    # vals = np.array([2, 3.0])
    # vals = rng.uniform(0, 10, size=(2,))
    # vals = rng.uniform(-10, 10, size=(2,))
    # vals = np.array([rng.uniform(-10, 0), rng.uniform(0, 10)])
    # value, derivative = smooth_min(vals[0], vals[1], r=0.1, compute_gradient=True)
    value, derivative = smooth_min(x=vals, r=0.1, compute_gradient=True)
    # Numerical derivative
    num_derivative = np.zeros_like(vals)
    for j in range(len(vals)):
        vals_plus = vals.copy()
        vals_minus = vals.copy()
        vals_plus[j] += eps
        vals_minus[j] -= eps
        value_plus = smooth_min(
            vals_plus[0], vals_plus[1], r=0.1, compute_gradient=False
        )
        value_minus = smooth_min(
            vals_minus[0], vals_minus[1], r=0.1, compute_gradient=False
        )
        num_derivative[j] = (value_plus - value_minus) / (2 * eps)
    error = np.linalg.norm(derivative - num_derivative)
    if error > 1e-3:
        print(f"vals:{vals}")
        print(f"value: {value}")
        print(f"!!!!Smooth min gradient check failed at iteration {i}: error = {error}")
        print("Analytical derivative:", derivative)
        print("Numerical derivative:", num_derivative)
        break

print("############# smooth min end")

""" Smoooth max test """
for i in range(10):
    vals = rng.uniform(-10., 10, size=(4,))
    # vals = np.array([2, 3.0])
    # vals = rng.uniform(0, 10, size=(2,))
    # vals = rng.uniform(-10, 10, size=(2,))
    # vals = np.array([rng.uniform(-10, 0), rng.uniform(0, 10)])
    # value, derivative = smooth_min(vals[0], vals[1], r=0.1, compute_gradient=True)
    value, derivative = smooth_max(x=vals, r=0.1, compute_gradient=True)
    # Numerical derivative
    num_derivative = np.zeros_like(vals)
    for j in range(len(vals)):
        vals_plus = vals.copy()
        vals_minus = vals.copy()
        vals_plus[j] += eps
        vals_minus[j] -= eps
        value_plus = smooth_max(
            vals_plus[0], vals_plus[1], r=0.1, compute_gradient=False
        )
        value_minus = smooth_max(
            vals_minus[0], vals_minus[1], r=0.1, compute_gradient=False
        )
        num_derivative[j] = (value_plus - value_minus) / (2 * eps)
    error = np.linalg.norm(derivative - num_derivative)
    if error > 1e-3:
        print(f"vals:{vals}")
        print(f"value: {value}")
        print(f"!!!!Smooth max gradient check failed at iteration {i}: error = {error}")
        print("Analytical derivative:", derivative)
        print("Numerical derivative:", num_derivative)
        break

print("end")


# %%
"""Gradient test"""
n_points = 3
max_iters = 100
h = 0.01
r = 0.1
eta = 10.0
# bounding_box = (-6, -6, 6, 6)
bounding_box = (-1.0, -1, 5, 5)

seed = 42  # 100 is cool

polygon = Polytope.random(
    num_vertices=7, radius_lim=(1e-1, 1.0), bbox=bounding_box, seed=seed
)
polygon2 = Polytope.random(
    num_vertices=5, radius_lim=(1e-1, 1.0), bbox=bounding_box, seed=seed + 1
)
polygon3 = Polytope.random(
    num_vertices=6, radius_lim=(1e-1, 1.0), bbox=bounding_box, seed=seed + 2
)

polygons = [
    polygon,
]

vertices = polygon.vertices
# bounding_box = (vertices.min(), vertices[:, 1].min(),
#                 vertices.max(), vertices[:, 1].max())
bounding_box = (-100, -100, -20, -20)
xs = np.linspace(bounding_box[0], bounding_box[2], n_points)
ys = np.linspace(bounding_box[1], bounding_box[3], n_points)

test = "out"
test = None
eps = 1e-5

print(vertices)

for x in xs:
    for y in ys:
        point = np.array([x, y]).reshape(-1, 1)
        print(point, point.shape)
        val, grad = signed_dist2convex(
            phi, point, polygon.A, polygon.b, r=r, test=test, compute_gradient=True
        )
        # Numerical gradient
        num_grad = np.zeros_like(point).ravel()
        delta_x = np.array([eps, 0]).reshape(-1, 1)
        delta_y = np.array([0, eps]).reshape(-1, 1)
        x_plus = point + delta_x
        x_minus = point - delta_x
        y_plus = point + delta_y
        y_minus = point - delta_y
        val_x_plus = signed_dist2convex(
            phi, x_plus, polygon.A, polygon.b, r=r, test=test, compute_gradient=False
        )
        val_x_minus = signed_dist2convex(
            phi, x_minus, polygon.A, polygon.b, r=r, test=test, compute_gradient=False
        )
        val_y_plus = signed_dist2convex(
            phi, y_plus, polygon.A, polygon.b, r=r, test=test, compute_gradient=False
        )
        val_y_minus = signed_dist2convex(
            phi, y_minus, polygon.A, polygon.b, r=r, test=test, compute_gradient=False
        )
        num_grad[0] = (val_x_plus - val_x_minus) / (2 * eps)
        num_grad[1] = (val_y_plus - val_y_minus) / (2 * eps)
        error = np.linalg.norm(grad - num_grad)
        if error > 1e-3:
            print(f"Point: {point}")
            print(f"val: {val}")
            print(f"Gradient check failed at point {point}: error = {error}")
            print("Analytical gradient:", grad)
            print("Numerical gradient:", num_grad)
            break

