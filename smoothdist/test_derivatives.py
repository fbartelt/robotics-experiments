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
from distances import signed_dist2convex, id_phi, phi, smooth_min, signed_dist2nonconvex, holder_mean
from polyhedron import add_polyhedron, create_isosurfaces

# %%
"""Holder mean test """
## OK ##
rng = np.random.default_rng(42)
eps = 1e-5

# for i in range(10):
#     vals = rng.uniform(0, 10, size=(10,))
#     value, derivative = holder_mean(vals, r=0.1, compute_gradient=True)
#     # Numerical derivative
#     num_derivative = np.zeros_like(vals)
#     for j in range(len(vals)):
#         vals_plus = vals.copy()
#         vals_minus = vals.copy()
#         vals_plus[j] += eps
#         vals_minus[j] -= eps
#         value_plus = holder_mean(vals_plus, r=0.1, compute_gradient=False)
#         value_minus = holder_mean(vals_minus, r=0.1, compute_gradient=False)
#         num_derivative[j] = (value_plus - value_minus) / (2 * eps)
#     error = np.linalg.norm(derivative - num_derivative)
#     if error > 1e-3:
#         print(f"Holder mean gradient check failed at iteration {i}: error = {error}")
#         print("Analytical derivative:", derivative)
#         print("Numerical derivative:", num_derivative)
#         break
#
""" Smoooth min test """
for i in range(10):
    vals = rng.uniform(-10, -1, size=(2,))
    vals = np.array([-2, -3.0])
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
        value_plus = smooth_min(vals_plus[0], vals_plus[1], r=0.1, compute_gradient=False)
        value_minus = smooth_min(vals_minus[0], vals_minus[1], r=0.1, compute_gradient=False)
        num_derivative[j] = (value_plus - value_minus) / (2 * eps)
    error = np.linalg.norm(derivative - num_derivative)
    if error > 1e-3:
        print(f"vals:{vals}")
        print(f"value: {value}")
        print(f"Smooth min gradient check failed at iteration {i}: error = {error}")
        print("Analytical derivative:", derivative)
        print("Numerical derivative:", num_derivative)
        break

print("end")

# %%
"""Gradient test"""
n_points = 100
max_iters = 100
h = 0.01
r = 0.1
eta = 10.0
# bounding_box = (-6, -6, 6, 6)
bounding_box = (-1, -1, 5, 5)

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

points = np.linspace(bounding_box[0], bounding_box[2], n_points)
X, Y = np.meshgrid(points, points)
test = "in"
eps = 1e-3

vertices = polygon.vertices
print(vertices)

for i in range(X.shape[0]):
    for j in range(X.shape[1]):
        p = np.array([X[i, j], Y[i, j]]).reshape(-1, 1)
        p = vertices[np.random.randint(0, vertices.shape[0]), :].reshape(-1, 1)
        dist, grad = signed_dist2convex(
            phi, p, polygon.A, polygon.b, r=r, h=h, test=test, compute_gradient=True
        )

        # Numerical gradient:
        p_x_plus = p + np.array([[eps], [0]])
        p_x_minus = p - np.array([[eps], [0]])
        p_y_plus = p + np.array([[0], [eps]])
        p_y_minus = p - np.array([[0], [eps]])
        dist_x_plus = signed_dist2convex(
            phi,
            p_x_plus,
            polygon.A,
            polygon.b,
            r=r,
            h=h,
            test=test,
            compute_gradient=False,
        )
        dist_x_minus = signed_dist2convex(
            phi,
            p_x_minus,
            polygon.A,
            polygon.b,
            r=r,
            h=h,
            test=test,
            compute_gradient=False,
        )
        dist_y_plus = signed_dist2convex(
            phi,
            p_y_plus,
            polygon.A,
            polygon.b,
            r=r,
            h=h,
            test=test,
            compute_gradient=False,
        )
        dist_y_minus = signed_dist2convex(
            phi,
            p_y_minus,
            polygon.A,
            polygon.b,
            r=r,
            h=h,
            test=test,
            compute_gradient=False,
        )
        num_grad_x = (dist_x_plus - dist_x_minus) / (2 * eps)
        num_grad_y = (dist_y_plus - dist_y_minus) / (2 * eps)
        num_grad = np.array([num_grad_x, num_grad_y]).reshape(-1, 1)
        error = np.linalg.norm(grad - num_grad)
        if error > 1e-3:
            print(f"Gradient check failed at point {p.flatten()}: error = {error}")
            print(f"Distance: {dist}")
            print("Analytical grad:", grad.ravel())
            print("Numerical grad:", num_grad.ravel())
            break



