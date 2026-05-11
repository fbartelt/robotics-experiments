# %%
import smoothfunctions as sf
import numpy as np
import plotly.graph_objects as go
from smoothfunctions import ESDF2D_CGAL, ESDF_CGAL
from polygon import (
    Polytope,
    add_polygon,
    create_level_sets,
    NonConvexPolygon,
    generate_random_polyhedron,
    generate_random_polyhedron_set,
)
from polyhedron import add_polyhedron


def add_vector(fig, p, grad_, size=None, color="black", name="vector"):
    grad = grad_.ravel() / (np.linalg.norm(grad_) + 1e-8)
    if size == None:
        grad = grad * np.linalg.norm(p)
    else:
        grad = grad * size
    fig.add_trace(
        go.Scatter(
            x=[p[0], p[0] + grad[0]],
            y=[p[1], p[1] + grad[1]],
            mode="lines+markers",
            marker=dict(size=5, color=color),
            line=dict(color=color, width=2),
            name=name,
        )
    )
    # Add arrowhead as triangle
    fig.add_trace(
            go.Scatter(
                x=[p[0] + grad[0]],
                y=[p[1] + grad[1]],
                mode="markers",
                marker=dict(size=10, color=color, symbol="triangle-up"),
                showlegend=False,
            )
        )

def add_point(fig, p, color="black", name="point"):
    fig.add_trace(
        go.Scatter(
            x=[p[0]],
            y=[p[1]],
            mode="markers",
            marker=dict(size=8, color=color),
            name=name,
        )
    )
# Square
seed = 42
bbox = (-2, -2, 2, 2.0)
polygon = Polytope.random(seed=42, bbox=bbox)
A = polygon.A
b = polygon.b

# A = np.array([[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]])
# b = np.array([1.0, 1.0, 1.0, 1.0])
# polygon = Polytope(A, b)

fig = go.Figure()
add_polygon(fig, A, b, add_reference=False)

for i in range(10):
    # Generate outside points (probably)
    if i < 5:
        p = np.random.uniform(-2, 2, size=(2, ))
    else:
        delta = np.random.uniform(-1, 1, size=(2, )) * 0.1
        p = polygon.centroid + delta

    dist, grad, closest = ESDF_CGAL(p, A, b)
    if closest.shape != p.shape:
        closest = closest[:2]
    size = np.linalg.norm(p - closest)
    print(f"Distance: {dist}, Gradient: {grad.ravel()}, size: {size}")
    print(f"Point: {p}, Closest: {closest}")

    grad = -grad if dist > 0 else grad
    add_vector(fig, p, grad, size=size, color="red", name="gradient")
    add_point(fig, p, color="blue", name="query point")
    add_point(fig, closest, color="green", name="closest point")
fig.show()

