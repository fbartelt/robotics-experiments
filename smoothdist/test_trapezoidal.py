import plotly.graph_objects as go
import numpy as np
from polygon import trapezoidal_decompose_to_halfspaces, Polytope, add_polygon
# --------------------------
# Example usage
# --------------------------
# Example non-convex polygon (clockwise):
poly_clockwise = [
    (0.0, 4.0),
    (2.0, 4.0),
    (2.0, 2.0),
    (4.0, 2.0),
    (4.0, 0.0),
    (0.0, 0.0),
]

fig = go.Figure()
cells = trapezoidal_decompose_to_halfspaces(poly_clockwise, axis='y')
print(f"Produced {len(cells)} cells (vertical slabs).")
for k, cell in enumerate(cells):
    print(f"\nCell {k}: slab x in {cell['slab']}")
    print("Vertices (clockwise):")
    for v in cell['poly']:
        print(f"  {v}")
        fig.add_trace(
            go.Scatter(
                x=[v[0]],
                y=[v[1]],
                mode='markers+text',
                text=[f"C{k}"],
                textposition="top center",
                marker=dict(size=8),
                name=f"Cell {k} Vertex"
            )
        )
    print("A matrix:")
    print(cell['A'])
    print("b vector:")
    print(cell['b'])
    S = Polytope(cell['A'], cell['b'])
    add_polygon(fig, S.A, S.b, add_reference=False)

fig.show()

