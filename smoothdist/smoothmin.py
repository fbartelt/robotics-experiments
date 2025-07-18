# %%
import numpy as np

def holder_mean(values, r=0.1):
    v = np.array(values)
    if np.any(v < 0):
        raise ValueError("All values must be non-negative")
    # Raise error if values is a list of lists or nested arrays
    if v.ndim > 1:
        raise ValueError("Input must be a 1D array or list")

    raised = list(map(lambda x: x**(-1/r), v))
    res = np.sum(raised)**(-r)
    return res

def smooth_min(x, y, r=0.1):
    if x >= 0 and y >= 0:
        return holder_mean([x, y], r)
    elif x < 0 and y < 0:
        return -1/holder_mean([-1/x, -1/y], r)
    else:
        return min(x, y)

def test_associativity():
    r = 0.1
    xs = np.arange(-100, 100, 1)
    ys, zs = xs.copy(), xs.copy()  # Ensure ys and zs are the same as xs for testing
    for x in xs:
        for y in ys:
            for z in zs:
                left = smooth_min(smooth_min(x, y, r), z, r)
                right = smooth_min(x, smooth_min(y, z, r), r)
                # print(f"Testing associativity for {x}, {y}, {z}: {left} vs {right}")
                assert np.isclose(left, right), f"Failed associativity for {x}, {y}, {z}"


test_associativity()
print("Associativity test passed.")
# print python path

import os, sys
print("Python path:", sys.executable)
# import manim
!which python
!which ipython
