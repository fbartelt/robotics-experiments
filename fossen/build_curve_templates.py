import os
import numpy as np
from pathlib import Path
from fossen.curves import build_curve

CURVE_X0, CURVE_Y0, CURVE_Z0 = -2.0 * 0, -2.0 * 0, -4.0
CURVE_POINTS = 15000 * 2

CIRCLE_RADIUS = 1.5 * 8
CURVE_N_ROLLS = 1.0

# Ellipse
ELLIPSE_A = 3.0 * 1
ELLIPSE_B = 2.0 * 1

# Lissajous
LISSAJOUS_A = 4.0
LISSAJOUS_B = 3.5
LISSAJOUS_C = 0.5

# Trefoil
TREFOIL_R = 1.0 * 2
TREFOIL_Z_AMP = 0.5 * 1

# Cylinder surface
CYLINDER_R = 2.0
CYLINDER_Z_AMP = 1.0
CYLINDER_N_OSCILLATIONS = 3


data_folder = "./fossen/data"
directory = Path(data_folder)
directory.mkdir(parents=True, exist_ok=True)

for CURVE_TYPE in ["ellipse", "trefoil", "vertical_coil"]:

    kwargs = {
        "n_points": CURVE_POINTS,
        "z0": CURVE_Z0,
        "center_xy": (CURVE_X0, CURVE_Y0),
    }

    if CURVE_TYPE == "circle":
        kwargs["radius"] = CIRCLE_RADIUS
        kwargs["n_rolls"] = CURVE_N_ROLLS
    elif CURVE_TYPE == "ellipse":
        kwargs["a"] = ELLIPSE_A
        kwargs["b"] = ELLIPSE_B
    elif CURVE_TYPE == "lissajous_3d":
        kwargs["A"] = LISSAJOUS_A
        kwargs["B"] = LISSAJOUS_B
        kwargs["C"] = LISSAJOUS_C
    elif CURVE_TYPE == "trefoil":
        kwargs["R"] = TREFOIL_R
        kwargs["z_amp"] = TREFOIL_Z_AMP
    elif CURVE_TYPE == "vertical_coil":
        kwargs["R"] = CYLINDER_R
        kwargs["z_amp"] = CYLINDER_Z_AMP
        kwargs["N_oscillations"] = CYLINDER_N_OSCILLATIONS

    curve, curve_derivative = build_curve(CURVE_TYPE, **kwargs)
    curve_file_name = f"{CURVE_TYPE}.npy"
    dcurve_file_name = f"{CURVE_TYPE}_derivative.npy"
    curve_path = os.path.join(data_folder, curve_file_name)
    dcurve_path = os.path.join(data_folder, dcurve_file_name)
    np.save(curve_path, curve)
    print(f"Saved resampled curve at '{curve_path}'")
    np.save(dcurve_path, curve_derivative)
    print(f"Saved resampled curve at '{dcurve_path}'")
