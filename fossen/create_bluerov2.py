import numpy as np
import uaibot as ub


def make_rov(htm=np.eye(4)):
    model = ub.Model3D(
        url="https://raw.githubusercontent.com/fbartelt/robotics-experiments/main/models/bluerov/bluerov2_heavy_textured.glb",
        scale=1,
        mesh_material=ub.GLBMeshMaterial(),
    )
    return ub.RigidObject(list_model_3d=[model], htm=htm)
