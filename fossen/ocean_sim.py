import uaibot as ub
from uaibot.simobjects.pointlight import PointLight
from uaibot.simobjects.box import Box
from uaibot.graphics.meshmaterial import MeshMaterial
from uaibot.graphics.texture import Texture
from uaibot.utils import Utils
from uaibot.simulation import Simulation


def create_sim_underwater(objects=[], light_intensity=5.0, port=None):
    base = (
        f"http://localhost:{port}/"
        if port
        else "https://raw.githubusercontent.com/fbartelt/robotics-experiments/main/"
    )

    # 1. The Ocean Skybox
    ldr = base + "ldr/ocean/ocean_"
    ldr_list = [ldr + f + ".png" for f in ["px", "nx", "py", "ny", "pz", "nz"]]

    # 2. The Sand Seabed
    sand_color = Texture(
        url=base + "textures/sand/sand_color.jpg",
        wrap_s="RepeatWrapping",
        wrap_t="RepeatWrapping",
        repeat=[30, 30],
    )
    sand_normal = Texture(
        url=base + "textures/sand/sand_normal.jpg",
        wrap_s="RepeatWrapping",
        wrap_t="RepeatWrapping",
        repeat=[30, 30],
    )

    seabed_mat = MeshMaterial(
        texture_map=sand_color,
        normal_map=sand_normal,
        normal_scale=[0.5, 0.5],
        roughness=1.0,
        metalness=0.0,
    )
    # Put the seabed down at z = -3
    sand_height = 0.01
    seabed = Box(
        name="seabed",
        width=100,
        depth=100,
        height=sand_height,
        htm=Utils.trn([0, 0, -8]),
        mesh_material=seabed_mat,
    )

    # 3. The Water Surface
    water_normal = Texture(
        url=base + "textures/water/water_normal.jpg",
        wrap_s="RepeatWrapping",
        wrap_t="RepeatWrapping",
        repeat=[8, 8],
    )

    water_mat = MeshMaterial(
        color="#2060A0",  # Deep ocean blue
        transmission=0.9,  # See-through (glass-like)
        roughness=0.05,  # Very smooth, like water
        ior=1.33,  # Index of refraction for water
        metalness=0.0,
        opacity=1.0,  # MUST be 1.0 if transmission > 0
        side="DoubleSide",  # So it renders from below
        normal_map=water_normal,
        normal_scale=[0.3, 0.3],
    )
    water_height = 0.02
    water = Box(
        name="water_surface",
        width=100,
        depth=100,
        height=water_height,
        # height=0.02,
        htm=Utils.trn([0, 0, -8 + water_height/2 + sand_height/2]),
        mesh_material=water_mat,
    )

    # 4. Underwater Lighting (Blue-shifted)
    lights = [
        PointLight(
            name=f"light{i}",
            color="#3FA8D8",
            # color="#FFFFFF",
            intensity=light_intensity,
            htm=Utils.trn(p),
        )
        for i, p in enumerate(
            [[-4, -4, -1], [-4, 4, -1], [4, -4, -1], [4, 4, -1], [0, 0, 0]]
        )
    ]

    # 5. Create Simulation
    sim = Simulation(
        objects,
        ambient_light_intensity=light_intensity,
        # ldr_urls=ldr_list,
        background_color="#0A2A4A",  # Dark blue fallback
        show_grid=False,
        width=1920,
        height=1080,
        # local_host_port=port,
    )

    sim.add(water)
    sim.add(seabed)
    for l in lights:
        sim.add(l)

    return sim


# test = create_sim_underwater(port=None, light_intensity=5.0)
# test.run_in_browser()
