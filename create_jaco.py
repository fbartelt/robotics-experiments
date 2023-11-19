# %%
import sys
sys.path.insert(1, '/home/fbartelt/Documents/UFMG/TCC/Sim/uaibot/uaibot')
from robot.links import Link
import robot as rb
from utils import Utils
from graphics.meshmaterial import MeshMaterial
from graphics.model3d import Model3D
from simobjects.rigidobject import RigidObject
from simobjects.pointlight import PointLight
from simulation import Simulation
from simobjects.frame import Frame
import numpy as np


def _create_jaco(name='jaco_robot', color='#3e3f42', opacity=1):
    """
    model: https://www.kinovarobotics.com/resources?r=79302&s
    docs: https://www.kinovarobotics.com/resources?r=339
    """
    pi = np.pi
    d1 = 0.2755
    d2 = 0.41
    d3 = 0.2073
    d4 = 0.0741
    d5 = 0.0741
    d6 = 0.16
    e2 = 0.0098
    aa = 30*pi/180
    sa = np.sin(aa)
    s2a = np.sin(2*aa)
    d4b = d3 + (sa/s2a) * d4
    d5b = (sa/s2a)*d4 + (sa/s2a)*d5
    d6b = (sa/s2a)*d5 + d6

    jaco_DH_theta = np.array([0,     0,    0,    0,    0,    0])
    jaco_DH_d = np.array([d1,    0,  -e2, -d4b, -d5b, -d6b])
    jaco_DH_a = np.array([0,    d2,    0,    0,    0,    0])
    jaco_DH_alpha = np.array([pi/2, pi, pi/2, 2*aa, 2*aa,    pi])

    jaco_DH_type = np.array([0, 0, 0, 0, 0, 0])
    link_info = np.array(
        [jaco_DH_theta, jaco_DH_d, jaco_DH_alpha, jaco_DH_a, jaco_DH_type])  # jaco_dummy

    scale = 1
    n = link_info.shape[1]
    base_3d_obj = []
    mesh = MeshMaterial(metalness=0.8, clearcoat=0.9, roughness=0.3,
                        normal_scale=[0.5, 0.5], color="#3e3f42",
                        opacity=opacity, side="DoubleSide")
    mesh_ring = MeshMaterial(metalness=0, roughness=1, clearcoat=0, clearcoat_roughness=0.03, ior=1.45,
                             normal_scale=[0.5, 0.5], color="#919090",
                             opacity=opacity, side="DoubleSide")
    mesh_nail = MeshMaterial(metalness=0, clearcoat=0, roughness=1,
                             normal_scale=[0.5, 0.5], color="#1d1d1f",
                             opacity=opacity, side="DoubleSide")
    # original model is rotated (Robot fron = plane X x Y)
    Q00 = Utils.rotx(0)
    Q001 = Utils.trn([0, 0, 1.5675e-1])
    Q01 = Q001 * (Utils.rotz(link_info[0, 0]) * Utils.trn([0, 0, link_info[1, 0]]) * Utils.rotx(link_info[2, 0]) * Utils.trn(
        [link_info[3, 0], 0, 0]))
    Q02 = Q01 * (Utils.rotz(link_info[0, 1] + pi/2) * Utils.trn([0, 0, link_info[1, 1]]) * Utils.rotx(link_info[2, 1]) * Utils.trn(
        [link_info[3, 1], 0, 0]))
    Q03 = Q02 * (Utils.rotz(link_info[0, 2] - pi/2) * Utils.trn([0, 0, link_info[1, 2]]) * Utils.rotx(link_info[2, 2]) * Utils.trn(
        [link_info[3, 2], 0, 0]))
    Q04 = Q03 * (Utils.rotz(link_info[0, 3] + 0) * Utils.trn([0, 0, link_info[1, 3]]) * Utils.rotx(link_info[2, 3]) * Utils.trn(
        [link_info[3, 3], 0, 0]))
    Q05 = Q04 * (Utils.rotz(link_info[0, 4] - pi) * Utils.trn([0, 0, link_info[1, 4]]) * Utils.rotx(link_info[2, 4]) * Utils.trn(
        [link_info[3, 4], 0, 0]))
    # Q06 = Q05 * (Utils.rotz(link_info[0, 5] + 0) * Utils.trn([0, 0, link_info[1, 5]]) * Utils.rotx(link_info[2, 5]) * Utils.trn(
    #    [link_info[3, 5], 0, 0]))
    q0 = [pi, pi, pi, pi, 0, pi/2]
    Q00 = Utils.trn([0, 0, 0])
    Q001 = Utils.trn([0, 0, 0])
    Q01 = (Utils.rotz(link_info[0, 0] - pi/2 - q0[0]) * Utils.trn([0, 0, link_info[1, 0]])
           * Utils.rotx(link_info[2, 0]) * Utils.trn([link_info[3, 0], 0, 0]))
    Q02 = Q01 @ (Utils.rotz(link_info[0, 1] - pi/2 - np.deg2rad(25.002) + q0[1]) * Utils.trn(
        [0, 0, link_info[1, 1]]) * Utils.rotx(link_info[2, 1]) * Utils.trn([link_info[3, 1], 0, 0]))
    Q03 = Q02 @ (Utils.rotz(link_info[0, 2] - pi/2 + np.deg2rad(25.002) + q0[2]) * Utils.trn(
        [0, 0, link_info[1, 2]]) * Utils.rotx(link_info[2, 2]) * Utils.trn([link_info[3, 2], 0, 0]))
    Q04 = Q03 @ (Utils.rotz(link_info[0, 3] + pi/2 + q0[3]) * Utils.trn(
        [0, 0, link_info[1, 3]]) * Utils.rotx(link_info[2, 3]) * Utils.trn([link_info[3, 3], 0, 0]))
    Q05 = Q04 @ (Utils.rotz(link_info[0, 4] + pi + q0[4]) * Utils.trn(
        [0, 0, link_info[1, 4]]) * Utils.rotx(link_info[2, 4]) * Utils.trn([link_info[3, 4], 0, 0]))
    Q06 = Q05 @ Utils.rotz(link_info[0, 5] + pi/2 + q0[5]) * Utils.trn(
        [0, 0, link_info[1, 5]]) * Utils.rotx(link_info[2, 5]) * Utils.trn([link_info[3, 5], 0, 0])

    link0_mth = Utils.inv_htm(Q00)
    base_3d_obj = [
        Model3D(url='https://raw.githubusercontent.com/fbartelt/robotics-experiments/main/models/jaco/base.obj',
                scale=scale, htm=link0_mth, mesh_material=mesh),
        Model3D(url='https://raw.githubusercontent.com/fbartelt/robotics-experiments/main/models/jaco/base_ring.obj',
                scale=scale, htm=link0_mth, mesh_material=mesh_ring)]
    link_3d_obj = []
    # Shoulder
    link1_mth = Utils.inv_htm(Q01)
    link_3d_obj.append([
        Model3D(url='https://raw.githubusercontent.com/fbartelt/robotics-experiments/main/models/jaco/shoulder.obj',
                scale=scale, htm=link1_mth, mesh_material=mesh),
        Model3D(url='https://raw.githubusercontent.com/fbartelt/robotics-experiments/main/models/jaco/shoulder_ring.obj',
                scale=scale, htm=link1_mth, mesh_material=mesh_ring),
    ])

    # Upper arm + elbow
    link2_mth = Utils.inv_htm(Q02)
    link_3d_obj.append([
        Model3D(url='https://raw.githubusercontent.com/fbartelt/robotics-experiments/main/models/jaco/upperarm.obj',
                scale=scale, htm=link2_mth, mesh_material=mesh),
        Model3D(url='https://raw.githubusercontent.com/fbartelt/robotics-experiments/main/models/jaco/upperarm_ring.obj',
                scale=scale, htm=link2_mth, mesh_material=mesh_ring),
    ])

    # Forearm
    link3_mth = Utils.inv_htm(Q03)
    link_3d_obj.append([
        Model3D(url='https://raw.githubusercontent.com/fbartelt/robotics-experiments/main/models/jaco/forearm.obj',
                scale=scale, htm=link3_mth, mesh_material=mesh),
        Model3D(url='https://raw.githubusercontent.com/fbartelt/robotics-experiments/main/models/jaco/forearm_ring.obj',
                scale=scale, htm=link3_mth, mesh_material=mesh_ring),
    ])

    link4_mth = Utils.inv_htm(Q04)
    link_3d_obj.append([
        Model3D(url='https://raw.githubusercontent.com/fbartelt/robotics-experiments/main/models/jaco/wrist1.obj',
                scale=scale, htm=link4_mth, mesh_material=mesh),
        Model3D(url='https://raw.githubusercontent.com/fbartelt/robotics-experiments/main/models/jaco/wrist1_ring.obj',
                scale=scale, htm=link4_mth, mesh_material=mesh_ring),
    ])

    link5_mth = Utils.inv_htm(Q05)
    link_3d_obj.append([
        Model3D(url='https://raw.githubusercontent.com/fbartelt/robotics-experiments/main/models/jaco/wrist2.obj',
                scale=scale, htm=link5_mth, mesh_material=mesh),
        Model3D(url='https://raw.githubusercontent.com/fbartelt/robotics-experiments/main/models/jaco/wrist2_ring.obj',
                scale=scale, htm=link5_mth, mesh_material=mesh_ring),
    ])

    link6_mth = Utils.inv_htm(Q06)
    link_3d_obj.append([
        Model3D(url='https://raw.githubusercontent.com/fbartelt/robotics-experiments/main/models/jaco/gripper.obj',
                scale=scale, htm=link6_mth, mesh_material=mesh),
        Model3D(url='https://raw.githubusercontent.com/fbartelt/robotics-experiments/main/models/jaco/handpalm.obj',
                scale=scale, htm=link6_mth, mesh_material=mesh_ring),
        Model3D(url='https://raw.githubusercontent.com/fbartelt/robotics-experiments/main/models/jaco/finger1_mounting.obj',
                scale=scale, htm=link6_mth, mesh_material=mesh_ring),
        Model3D(url='https://raw.githubusercontent.com/fbartelt/robotics-experiments/main/models/jaco/finger1_proximal.obj',
                scale=scale, htm=link6_mth, mesh_material=mesh_ring),
        Model3D(url='https://raw.githubusercontent.com/fbartelt/robotics-experiments/main/models/jaco/finger1_distal.obj',
                scale=scale, htm=link6_mth, mesh_material=mesh_ring),
        Model3D(url='https://raw.githubusercontent.com/fbartelt/robotics-experiments/main/models/jaco/finger1_nail.obj',
                scale=scale, htm=link6_mth, mesh_material=mesh_nail),
        Model3D(url='https://raw.githubusercontent.com/fbartelt/robotics-experiments/main/models/jaco/finger2_mounting.obj',
                scale=scale, htm=link6_mth, mesh_material=mesh_ring),
        Model3D(url='https://raw.githubusercontent.com/fbartelt/robotics-experiments/main/models/jaco/finger2_proximal.obj',
                scale=scale, htm=link6_mth, mesh_material=mesh_ring),
        Model3D(url='https://raw.githubusercontent.com/fbartelt/robotics-experiments/main/models/jaco/finger2_distal.obj',
                scale=scale, htm=link6_mth, mesh_material=mesh_ring),
        Model3D(url='https://raw.githubusercontent.com/fbartelt/robotics-experiments/main/models/jaco/finger2_nail.obj',
                scale=scale, htm=link6_mth, mesh_material=mesh_nail),
        Model3D(url='https://raw.githubusercontent.com/fbartelt/robotics-experiments/main/models/jaco/thumb_mounting.obj',
                scale=scale, htm=link6_mth, mesh_material=mesh_ring),
        Model3D(url='https://raw.githubusercontent.com/fbartelt/robotics-experiments/main/models/jaco/thumb_proximal.obj',
                scale=scale, htm=link6_mth, mesh_material=mesh_ring),
        Model3D(url='https://raw.githubusercontent.com/fbartelt/robotics-experiments/main/models/jaco/thumb_distal.obj',
                scale=scale, htm=link6_mth, mesh_material=mesh_ring),
        Model3D(url='https://raw.githubusercontent.com/fbartelt/robotics-experiments/main/models/jaco/thumb_nail.obj',
                scale=scale, htm=link6_mth, mesh_material=mesh_nail),
    ])

    # com_coordinates = [[0.534615, 0, 0.15], [1.5353, 0, 0.15]]
    com_coordinates = [np.eye(4)[:-1, :] @ (Utils.inv_htm(Q01) @ np.array([-3.11506292e-3,  1.62075358e-5,  2.66810879e-1, 1]).reshape(-1, 1)),
                       np.eye(4)[:-1, :] @ (Utils.inv_htm(Q02) @ np.array(
                           [-0.00592762,  0.14709695,  0.5909634, 1]).reshape(-1, 1)),
                       np.eye(4)[:-1, :] @ (Utils.inv_htm(Q03) @ np.array(
                           [0.01162087, 0.04102689, 0.53732661, 1]).reshape(-1, 1)),
                       np.eye(4)[:-1, :] @ (Utils.inv_htm(Q04) @ np.array(
                           [0.00971901, -0.01057426,  0.44918022, 1]).reshape(-1, 1)),
                       np.eye(4)[:-1, :] @ (Utils.inv_htm(Q05) @ np.array(
                           [0.0097224, -0.03312561,  0.3785306, 1]).reshape(-1, 1)),
                       np.eye(4)[:-1, :] @ (Utils.inv_htm(Q06) @ np.array([0.0033009, -0.09643814,  0.32164165, 1]).reshape(-1, 1))]
    list_mass = [0.740185285501933, 0.8489361778912677, 0.48326882919212083,
                 0.43217459270218916, 0.43217590992539645, 0.6208313337899377]
    list_inertia_mat = []

    # Icm + Steiner theorem (Inertia mat is in respect to DH frame)
    list_inertia_mat.append(np.array([[1.066e-03,  0.000e+00,  3.800e-05], [0.000e+00,  1.038e-03, -0.000e+00], [
                            3.800e-05, -0.000e+00,  5.400e-04]]) - list_mass[0] * Utils.S(com_coordinates[0].T) @ Utils.S(com_coordinates[0]))  # in world frame
    list_inertia_mat.append(np.array([[0.014208, -0.000403, -0.000865], [-0.000403,  0.011765, -0.005182],
                            [-0.000865, -0.005182,  0.003068]]) - list_mass[1] * Utils.S(com_coordinates[1].T) @ Utils.S(com_coordinates[1]))
    # list_inertia_mat.append(np.array([[ 2.296e-03, -9.600e-05, -7.900e-05], [-9.600e-05,  8.140e-04, -1.189e-03], [-7.900e-05, -1.189e-03,  1.686e-03]])) # NOT POSITIVE DEF.
    list_inertia_mat.append(np.array([[0.002, -0., -0.], [-0., 0.001, -0.001], [-0., -
                            0.001,  0.002]]) - list_mass[2] * Utils.S(com_coordinates[2].T) @ Utils.S(com_coordinates[2]))
    list_inertia_mat.append(np.array([[3.03e-04,  0.00e+00, -0.00e+00], [0.00e+00,  3.05e-04, -3.00e-06],
                            [-0.00e+00, -3.00e-06,  1.97e-04]]) - list_mass[3] * Utils.S(com_coordinates[3].T) @ Utils.S(com_coordinates[3]))
    list_inertia_mat.append(np.array([[3.03e-04, -0.00e+00, -0.00e+00], [-0.00e+00,  2.64e-04, -5.20e-05],
                            [-0.00e+00, -5.20e-05,  2.39e-04]]) - list_mass[4] * Utils.S(com_coordinates[4].T) @ Utils.S(com_coordinates[4]))
    list_inertia_mat.append(np.array([[1.731e-03, -5.300e-05, -6.400e-05], [-5.300e-05,  1.822e-03, -3.500e-05],
                            [-6.400e-05, -3.500e-05,  1.079e-03]]) - list_mass[5] * Utils.S(com_coordinates[5].T) @ Utils.S(com_coordinates[5]))

    links = []
    for i in range(n):
        links.append(Link(i, theta=link_info[0, i], d=link_info[1, i], alpha=link_info[2, i], a=link_info[3, i], joint_type=link_info[4, i],
                          list_model_3d=link_3d_obj[i], com_coordinates=com_coordinates[i], mass=list_mass[i], inertia_matrix=list_inertia_mat[i]))

    # q0 = [0, pi/2, -pi/2, 0, -pi]#, 0]
    htm_n_eef = Utils.rotz(-pi) * Utils.rotx(0.3056*pi) * \
        Utils.rotx(0.3056*pi) * Utils.trn([0, 0, 0.052])
    htm_n_eef = Utils.trn([0, 0, 0])
    # Utils.trn([-3.2712e-05, -1.7324e-05, 1.5675e-01])
    htm_base_0 = Utils.trn([0, 0, 0*1.5675e-1])

    # Create joint limits
    joint_limits = np.matrix([[-3*np.pi, 3*np.pi], [-np.deg2rad(47), np.deg2rad(266)],
                              [-np.deg2rad(19), np.deg2rad(322)
                               ], [-3*np.pi, 3*np.pi],
                              [-3*np.pi, 3*np.pi], [-3*np.pi, 3*np.pi]])  # , [-np.pi, np.pi]])

    return links, base_3d_obj, htm_base_0, htm_n_eef, q0, joint_limits


def create_jaco2(name='jaco_robot', opacity=1):
    links, base_3d_obj, htm_base_0, htm_n_eef, q0, joint_limits = _create_jaco(
        name=name, opacity=opacity)
    jaco = rb.Robot(name=name, links=links, list_base_3d_obj=base_3d_obj, htm=np.identity(4),
                    htm_base_0=htm_base_0, htm_n_eef=htm_n_eef, q0=q0, eef_frame_visible=True, joint_limits=joint_limits)
    return jaco


# %%
if __name__ == '__main__':
    internet_down = 1
    while internet_down:
        try:
            robot = create_jaco2()
            internet_down = 0
        except:
            internet_down = 1
    light1 = PointLight(name="light1", color="white",
                        intensity=2.5, htm=Utils.trn([-1, -1, 1.5]))
    light2 = PointLight(name="light2", color="white",
                        intensity=2.5, htm=Utils.trn([-1, 1, 1.5]))
    light3 = PointLight(name="light3", color="white",
                        intensity=2.5, htm=Utils.trn([1, -1, 1.5]))
    light4 = PointLight(name="light4", color="white",
                        intensity=2.5, htm=Utils.trn([1, 1, 1.5]))
    sim = Simulation.create_sim_grid([robot, light1, light2, light3, light4])
    sim.set_parameters(width=1200, height=600, ambient_light_intensity=4)
    axis = 'dh'
    frames_ref = robot.fkm(axis=axis)
    frames = []
    for i, htm in enumerate(frames_ref):
        frames.append(Frame(name=f'{axis}_{i}', htm=htm, axis_color=[
                      'magenta', 'olive', 'cyan'], size=0.15))

    for frame in frames:
        sim.add(frame)
    sim.run()
# %%
