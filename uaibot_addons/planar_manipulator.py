# %%
import sys
sys.path.insert(1, '/home/fbartelt/Documents/UFMG/TCC/Sim/uaibot/uaibot')
import numpy as np
import robot as rb
from graphics.meshmaterial import MeshMaterial
from graphics.model3d import Model3D
from robot.links import Link
from utils import Utils

def _create_2dof(color='#3e3f42', opacity=1):

    DH_theta = np.array([0, 0,])
    DH_d = np.array([0, 0,])
    DH_a = np.array([1, 0,])
    DH_alpha = np.array([0.0, -np.pi/2*0, ])

    DH_type = np.array([0, 0])
    link_info = np.array([DH_theta, DH_d, DH_alpha, DH_a, DH_type])

    scale = 1
    n = link_info.shape[1]
    base_3d_obj = []
    mesh = MeshMaterial(metalness=0.5, clearcoat=0, roughness=0.5,
                        normal_scale=[0.5, 0.5], color=color,
                        opacity=opacity, side="DoubleSide")

    htm_base_0 = Utils.trn([0, 0, 0.15])

    Q00 = Utils.rotz(np.pi/2) * Utils.rotx(np.pi/2) * Utils.trn(
        [0, 0.15, -0.15]) * Utils.rotz(np.pi/2) * Utils.trn([1, 0, 0.15])
    Q01 = Q00
    Q02 = Q01

    link_3d_obj = []
    link1_mth = Utils.inv_htm(Q01)

    link_3d_obj.append([
        Model3D(url='https://raw.githubusercontent.com/fbartelt/robotics-experiments/main/models/l1.obj',
                scale=scale, htm=link1_mth, mesh_material=mesh,),
    ])

    link2_mth = Utils.inv_htm(Q02)
    link_3d_obj.append([
        Model3D(url='https://raw.githubusercontent.com/fbartelt/robotics-experiments/main/models/l2.obj',
                scale=scale, htm=link2_mth, mesh_material=mesh,),
    ])

    # com_coordinates = [[0.534615, 0, 0.15], [1.5353, 0, 0.15]]
    com_coordinates = [[0.534615 - 1, 0, 0], [1.5353 - 1, 0, 0]]
    list_inertia_mat = []

    # Icm + parallel axis theorem (Inertia mat is in respect to DH frame)
    list_inertia_mat.append(np.diag([0.010533, 0.010573, 0.001061]) +
                            (np.array([-(0.534615 - 1), 0, 0]).reshape(1, -1) @ np.array([-(0.534615 - 1), 0, 0]).reshape(-1, 1)) * np.eye(3) - (np.array([-(0.534615 - 1), 0, 0]).reshape(-1, 1) @ np.array([-(0.534615 - 1), 0, 0]).reshape(1, -1)))
    list_inertia_mat.append(np.diag([0.010573, 0.010613, 0.001061]) +
                            (np.array([-(1.5353 - 1), 0, 0]).reshape(1, -1) @ np.array([-(1.5353 - 1), 0, 0]).reshape(-1, 1)) * np.eye(3) - (np.array([-(1.5353 - 1), 0, 0]).reshape(-1, 1) @ np.array([-(1.5353 - 1), 0, 0]).reshape(1, -1)))
    list_mass = [159.779, 159.949]
    links = []
    for i in range(n):
        links.append(Link(i, theta=link_info[0, i], d=link_info[1, i], alpha=link_info[2, i], a=link_info[3, i], joint_type=link_info[4, i],
                          list_model_3d=link_3d_obj[i], com_coordinates=com_coordinates[i], mass=list_mass[i], inertia_matrix=list_inertia_mat[i]))

    q0 = [0, np.pi/2*0]
    htm_n_eef = Utils.trn([1.05, 0, 0]) @ Utils.roty(np.pi/2)

    # Create joint limits
    joint_limits = np.matrix([[-3*np.pi, 3*np.pi], [-1.8, 1.8]])

    return links, base_3d_obj, htm_base_0, htm_n_eef, q0, joint_limits


def create_2dof(name='jaco_robot', htm=np.identity(4), opacity=1):
    links, base_3d_obj, htm_base_0, htm_n_eef, q0, joint_limits = _create_2dof(
        opacity=opacity)
    jaco = rb.Robot(name=name, links=links, list_base_3d_obj=base_3d_obj, htm=htm,
                    htm_base_0=htm_base_0, htm_n_eef=htm_n_eef, q0=q0, eef_frame_visible=True, joint_limits=joint_limits)
    return jaco


""" TESTS"""


class planar_manipulator(rb.Robot):

    inertia_tensor = []  # In COM coords
    inertia_tensor.append(np.diag([0.010533, 0.010573, 0.001061]))
    inertia_tensor.append(np.diag([0.010573, 0.010613, 0.001061]))

    def __init__(self, name='jaco_robot', opacity=1):
        links, base_3d_obj, htm_base_0, htm_n_eef, q0, joint_limits = _create_2dof(
            opacity=opacity)
        super().__init__(name=name, links=links, list_base_3d_obj=base_3d_obj, htm=np.identity(4),
                         htm_base_0=htm_base_0, htm_n_eef=htm_n_eef, q0=q0, eef_frame_visible=True, joint_limits=joint_limits)

    def M(self, q=None):
        """Computes inertia matrix M(q)
        """
        if q is None:
            q = self.q

        n = len(self.links)
        Jv_list, _ = self.jac_geo(q=q, axis='com')

        for i in range(n):
            Ji = Jv_list[i][:3, :]  # velocity jacobian of ith part
            Wi = Ji = Jv_list[i][3:, :]
            mi = self.links[i].mass
            linear_component = mi * (Ji.T @ Ji)
            rotation_component = Wi.T @ self.inertia_tensor[i] @ Wi
            M_ += linear_component + rotation_component
        return M_

    def G(self, q=None, gravity=9.81):
        """Computes potential energy
        """
        if q is None:
            q = self.q

        V_ = 0
        n = len(self.links)
        Jv_list, _ = self.jac_geo(q=q, axis='com')

        for i in range(n):
            pseudo_height = Jv_list[i][2, :].reshape(-1, 1)
            V_ += self.links[i].mass * gravity * pseudo_height

        return V_
