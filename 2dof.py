# %%
import sys
sys.path.insert(1, '/home/fbartelt/Documents/UFMG/TCC/Sim/uaibot/uaibot')
import time
import numpy as np
import robot as rb
import plotly.graph_objects as go
from simulation import Simulation
from utils import Utils
from simobjects.box import Box
from simobjects.ball import Ball
from simobjects.frame import Frame
from simobjects.pointcloud import PointCloud
from cvxopt import matrix, solvers
from graphics.meshmaterial import MeshMaterial, Texture
from graphics.model3d import Model3D
from robot.links import Link
from plotly.subplots import make_subplots
from demo import Demo
from IPython.display import clear_output


def _create_2dof(name='double_pendulum', color='#3e3f42', opacity=1):

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

    links = []
    for i in range(n):
        links.append(Link(i, theta=link_info[0, i], d=link_info[1, i], alpha=link_info[2, i], a=link_info[3, i], joint_type=link_info[4, i],
                          list_model_3d=link_3d_obj[i]))

    q0 = [0, np.pi/2*0]
    htm_n_eef = Utils.trn([1.05, 0, 0]) @ Utils.roty(np.pi/2)

    # Create joint limits
    joint_limits = np.matrix([[-3*np.pi, 3*np.pi], [-3*np.pi, 3*np.pi]])

    return links, base_3d_obj, htm_base_0, htm_n_eef, q0, joint_limits


def create_2dof(name='jaco_robot', opacity=1):
    links, base_3d_obj, htm_base_0, htm_n_eef, q0, joint_limits = _create_2dof(
        name='double_pendulum', opacity=opacity)
    jaco = rb.Robot(name='robot2dof', links=links, list_base_3d_obj=base_3d_obj, htm=np.identity(4),
                    htm_base_0=htm_base_0, htm_n_eef=htm_n_eef, q0=q0, eef_frame_visible=True, joint_limits=joint_limits)
    return jaco


double_pendulum = create_2dof(opacity=0.7)
_, htms_dh = double_pendulum.jac_geo(axis='dh')
sim = Simulation.create_sim_grid([double_pendulum])
sim.add(Frame(name='base', htm=double_pendulum.htm_base_0, size=0.5))
frames = []
for i, htm in enumerate(htms_dh[1:]):
    frames.append(Frame(name=f'dh_{i}', htm=htm, size=0.3))

for frame in frames:
    sim.add(frame)

for i in range(40):
    q = [i*0.1, i*0.2]
    double_pendulum.add_ani_frame(time=i*0.05, q=q)
    htms = double_pendulum.fkm(q=q, axis='dh')
    for j, frame in enumerate(frames):
        frame.add_ani_frame(time=i*0.05, htm=htms[j])


sim.run()
# %%
