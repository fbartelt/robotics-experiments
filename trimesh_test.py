# %%
import os
import random
import re
import string
import trimesh as tm
import numpy as np
from itertools import groupby
from typing import List

def group_parts(path: str, extension: str = 'obj', ignore: list = [], group_also: list = [[]]) -> List[List[str]]:
    """Groups model parts into groups based on prefix or ``group_also``.

    Groups are automatically generated based on prefixes, i.e. if there's a hand
    model that is separated into hand_palm, hand_finger1, hand_knuckle1, etc. all
    of them will be automatically grouped by the "hand_" prefix. If your files
    were not organized in this way, you can use ``group_also`` parameter to set a
    list of non-trivial groups. 

    Params
    ------
    path: str
        path of models folder
    extension: str
        the models file extension
    ignore: list
        list of files to ignore (without extension)
    group_also: list of list
        list of non-trivial groups, e.g. [['foo', 'bar'], ['fizz', 'buzz']].
        The comparison for these groups are handled by regex, so if there's some
        pattern, it is possible to use it instead of the complete file name
    """
    # get posix/windows path format correctly
    path = os.path.abspath(os.path.expanduser(path))
    # files without extension
    files = [os.path.splitext(filename)[0] for filename in os.listdir(path) if os.path.splitext(filename)[1]=='.'+extension]
    if ignore:
        files = [f for f in files if f not in ignore]
    if group_also:
        group_keys = [''.join(random.choices(string.ascii_letters, k=8)) + '_'
                      for _ in range(len(group_also))]
        files = [group_keys[m] + f if re.match(fr'({"|".join(group)}).*', f) else f for f in files
                 for m, group in enumerate(group_also)]
    files.sort()
    groups = [list(group)
              for _, group in groupby(files, lambda x: x.split('_')[0])]
    #remove aux group random prefix
    if group_also:
        for i, group in enumerate(groups):
            newgroup = [g.removeprefix(key) for key in group_keys for g in group if key in g]
            if newgroup:
                groups[i] = newgroup
    print(f"=== {len(groups)} groups identified: ===")
    print(groups)
    print('=== If any group is missing, change <group_also> ===\n')

    return groups

def _groupdict_ravel(group_dict: dict) -> dict:
    """ Given a dict {1: ['a', 'b', 'c'], 2:['d', 'e'], 3:'f'}, return a dict
    as {'a': 1, 'b': 1, 'c': 1, 'd': 2, 'e': 2, 'f': 3}
    """
    dict_new = {}
    for key, value in group_dict.items():
        if isinstance(value, (list, tuple)):
            dict_aux = {v:key for v in value}
            dict_new = dict_new | dict_aux
        else:
            dict_new[value] = key
    return dict_new

def get_inertial_parameters(path: str, densities: dict, extension: str = 'obj', ignore: list = [], group_also: list = [[]]) -> None:
    """Calculates mass, center of mass and inertia tensors for models composed
    of multiple parts. Inertia tensors are given in respect to the center of
    mass of the model.

    Params
    ------
    path: str
        path of models folder
    densities: dict
        a dict that maps density values to a list of models, e.g. 
        {1: ['a', 'b', 'c'], 2:['d', 'e'], 3:'f'} if models a,b,c have density 1
        d,e have density 2 and f has density 3.
    extension: str
        the models file extension
    ignore: list
        list of files to ignore (without extension)
    group_also: list of list
        list of non-trivial groups, e.g. [['foo', 'bar'], ['fizz', 'buzz']].
        The comparison for these groups are handled by regex, so if there's some
        pattern, it is possible to use it instead of the complete file name
    """
    groups = group_parts(path=path, ignore=ignore, group_also=group_also)
    densities = _groupdict_ravel(densities)
    mass_list, com_list, tensor_list = [], [], []
    # model_mesh = tm.creation.axis(origin_size=0.02)
    scene = tm.Scene(tm.creation.axis(origin_size=0.02))
    for i, group in enumerate(groups):   
        print(f'Group {i}: ', group)
        mass = 0
        com = np.zeros(3)
        for part in group:
            full_path = os.path.join(path, part + '.' + extension.lstrip('.'))
            mesh = tm.load_mesh(full_path, file_type=extension.lstrip('.'))
            # model_mesh = tm.util.concatenate([model_mesh, mesh])
            scene.add_geometry(mesh)
            mesh.density = densities[part]
            props = mesh.mass_properties
            mass += props.mass
            com += props.center_mass * props.mass
            # print(props.mass, np.round(props.center_mass, 6), np.round(props.inertia, 6))
        # print(f'Mass: {mass}\nCenter Of Mass: {com}')
        com /= mass
        mass_list.append(mass)
        com_list.append(com)
        axis = tm.creation.axis(transform=tm.transformations.translation_matrix(com), origin_size=0.02)
        # model_mesh = tm.util.concatenate([model_mesh, axis])
        scene.add_geometry(axis)
        tensor = np.zeros((3,3))
        for part in group:
            full_path = os.path.join(path, part + '.' + extension.lstrip('.'))
            mesh = tm.load_mesh(full_path, file_type=extension.lstrip('.'))
            mesh.density = densities[part]
            tensor += mesh.moment_inertia_frame(tm.transformations.translation_matrix(com))
        tensor_list.append(tensor)
        print(f'Mass: {mass}\nCenter Of Mass: {np.round(com, 6)}\nInertia Tensor: {np.round(tensor, 6)}')
    
    return mass_list, com_list, tensor_list, scene

densities = {1200:['finger1_mounting', 'finger1_distal', 'finger1_proximal', 
                   'finger1_nail', 'finger1_proximal', 'finger2_mounting', 
                   'finger2_distal', 'finger2_proximal', 'finger2_nail', 
                   'finger2_proximal', 'thumb_mounting', 'thumb_distal', 
                   'thumb_proximal', 'thumb_nail', 'thumb_proximal', 'handpalm', 
                   'forearm_ring', 'shoulder_ring', 'upperarm_ring', 
                   'wrist1_ring', 'wrist2_ring', 'base_ring'], 
            1750:['base', 'forearm', 'gripper', 'shoulder', 'upperarm', 
                  'wrist1', 'wrist2'], 
            2710:['finger1_actuator','finger2_actuator', 'thumb_actuator', 
                  'forearm_actuator', 'shoulder_actuator', 'upperarm_actuator', 
                  'wrist1_actuator', 'wrist2_actuator', 'base_actuator']}

mass_list, com_list, tensor_list, model_mesh = get_inertial_parameters('models/jaco', densities=densities, ignore=['hand3fingers'], group_also=[['handpalm', 'gripper', 'finger', 'thumb']])
# mass_list, com_list, tensor_list = get_inertial_parameters('models', {1750:['l1', 'l2']})
# mesh = tm.load_mesh('models/jaco/shoulder.obj', file_type='obj')
# mesh2 = tm.load_mesh('models/l1.obj', file_type='obj')
# print(mesh.moment_inertia)
# %%
model_mesh.show(viewer='gl')

# %%
