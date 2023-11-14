# %%
import os
import random
import re
import string
import trimesh as tm
import numpy as np
from itertools import groupby
from typing import List

def group_parts(path: str, ignore: list = [], group_also: list = [[]]) -> List[List[str]]:
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
    files = [os.path.splitext(filename)[0] for filename in os.listdir(path)]
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

def get_COM(path: str, extension: str = 'obj', ignore: list = [], group_also: list = [[]]) -> None:
    """Calculates....

    Params
    ------
    path: str
        path of models folder
    ignore: list
        list of files to ignore (without extension)
    group_also: list of list
        list of non-trivial groups, e.g. [['foo', 'bar'], ['fizz', 'buzz']].
        The comparison for these groups are handled by regex, so if there's some
        pattern, it is possible to use it instead of the complete file name
    """
    groups = group_parts(path=path, ignore=ignore, group_also=group_also)
    for i, group in enumerate(groups):
        if i > 1:
            break   
        print(f'Group {i}: ', group)
        mass = 0
        com = np.zeros(3)
        tensor = np.zeros((3,3))
        for part in group:
            full_path = os.path.join(path, part + '.' + extension.lstrip('.'))
            mesh =  tm.load_mesh(full_path, file_type='obj')
            props = mesh.mass_properties
            mass += props.mass
            com += props.center_mass
            tensor += props.inertia
            print(props.mass, props.center_mass)
        print(f'Mass: {mass}\nCenter Of Mass: {com}')
        com /= mass
        tensor /= mass
        print(f'Mass: {mass}\nCenter Of Mass: {com}\nInertia Tensor: {tensor}')

get_COM('models/jaco', ignore=['hand3fingers'], group_also=[['handpalm', 'gripper', 'finger', 'thumb']])
# mesh = tm.load_mesh('models/jaco/shoulder.obj', file_type='obj')
# mesh2 = tm.load_mesh('models/l1.obj', file_type='obj')
# print(mesh.moment_inertia)
# %%
