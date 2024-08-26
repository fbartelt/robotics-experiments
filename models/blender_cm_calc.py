import bpy
import time
from enum import Flag,auto
from mathutils import Vector
"""https://blender.stackexchange.com/questions/31062/how-can-i-calculate-and-draw-the-center-of-mass-of-various-objects-with-python
Open this and run inside blender scripting tab
"""
# ---------------------------------------
# some parameters you might want to tweak
# ---------------------------------------

# Don't generate an exception when errors are detected. Enable at your own risk
Quiet = False

# Print a list of all density-defining materials and associated massive objects
# Temporarily enabling this option might help locating your "massive" objects, organizing your materials or sorting out typos.
Report=True

# Ignore hidden objects even when they are selected. Can be safely disabled, at your convenience
SkipHiddenObjects = True

# Use a dedicated object to mark the CG position
# If the object is not found, the cursor will be moved instead
CGMarkerName="CG marker"

# ---------------------------------------
# units
# ---------------------------------------

#   As a convenience, a dictionary of metric and imperial units is provided to make the definitions less painful
#   The only requirement is to keep the same weight unit for all densities. If you want to do more fancy stuff, you'll
# have to define the conversion coefficients yourself.
#   For instance this will set units to pounds, pounds/yard, pounds/square inch and pounds/cubic foot
# (assuming you read the "10.5" displayed by the script as "10.5 pounds")
#LinearUnit     = "yd"
#SurfaceUnit    = "in"
#VolumetricUnit = "ft"

LinearUnit     = "m"   
SurfaceUnit    = "dm"
VolumetricUnit = "m" #cm

Units = { # The coefficients are simply units expressed in meters
    # metric
    "km":1000,"hm":100,"dam":10, "m" : 1, "dm" : 0.1, "cm":0.01, "mm":0.001, "um":0.000001,
    # imperial
    "mi" :1609.344, # mile
    "fur":2012.168, # furlong
    "ch" :20.1168,  # chain
    "yd" :0.9144,   # yard
    "ft" :0.3048,   # foot
    "in" :0.0254,   # inch
    "mil":0.0000254 # thou
}

# You can chuck my little list of units out the window and set these coefficients directly, if you know what you're doing
scaling = bpy.context.scene.unit_settings.scale_length
ConvertLength  = pow (Units[LinearUnit    ] / scaling,-1)
ConvertSurface = pow (Units[SurfaceUnit   ] / scaling,-2)
ConvertVolume  = pow (Units[VolumetricUnit] / scaling,-3)

# ------------------------------------------------------------------------------------------------
# The code proper. Tinker at your own risk
# ------------------------------------------------------------------------------------------------

failure = False 
def croak (*args, **kwargs):
    global failure # hopefully not!
    print ("!?!", *args, *kwargs)
    failure = True

class Density (Flag):
    Punctual   = auto()
    Linear     = auto()
    Surface    = auto()
    Volumetric = auto()
Weightless = Density(0)

ValidDensities = {
    'EMPTY'   : Density.Punctual,
    'SURFACE' : Density.Punctual |                  Density.Surface,
    'CURVE'   : Density.Punctual | Density.Linear | Density.Surface | Density.Volumetric,
    'META'    : Density.Punctual |                  Density.Surface | Density.Volumetric,
    'MESH'    : Density.Punctual | Density.Linear | Density.Surface | Density.Volumetric
    }

def triangles (polygon):
    """enumerate triangles in a face"""
    # Loop triangles are not available for visualization meshes, we need to use polygonal faces instead
    for i in range (1, len(polygon)-1):
        yield (polygon[0], polygon[i], polygon[i+1])

def scale_vertices (vertices, scale):
    """ scale object coordinates (according to world matrix) """
    # pre-scaling vertices is more than 2.5 times faster, for a reasonable memory cost
    # lists appear to be slightly faster than tuples in that case
    return [Vector (x * y for x,y in zip (vertex.co,scale)) for vertex in vertices]

def mesh_length (mesh,scale):
    """center of gravity and mass of a mesh seen as a wire frame (a collection of edges with no faces)"""
    center = Vector()
    length = 0
    scaled_vertices = scale_vertices (mesh.vertices, scale)
    for segment in mesh.edges:
        a, b = (scaled_vertices[v] for v in segment.vertices)
        l = (a-b).length        # segment length
        center += l * (a+b)     # 2 x middle point, weighted by length
        length += l
    if length != 0: # constants moved out of the loop
        center /= length * 2 
    return center, length
    
def mesh_surface (mesh,scale):
    """center of mass of a mesh seen as an infinitely thin surface"""
    center = Vector()
    surface = 0
    scaled_vertices = scale_vertices (mesh.vertices, scale)
    for face in mesh.polygons:
        for triangle in triangles (face.vertices):
            a,b,c = (scaled_vertices[v] for v in triangle)
            s = (b-a).cross(c-a).length # 2 x triangle surface
            center += s * (a+b+c)       # 2 x 3 x center, weighted by surface
            surface += s
    if surface != 0:  # constants moved out of the loop
        center /= surface * 3
        surface /= 2 
    return center, surface
    
def mesh_volume (mesh,scale):
    """center of mass of a mesh seen as a constant density volume (only watertight meshes will produce correct results)"""
    # The algorithm sums signed volumes of tetrahedrons built by adding a fixed reference point to every triangle
    # in the mesh. When the mesh is watertight, the bits of tetrahedrons that lie outside the actual
    # shape end up canceling each other out. When the shape is leaky, bits of tetrahedrons are left sticking
    # out, so to speak, which can produce complete garbage, even with inconspicuous holes.
    #   Leaky shapes are very likely to yield different results when the reference point changes. The heuristic
    # picks a second reference point beside the origin, computes both volumes and looks for discrepancies.
    #   Some leaky meshes could slip under the radar, but it's simple and apparently efficient
    center = Vector()
    volume = 0
    d = mesh.vertices[0].co # pick an arbitrary point from the mesh for water tightness check
    vcheck = 0
    for face in mesh.polygons:
        for triangle in triangles(face.vertices):
            a,b,c = (mesh.vertices[v].co for v in triangle)
            v = a.cross(b).dot(c) # 6 x volume of a tetrahedron with the 4th point at the origin
            center += v * (a+b+c) # 6 x 4 x center, weighted by volume (the 4th point has null coordinates)
            volume += v
            vcheck += (a-d).cross(b-d).dot(c-d) # 6 x volume of a second tetrahedron using another reference point
    vmin = min (volume,vcheck)
    if vmin != 0 and abs(volume-vcheck)/vmin > 0.01: # 1% relative error
        center = None # the caller just cannot overlook this, but it's a terrible way of reporting an error
    else:    
        # contrary to length and surface, volume can be scaled globally without adjusting every single vertex.
        for factor in scale: volume *= factor
        if volume != 0: # constants moved out of the loop
            center /= volume * 4
            volume /= 6 
    return center, volume

def obj_density (obj):
    """retrieve density settings from object and material custom properties"""
    density = Weightless
    factor = 0
    try:
        factor = obj['weight']
        density = Density.Punctual
    except KeyError:
        count = 0
        if obj.active_material != None:
            #print(obj.active_material.name, obj.active_material.keys(), obj.name)
            try:
                factor = obj.active_material['linear density'] * ConvertLength
                density |= Density.Linear
                count+=1
            except KeyError: pass
            try:
                factor = obj.active_material['surface density'] * ConvertSurface
                density |= Density.Surface
                count+=1
            except KeyError: pass
            try:
                factor = obj.active_material['density'] * ConvertVolume
                density |= Density.Volumetric
                count+=1
            except KeyError: pass
            if count > 1 :
                croak ("Multiple densities defined in material", obj.active_material.name, ", skipping object", obj.name)
                return Weightless,0
 
    # check compatibility with object type
    if obj.type in ValidDensities:
        if not density in ValidDensities[obj.type]:
            croak (obj.name,"of type",obj.type,"cannot have a", density.name.lower(), "density")
            return Weightless,0
    else:
        croak (obj.name,"of type",obj.type,"cannot be assigned a weight")
        return Weightless,0
    return density, factor
    
def center_of_gravity_and_weight (obj, density, factor):
    # take care of punctual masses
    if density == Density.Punctual: return obj.location, factor
    
    # prepare object for processing (i.e. strip the curves from their volume when needed)
    if obj.type == 'CURVE' and density == Density.Linear:
        # remove beveling and extrusion for curves with linear weight
        prepared = obj.copy()
        prepared.data = obj.data.copy()
        prepared.data.bevel_depth = 0
        prepared.data.bevel_object = None
        prepared.data.extrude = 0
    else:
        # other objects need no tweaking before applying their modifiers
        prepared = obj
        
    # apply active modifiers
    evaluated = prepared.evaluated_get(depsgraph)
    
    # convert object data to mesh if needed
    if obj.type == 'MESH':
        mesh = evaluated.data
    else:
        mesh = bpy.data.meshes.new_from_object(evaluated)

    # last sanity checks now that we got the final mesh
    center, weight = None, 0
    if not mesh.polygons and not mesh.edges:
        croak (obj.name,"has no computable geometry and cannot be assigned a weight. You might want to use an empty instead")
    elif mesh.polygons and density == Density.Linear:
        croak (obj.name,"has a surface and cannot be assigned a linear weight")
    elif not mesh.polygons and density != Density.Linear:
        croak (obj.name,"is a wire frame and can only be assigned a linear weight")
    else:
        # compute center and weight at last
        center,magnitude = {
            Density.Linear     : mesh_length,
            Density.Surface    : mesh_surface,
            Density.Volumetric : mesh_volume
            }[density](
                mesh,
                obj.matrix_world.to_scale()) # take scaling into account        
        weight = magnitude * factor
        if center == None:
            croak(obj.name, "might not be watertight")
    # cleanup
    if mesh != evaluated.data: bpy.data.meshes.remove(mesh)
    
    # back to world coordinates
    if center == None: center = Vector() # position the CG at local origin if volume computation failed
    return obj.matrix_world @ center, weight

# ------------------------------------------------------------------------------------------------
# Main 
# ------------------------------------------------------------------------------------------------
center = Vector()
file_name = '/home/fbartelt/Documents/Projetos/robotics-experiments/comNmasses.txt'
weight = 0

def write_to_file(filename, msg):
    with open(filename, 'a') as file1:
        file1.write(msg)
        file1.write('\n')
# allows to access the representation of a primitive object with active modifiers applied
depsgraph = bpy.context.evaluated_depsgraph_get()
write_to_file(file_name, '')

start = time.time()
for obj in bpy.context.selected_objects:
    print(obj.name)
    write_to_file(file_name, obj.name)
    # optionally skip hidden objects
    if SkipHiddenObjects and obj.hide_render: continue
    
    # get density definitions from object and material custom properties
    density, factor = obj_density (obj)
    if density == Weightless : continue # skip invalid or weightless objects
    
    # attempt CG computation
    c, w = center_of_gravity_and_weight (obj, density, factor)
    center += c * w
    weight += w

if Report:
    # display weighty materials and objects
    weighty_materials = []
    for density in ("linear ","surface ",""):
        print(f"====== {density if density != '' else 'volumetric '}density")
        write_to_file(file_name, f"====== {density if density != '' else 'volumetric '}density")
        for mat in bpy.data.materials:
            if not density+"density" in mat: continue
            weighty_materials.append(mat)
            print (f"--- {mat.name} ({mat[density+'density']:.2f})")
            for obj in bpy.data.objects:
                if obj.get("weight") is not None: continue
                if obj.active_material == mat: print (obj.name)
    print("====== individual weights")
    for obj in bpy.data.objects:
        if obj.get("weight") is None: continue
        print (f"{obj.name} ({obj['weight']:.2f})",end="")
        if obj.active_material in weighty_materials:
            print (" overriding", obj.active_material.name,end="")
        print('')
                
if weight != 0:
    center /= weight
    try:
        # move the CG marker object if present
        marker = bpy.data.objects[CGMarkerName]
        marker.location = center
    except KeyError:
        # or the cursor if no marker was found
        bpy.context.scene.cursor.location = center
    print (f"CG [{center.x:.6f} {center.y:.6f} {center.z:.6f}], total weight {weight:.6f}, done in {time.time()-start:.2f}s")
    write_to_file(file_name, f"CG [{center.x:.6f} {center.y:.6f} {center.z:.6f}], total weight {weight:.6f}, done in {time.time()-start:.2f}s")
else :
    print ("No valid massive object found")
# throwing an exception is ugly, but a sure way to get the user's attention
if failure:
    msg = "Problems were detected. The result might be invalid"
    if Quiet: print(msg)
    else: raise Exception(msg)