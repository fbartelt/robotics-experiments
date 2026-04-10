import numpy as np
import uaibot as ub
import numpy as np
from scipy.optimize import linprog
from scipy.spatial import HalfspaceIntersection, ConvexHull

def is_feasible_and_bounded(A, b):
    n_dim = A.shape[1]

    for i in range(n_dim):
        c = np.zeros(n_dim)
        c[i] = 1  

        res = linprog(c, A_ub=A, b_ub=b, method='highs', bounds=[(None, None)] * n_dim)

        if res.status != 0 or abs(res.x[i])>1e2: 
            return False  

        c[i] = -1  

        res = linprog(c, A_ub=A, b_ub=b, method='highs', bounds=[(None, None)] * n_dim)

        if res.status != 0 or abs(res.x[i])>1e2: 
            return False  
        
    return True  

def generate_bounded_polytope(n_halfspaces=10, size = 1.0):
    while True:
        rng = np.random.default_rng(None)

        m = int(n_halfspaces)
        if m < 4:
            raise ValueError("n_halfspaces must be >= 4 to form a bounded polyhedron in 3D.")
        if size <= 0:
            raise ValueError("size must be > 0.")

        # 1) Random unit normals uniformly on S^2
        #    (normalizing Gaussian vectors)
        normals = rng.normal(size=(m, 3))
        normals /= np.linalg.norm(normals, axis=1, keepdims=True)

        # 2) Pick an interior feasible point
        center = rng.normal(scale=5 * size, size=3)

        # 3) Positive offsets (support distances)
        margins = rng.uniform(0.6 * size, 1.4 * size, size=m)

        # Enforce feasibility of 'center'
        b_arr = (normals @ center) + margins

        A = np.matrix(normals)
        b = np.matrix(b_arr).reshape(-1, 1)
        
        if is_feasible_and_bounded(A, b):
            return A, b

def generate_rand_object(objtype = -1, size = 1.0, spread = 1.0, color='red'):
    
    if objtype == -1:
        coin = np.random.randint(0,5)
    else:
        coin = objtype

    htm = ub.Utils.htm_rand([-spread*1.5,-spread*1.5,spread*0.5],[spread*1.5,spread*1.5,spread*1.5])
    
    if coin==0:
        r = size* np.random.uniform(0.3,1.0)
        obj = ub.Ball(htm=htm, radius=r, color=color)
        
        return obj
        
    if coin==1:
        w = size*np.random.uniform(0.3,0.8)
        d = size*np.random.uniform(0.3,0.8)
        h = size*np.random.uniform(0.3,0.8)
        obj = ub.Box(htm=htm, width=w, depth=d, height=h, color=color)
        
        return obj


    if coin==2:
        r = size*np.random.uniform(0.3,0.8)
        h = size*np.random.uniform(0.3,0.8)
        obj = ub.Cylinder(htm=htm, radius = r, height=h, color=color)
        
        return obj

        
    if coin==3:
        n = np.random.randint(500,1000)  
        m1 = np.matrix(np.random.uniform(low=0, high=1, size=(3, n))) 
        m2 = np.matrix(np.random.uniform(low=0, high=1, size=(3, n)))
        w = size*np.random.uniform(0.3,0.8)
        d = size*np.random.uniform(0.3,0.8)
        h = size*np.random.uniform(1.0,1.5)
        pc = np.matrix([w,d,h]).transpose()
        
        w = size*np.random.uniform(0.3,0.8)
        d = size*np.random.uniform(0.3,0.8)
        h = size*np.random.uniform(1.0,1.5)
        
        points = np.diag([w,d,h])*htm[0:3,0:3]*(m1+m2)+pc
        obj = ub.PointCloud(points = points, color=color)
        
        return obj
 
        
    if coin==4:
        A, b = generate_bounded_polytope(size = size)
        obj = ub.ConvexPolytope(A = A, b = b, color=color)
        obj.add_ani_frame(0,htm)
        
        return obj

def create_cone(htm, radius, height):
#Create a cone-like object using Uaibot's capabilities of creating convex polytopes 
#defined by Ap<=b. 'htm' is the homogeneous transformation matrix representing the pose of the cylinder.
#'radius' is the radius of the base of the cone, and 'height' the height of the cone

    #Create the side inequalities Ap<=b for the cone
    no_sides = 6
    A = np.matrix(np.zeros((0,3)))
    b = np.matrix(np.zeros((0,1)))


    phi = np.arcsin(radius/np.sqrt(height**2+radius**2))

    for i in range(no_sides):
        theta = 2*np.pi*i/no_sides
        a = np.matrix([np.cos(phi)*np.cos(theta), np.cos(phi)*np.sin(theta), np.sin(phi)])
        A = np.vstack((A,a))
        b = np.vstack((b,radius))
        
    ##Add an inequality in the bottom (to cover the cone from the bottom)
    A = np.vstack((A,np.matrix([0,0,-1])))
    b = np.vstack((b,0))
    
    #Create the convex polytope
    cone = ub.ConvexPolytope(A=A,b=b,htm=ub.Utils.trn([0,0,0]),color='yellow')
    
    #Set the desired htm at time t=0
    cone.add_ani_frame(time=0,htm=htm)
    
    return cone
   
def create_pyramid(htm, radius, height):
# Create a 4-sided pyramid using Uaibot's ConvexPolytope.
# The square base lies in the plane z=0, centered at (0,0,0),
# with side length 2*radius, and the apex is at (0,0,height).



    # Define the apex in local coordinates.
    apex = np.array([0, 0, height])

    # We'll collect A and b for the half-spaces A*x <= b.
    A_list = []
    b_list = []

    # Number of sides in the base (4 -> a square).
    no_sides = 4

    # Loop over each side of the square base
    for i in range(no_sides):
        # Corner i of the base
        theta1 = 2 * np.pi * i / no_sides
        # Corner i+1 (the "next" corner)
        theta2 = 2 * np.pi * (i + 1) / no_sides

        # Points in R^3 (local frame)
        p1 = apex  # apex
        p2 = np.array([radius * np.cos(theta1), radius * np.sin(theta1), 0])
        p3 = np.array([radius * np.cos(theta2), radius * np.sin(theta2), 0])

        # Normal vector to the plane (p1, p2, p3)
        normal = np.cross(p2 - p1, p3 - p1)

        # We want the "inside" of the plane to include the center of the base (0,0,0).
        # So we check the sign of normal dot (center - apex).
        center = np.array([0, 0, 0])
        if np.dot(normal, center - p1) > 0:
            normal = -normal

        # The plane is normal^T x <= normal^T p1
        A_list.append(normal)
        b_list.append(np.dot(normal, p1))

    # Finally, add the plane for z >= 0, i.e. -z <= 0
    A_list.append([0, 0, -1])
    b_list.append(0)

    # Convert everything to numpy arrays
    A = np.array(A_list, dtype=float)
    b = np.array(b_list, dtype=float).reshape(-1, 1)

    # Build the convex polytope at the local frame
    pyramid = ub.ConvexPolytope(A=A, b=b, htm=ub.Utils.trn([0, 0, 0])*ub.Utils.rotz(np.pi/4), color="yellow")

    # Now apply the desired pose 'htm'
    pyramid.add_ani_frame(time=0, htm=htm)

    return pyramid
   
       
def load_timed_data(filename):
    t = []
    data = []
    with open(filename, "r") as f:
        for line in f:
            parts = list(map(float, line.strip().split()))
            t.append(parts[0])
            data.append(parts[1:])
    return t, data
   
def funF(r):
#Function that is used in the control loop

    f = np.matrix(r)
    for j in range(np.shape(r)[0]):
        f[j,0] = np.sign(r[j,0])*np.sqrt(np.abs(r[j,0]))
        
    return f     
        
