import numpy as np
import uaibot as ub
from create_franka_emika_3_mod import *
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Qt5Agg') 


######################################################
#PARAMETERS
######################################################

#Operation mode, mode = 0 (Euclidean) and mode = 1 (Our proposed distance)
mode = 1

#Choose the parameters h, epsilon (eps) and sigma in the paper.
#This is already pre-selected by the 'mode' variable, but you can change it if you want.
#As an heuristic, sigma is always set internally to sqrt(max(0,1-2*eps)).
#The value of k is set to 1 (so the distance is always one time differentiable).
#It also selects the safety margin delta for obstacles, delta_obs 
# (in meters) and for auto collision delta_auto. They both should be different depending on the
#smoothing parameter.

if mode == 0:
    h=1e-6
    eps=0
    delta_obs=0.03
    delta_auto=0.01
else:
    h=0.1
    eps=0.01
    delta_obs=0.002
    delta_auto=0.0001
    
    
#Obstacles
obstacles = []
obstacles.append(ub.Box(htm = ub.Utils.trn([0.53, 0.16, 0.45]), width=0.35,depth=0.05,height=0.90,color='magenta'))
obstacles.append(ub.Box(htm = ub.Utils.trn([0.53,-0.16, 0.45]), width=0.35,depth=0.05,height=0.90,color='magenta'))
obstacles.append(ub.Box(htm = ub.Utils.trn([0.53, 0.00, 0.925]), width=0.35,depth=0.35,height=0.05,color='magenta'))

#Initial configuration (rad)
q = np.matrix([[ 1.0582, -1.3811,  0.3629, -1.9647, -0.959,   1.4881, -0.1534]]).T

#Target pose
htm_tg = ub.Utils.trn([0.64,0,0.75])*ub.Utils.roty(np.pi/2)*ub.Utils.rotz(np.pi/2) 

#Sampling time (seconds)
dt = 0.01 

#Control matrix for the task function (1/second)
K = np.diag([0.4,0.4,0.4,0.4,0.4,0.4])

#Regularization factor for the task function
reg = 0.01

#Gain for the CBF inequality (1/second)
eta = 0.5

#Maximum simulation time (seconds)
t_max = 35

#Maximum number of iterations for the generalized Von Neumman's algorithm
no_iter_max = 300

#Tolerance for convergence for the generalized Von Neumman's algorithm
tol = 2e-4


######################################################
# INITIALIZATIONS
######################################################

sim_time = 0
robot = create_franka_emika_3_mod()

sim = ub.Simulation(background_color='lightblue')
sim.add(robot)
robot.add_ani_frame(0,q)

for obs in obstacles:
    sim.add(obs)
    
frame_tg = ub.Frame(htm=htm_tg)
sim.add(frame_tg)

#Auxiliary functions



def get_joint_config():
    #In a real application, this should be replaced by the 
    #function that measures the real joint position in the robot
    global robot
    return robot.q

def compute_controller(_q):
    #Compute the control input 
    global robot
    global htm_tg
    global eps
    global delta
    global K
    global eta
    global obstacles
    global reg
    global no_iter_max
    
    #Get the number of configurations
    n = np.shape(_q)[0]

    #Initialize matrices A and b
    mat_A = np.matrix(np.zeros((0,n)))
    mat_b = np.matrix(np.zeros((0,1)))
    
    #Implement obstacle avoidance constraints and stack into A and b
    for obs in obstacles:
        dr = robot.compute_dist(q=_q, obj=obs, h=h, eps=eps,no_iter_max=no_iter_max,tol=tol)
        mat_A = np.vstack((mat_A, dr.jac_dist_mat))
        mat_b = np.vstack((mat_b, -eta*(dr.dist_vect-delta_obs)))
        
    #Implement auto-collision avoidance and stack into A and b
    dr = robot.compute_dist_auto(q=_q, h=h, eps=eps,no_iter_max=no_iter_max,tol=tol)
    mat_A = np.vstack((mat_A, dr.jac_dist_mat))
    mat_b = np.vstack((mat_b, -eta*(dr.dist_vect-delta_auto)))
        
    #Implement constraints for joint limits avoidance and stack into A and b   
    mat_A = np.vstack((mat_A, np.identity(n)))
    mat_b = np.vstack((mat_b, -eta*(_q-robot.joint_limit[:,0])))
    
    mat_A = np.vstack((mat_A,-np.identity(n)))
    mat_b = np.vstack((mat_b,-eta*(robot.joint_limit[:,1]-_q)))
    
    #Compute task function
    r, jac_r = robot.task_function(q=_q, htm_tg=htm_tg)
    
    #Assemble the H and f matrices of the optimization problem 
    mat_H = jac_r.T*jac_r + reg*np.identity(n)
    mat_f = jac_r.T*(K*r)
    
    #Compute the control input 
    try:
        u = ub.Utils.solve_qp(mat_H, mat_f, mat_A, mat_b)
    except:
        u = 0*_q
        print("Unfeasible!")
    
    return u
        
def send_joint_velocity(_dotq):
    #In a real application, this should be replaced by the
    #function that sends the joint velocity _dotq to the robot
    global robot
    global dt
    global sim_time
    
    sim_time += dt
    robot.add_ani_frame(sim_time, robot.q+_dotq*dt)
    
#Simulation
hist_u=[]
hist_t=[]
for i in range(round(t_max/dt)):
    print("Percent: "+str(round(100*sim_time/t_max)))
    q = get_joint_config()
    u = compute_controller(q)
    send_joint_velocity(u)
    
    hist_u.append(u)
    hist_t.append(sim_time)
    
print("Done!")
   
#Plot the control input of the last joint
plt.plot(hist_t, [u[-1,0] for u in hist_u])
plt.show()


#Save  the simulation to see the results (open the html file control_sim.html that will 
# be generated in the same folder the script was ran)

sim.save(file_name = "control_sim")


    