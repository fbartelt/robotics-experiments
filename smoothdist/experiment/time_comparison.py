import numpy as np
import uaibot as ub
import matplotlib
import matplotlib.pyplot as plt
from utils_paper import *
import time
import uaibot_cpp_bind as ub_cpp
import statistics as stats


######################################################
#PARAMETERS
######################################################


#Maximum number of iterations for the generalized Von Neumman's algorithm
no_iter_max = 2000

#Tolerance for convergence for the generalized Von Neumman's algorithm
tol = 1e-5

#Number of tests per smoothing parameter h
no_tests_per_h = 500

#Epsilon
eps = 0.001

#List of different values of h to test
list_h = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08]

######################################################
# CODE
######################################################

def test_dist(h):
    global no_iter_max
    global tol
    global no_tests_per_h
    global eps
    
    time_gjk = []
    time_ours = []


    for i in range(no_tests_per_h):
        
        #Generate random boxes and cylinders
        box = generate_rand_object(1)
        cylinder = generate_rand_object(2)
        a0 = ub.Utils.htm_rand()[0:3,-1]

        ######################################
        #Compute our distance
        ######################################
        
        t0 = time.process_time()
        _, _, _, hist_error = ub.Utils.compute_dist(box, cylinder, h = h, eps=eps, tol = tol, p_a_init = a0, no_iter_max = no_iter_max)
        tf = time.process_time() - t0

        ######################################
        #Compute Euclidean distance
        ######################################
        
        t0 = time.process_time()
        dr = ub.Utils.compute_dist(box, cylinder)
        tf = time.process_time() - t0
        
        time_gjk.append(tf)
        
    
    return [10**6*t for t in time_gjk], [10**6*t for t in time_ours]
    


hist_m_gjk = []
hist_s_gjk = []
hist_m_ours = []
hist_s_ours = []
hist_t_gjk = []
hist_t_ours = []


i = 0
for h in list_h:
    print("Percentage: "+str(round(100*i/len(list_h))))
    t_gjk, t_ours = test_dist(h)
  
    hist_t_gjk+=t_gjk
    hist_t_ours+=t_ours
    i+=1

    

mean_t_gjk = round(stats.mean(hist_t_gjk),1)
mean_t_ours = round(stats.mean(hist_t_ours),1)
std_t_gjk = round(stats.stdev(hist_t_gjk),1)
std_t_ours = round(stats.stdev(hist_t_ours),1)

print("------------------------------------------")
print("Statistics for average time taken for computation")
print("Number of tests: "+str(len(list_h)*no_iter_max))
print("Time to compute our distance: "+str(mean_t_ours)+" +-"+str(std_t_ours)+" mus")
print("Time to compute Euclidean distance: "+str(mean_t_gjk)+" +-"+str(std_t_gjk)+" mus")