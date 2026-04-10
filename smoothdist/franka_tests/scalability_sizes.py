import numpy as np
import uaibot as ub
import matplotlib
import matplotlib.pyplot as plt
from utils_paper import *
import time
import uaibot_cpp_bind as ub_cpp
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Qt5Agg') 

######################################################
#PARAMETERS
######################################################


#Maximum number of iterations for the generalized Von Neumman's algorithm
no_iter_max = 5000

#Tolerance for convergence for the generalized Von Neumman's algorithm
tol = 1e-4

#Number of tests per smoothing parameter h
no_tests_per_h = 50

#Maximum number of faces to try (from 5 to this number)
no_faces = 100

#Epsilon
eps = 0.001

#List of different values of h to test
list_h = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08]

#Minimum Euclidean distance between the objects to try 
min_dist = 0.05

######################################################
# CODE
######################################################


time_taken = []
no_failures = 0
no_total = 0

for j in range(5,no_faces+1):
    print("Percentage: "+str(round(100*(j-5)/(no_faces-5))))
    time_taken.append([])
    for i in range(no_tests_per_h):
         
        cont = True
        
        while cont:
            A, b = generate_bounded_polytope(n_halfspaces=j)
            obj1 = ub.ConvexPolytope(A = A, b = b)
            
            A, b = generate_bounded_polytope(n_halfspaces=j)
            obj2 = ub.ConvexPolytope(A = A, b = b)
            
            cont = obj1.compute_dist(obj2)[2]<min_dist
            
        for h in list_h:
            no_total+=1
            t0 = time.process_time()
            _, _, _, hist_error = ub.Utils.compute_dist(obj1, obj2, h = h, eps=0.001, tol = tol,  no_iter_max = no_iter_max)
            tf = time.process_time() - t0
            
            if len(hist_error)>no_iter_max:
                #A failure is when the algorithm was not able to achieve the desired
                #precision with the number of iterations
                no_failures+=1
            else:
                time_taken[-1].append(10**6*tf)


means = np.array([np.mean(v) for v in time_taken])
stds  = np.array([np.std(v, ddof=1) for v in time_taken])  # sample std

print("Percentage of failures: "+str(round(100*no_failures/no_total,3)))
n = [i for i in range(5,no_faces+1)]
plt.figure()
plt.plot(n, means, marker='o', label="Mean")
plt.fill_between(
    n,
    means - stds,
    means + stds,
    alpha=0.3,
    label="±1 std"
)

plt.xlabel("Number of faces")
plt.ylabel("Time taken (mus)")
plt.legend()
plt.grid(True)
plt.show()
