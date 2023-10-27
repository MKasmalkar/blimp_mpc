import numpy as np
import scipy
from rta.blimp import Blimp
from BlimpTrackingCasadi import BlimpTrackingCasadi
import time
import csv
import sys

if len(sys.argv) < 3:
    print("Please run with simulation time as first argument and output data file as second")

dT = 0.25

my_blimp = Blimp()
A = my_blimp.jacobian_np(np.zeros((12, 1)))
B = my_blimp.B

A_dis = scipy.linalg.expm(A) #np.eye(12) + dT * A
B_int = np.zeros((12,12))
for i in range(10000):
    dTau = dT / 10000
    tau = i * dTau
    B_int += scipy.linalg.expm(A * tau) * dTau
B_dis = B_int @ B #np.linalg.inv(A) @ (A_dis - np.eye(12)) @ B

C = np.matrix([[0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
               [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
               [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0]])
D = np.zeros((3, 4))

P = np.identity(3) * 10
Q = np.identity(3) * 10
R = np.matrix([1]) * 10

out_file = sys.argv[2]

TIME_STOP = int(sys.argv[1])
time_vec = np.arange(0, TIME_STOP, dT)

N = 10
blimp_controller = BlimpTrackingCasadi(A_dis,
                                       B_dis,
                                       C,
                                       D,
                                       P,
                                       Q,
                                       R,
                                       N)

# Initial state 

y0 = 0
y1 = 0
y2 = 0

x = np.matrix([[0],
               [0],
               [0],
               [0],
               [0],
               [0],
               [y0],   # x
               [y1],   # y
               [y2],   # z
               [0],
               [0],
               [0],])
y = C @ x

ref_idx = 0
reference_points = [
                    np.matrix([5, 5, 5]).T,
                    np.matrix([5, -5, -5]).T,
                    np.matrix([-5, -5, 2.5]).T,
                    np.matrix([-5, 5, -2.5]).T
                    ]
NUM_REF_PTS = 4

y0_vals = []
y1_vals = []
y2_vals = []

u0_vals = []
u1_vals = []
u2_vals = []
u3_vals = []

ref0_vals = []
ref1_vals = []
ref2_vals = []

error_vals = []

def distance_to_goal(state, goal):
    return np.linalg.norm(state - goal)

xmin = np.matrix([[-np.inf],
                  [-np.inf],
                  [-np.inf],
                  [-np.inf],
                  [-np.inf],
                  [-np.inf],
                  [-10],   # x
                  [-10],   # y
                  [-10],   # z
                  [-np.inf],
                  [-np.inf],
                  [-np.inf]
                  ])

xmax = np.matrix([[np.inf],
                  [np.inf],
                  [np.inf],
                  [np.inf],
                  [np.inf],
                  [np.inf],
                  [10],   # x
                  [10],   # y
                  [10],   # z
                  [np.inf],
                  [np.inf],
                  [np.inf]
                  ])

umin = np.matrix([[-0.005],
                  [-0.005],
                  [-0.005],
                  [-np.inf]])

umax = np.matrix([[0.005],
                  [0.005],
                  [0.005],
                  [np.inf]])

DEADBAND = 1

TIMESTEPS_TO_SETTLE = 10
settling_timer = TIMESTEPS_TO_SETTLE

# i = 0

# go = False

times = []

for t in time_vec:

    print()
    print("Time t=" + str(t))

    y0_vals.append(float(y[0].item()))
    y1_vals.append(float(y[1].item()))
    y2_vals.append(float(y[2].item()))

    ref0_vals.append(float(reference_points[ref_idx][0].item()))
    ref1_vals.append(float(reference_points[ref_idx][1].item()))
    ref2_vals.append(float(reference_points[ref_idx][2].item()))

    start_time = time.time()
    u = blimp_controller.get_tracking_ctrl(x,
                                           reference_points[ref_idx],
                                           xmin,
                                           xmax,
                                           umin,
                                           umax)
    times.append(time.time() - start_time)

    u0_vals.append(float(u[0].item()))
    u1_vals.append(float(u[1].item()))
    u2_vals.append(float(u[2].item()))
    u3_vals.append(float(u[3].item()))

    #print("Old outputs: " + str(y.T))
    #print("Input: " + str(u.T))
    
    x = A_dis @ x + B_dis @ u
    y = C @ x + D @ u

    #print("New outputs: " + str(y.T))

    error = distance_to_goal(y, reference_points[ref_idx])
    error_vals.append(error)
    #print("Error: " + str(error))

    if error < DEADBAND:
        settling_timer -= 1
        if settling_timer == 0:
            settling_timer = TIMESTEPS_TO_SETTLE
            ref_idx = (ref_idx + 1) % NUM_REF_PTS
            #print()
            #print("UPDATING REFERENCE ==========================================================================")
            #print("UPDATING REFERENCE ==========================================================================")
    else:
        settling_timer = TIMESTEPS_TO_SETTLE

with open('outputs/' + out_file, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile, delimiter=',')


    print("Writing to CSV...")
    writer.writerow(['Time',
                     'X', 'Y', 'Z',
                     'Reference X', 'Reference Y', 'Reference Z',
                     'Error',
                     'u0', 'u1', 'u2', 'u3',
                     'deltaT'])
    
    for i in range(len(time_vec)):
        #print("Row " + str(i) + " out of " + str(len(time_vec)))
        writer.writerow([time_vec[i],
                         y0_vals[i], y1_vals[i], y2_vals[i],
                         ref0_vals[i], ref1_vals[i], ref2_vals[i],
                         error_vals[i],
                         u0_vals[i], u1_vals[i], u2_vals[i], u3_vals[i],
                         times[i]])
        
    print("Done writing to CSV!")
        