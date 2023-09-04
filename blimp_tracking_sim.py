import numpy as np
import scipy
import matplotlib.pyplot as plt
from rta.blimp import Blimp
import random
from BlimpTrackingMPC import BlimpTrackingMPC

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

TIME_STOP = 250
time_vec = np.arange(0, TIME_STOP, dT)

N = 10
blimp_controller = BlimpTrackingMPC(A_dis,
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


fig1 = plt.figure()
plt.ion()
ax = fig1.add_subplot(111, projection='3d')
ax.set_xlabel('y0')
ax.set_ylabel('y1')
ax.set_zlabel('y2')

# i = 0

# go = False

for t in time_vec:
    
    print()
    print("Time t=" + str(t))

    y0_vals.append(float(y[0]))
    y1_vals.append(float(y[1]))
    y2_vals.append(float(y[2]))

    ref0_vals.append(float(reference_points[ref_idx][0]))
    ref1_vals.append(float(reference_points[ref_idx][1]))
    ref2_vals.append(float(reference_points[ref_idx][2]))

    ax.cla()
    ax.scatter(y0_vals, y1_vals, y2_vals, color='b')
    ax.scatter(ref0_vals, ref1_vals, ref2_vals, color='r', s=100)
    ax.scatter(y0, y1, y2, color='m', s=100)
    ax.scatter(y[0], y[1], y[2], color='c', s=75)
    ax.set_xlabel("y0")
    ax.set_ylabel('y1')
    ax.set_zlabel('y2')
    plt.draw()
    plt.pause(0.001)
    
    # if not go:
    #     input("Press enter to start...")
    #     go = True

    u = blimp_controller.get_tracking_ctrl(x,
                                           reference_points[ref_idx],
                                           xmin,
                                           xmax,
                                           umin,
                                           umax)
    
    # if i < len(time_vec) / 2:
    #     u = np.matrix([[-0.005],
    #                 [-0.005],
    #                 [-0.005],
    #                 [0]])
    # else:
    #     u = np.matrix([[0.005],
    #                 [0.005],
    #                 [0.005],
    #                 [0]])

    # i += 1

    u0_vals.append(float(u[0]))
    u1_vals.append(float(u[1]))
    u2_vals.append(float(u[2]))
    u3_vals.append(float(u[3]))

    # print("Old outputs: " + str(y.T))
    # print("Input: " + str(u.T))
    
    x = A_dis @ x + B_dis @ u
    y = C @ x + D @ u

    # print("New outputs: " + str(y.T))

    error = distance_to_goal(y, reference_points[ref_idx])
    error_vals.append(error)
    print("Error: " + str(error))

    if error < DEADBAND:
        settling_timer -= 1
        if settling_timer == 0:
            settling_timer = TIMESTEPS_TO_SETTLE
            ref_idx = (ref_idx + 1) % NUM_REF_PTS
            print()
            print("UPDATING REFERENCE ==========================================================================")
            print("UPDATING REFERENCE ==========================================================================")
    else:
        settling_timer = TIMESTEPS_TO_SETTLE

plt.show(block=True)

# fig2 = plt.figure()

# plt.subplot(4, 1, 1)
# plt.plot(time_vec, u0_vals)
# plt.plot(time_vec, u1_vals)
# plt.plot(time_vec, u2_vals)
# plt.plot(time_vec, u3_vals)
# plt.legend(['u0', 'u1', 'u2', 'u3'])
# plt.xlabel('Time')
# plt.title("Input amplitudes")

# plt.subplot(4, 1, 2)
# plt.plot(time_vec, y0_vals)
# plt.plot(time_vec, y1_vals)
# plt.plot(time_vec, y2_vals)
# plt.xlabel('Time')
# plt.legend(['y0', 'y1', 'y2'])
# plt.title("States")

# plt.subplot(4, 1, 3)
# plt.plot(time_vec, ref0_vals)
# plt.plot(time_vec, ref1_vals)
# plt.plot(time_vec, ref2_vals)
# plt.xlabel('Time')
# plt.legend(['y0 ref', 'y1 ref', 'y2 ref'])
# plt.title("Setpoints")

# plt.subplot(4, 1, 4)
# plt.plot(time_vec, error_vals)
# plt.xlabel('Time')
# plt.ylabel('Error')
# plt.title('Error')

# plt.subplots_adjust(hspace=0.75)

# plt.show()