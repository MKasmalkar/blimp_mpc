from CasadiTrackingMPC import CasadiTrackingMPC
from ModelPredictiveController import ModelPredictiveController
import numpy as np
import matplotlib.pyplot as plt
import time

print("MPC SYSTEM")
print()

A, B, C, D, P, Q, R = ModelPredictiveController.load_dynamics('dynamics')
controller = CasadiTrackingMPC('dynamics', 3)

x = np.matrix([[10],
               [-10]])

TIME_STEP = 0.1
TIME_STOP = 20

DATA_SIZE = 8

time_vec = np.arange(0, TIME_STOP, TIME_STEP)
x0 = []
x1 = []
u_history = []

y_ref = 2 * np.exp(-0.15*time_vec) * np.sin(time_vec) * time_vec - 2

try:

    start_time = time.time()
    idx = 0
    for t in time_vec:
        print("Time t=" + str(t))

        x0.append(float(x[0]))
        x1.append(float(x[1]))
        
        x_old = x.copy()

        u = float(controller.get_tracking_ctrl(x, y_ref[idx]))
        u_history.append(u)

        print("Old state: " + str(x_old.T))
        print("Input: " + str(u))
        x = (A @ x) + (B * u)
        print("New state: " + str(x.T))
        print()

        idx = idx + 1

    print("Time: " + str(time.time() - start_time))

    plt.subplot(3, 1, 1)
    plt.plot(x0, x1)
    plt.xlabel('State x0')
    plt.ylabel('State x1')

    plt.subplot(3, 1, 2)
    plt.plot(time_vec, x0)
    plt.plot(time_vec, x1)
    plt.plot(time_vec, y_ref)
    plt.xlabel('Time')
    plt.legend(['x0', 'x1', 'y ref'])

    plt.subplot(3, 1, 3)
    plt.plot(time_vec, u_history)
    plt.xlabel('Time')
    plt.ylabel('Control input u')

    plt.subplots_adjust(hspace=0.4)

    plt.show()

except KeyboardInterrupt:
    print("Done!")