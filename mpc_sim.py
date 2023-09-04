from CasadiExampleMPC import CasadiExampleMPC
from ExampleMPC import ExampleMPC
from ModelPredictiveController import ModelPredictiveController
import numpy as np
import matplotlib.pyplot as plt
import time

print("MPC SYSTEM")
print()

A, B, C, D, P, Q, R = ModelPredictiveController.load_dynamics('dynamics')
controller = CasadiExampleMPC('dynamics', 3)

x = np.matrix([[-2],
               [1]])

TIME_STEP = 0.1
TIME_STOP = 10

DATA_SIZE = 8

time_vec = np.arange(0, TIME_STOP, TIME_STEP)
x0 = []
x1 = []

try:

    start_time = time.time()

    for t in time_vec:
        # print("Time t=" + str(t))

        x0.append(float(x[0]))
        x1.append(float(x[1]))
        
        x_old = x.copy()

        u = float(controller.get_control_vector(x))

        # print("Old state: " + str(x_old.T))
        # print("Input: " + str(u))
        x = (A @ x) + (B * u)
        # print("New state: " + str(x.T))
        # print()

    print("Time: " + str(time.time() - start_time))

    plt.subplot(2, 1, 1)
    plt.plot(x0, x1)
    plt.subplot(2, 1, 2)
    plt.plot(time_vec, x0)
    plt.plot(time_vec, x1)
    plt.show()

except KeyboardInterrupt:
    print("Done!")