import socket
from ModelPredictiveController import ModelPredictiveController
import numpy as np
import matplotlib.pyplot as plt
import struct
import sys

print("MPC SYSTEM")
print()

HOST = '127.0.0.1'
PORT = 8000

A, B, C, D, P, Q, R = ModelPredictiveController.load_dynamics('dynamics')

x = np.matrix([[-4.5],
               [2]])

TIME_STEP = 0.1
TIME_STOP = 30

DATA_SIZE = 8

time = np.arange(0, TIME_STOP, TIME_STEP)
x0 = []
x1 = []

try:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect((HOST, PORT))

        for t in time:
            x0.append(float(x[0]))
            x1.append(float(x[1]))

            #print()
            
            x_old = x.copy()

            s.sendall(struct.pack('<2f', *x))
        
            data = s.recv(DATA_SIZE)
            if not data:
                print("Failed to receive data")
                sys.exit(1)

            u = np.matrix(float(struct.unpack('<1f', data)[0]))
            
            #print("Old state: " + str(x_old.T))
            #print("Input: " + str(u.T))
            x = (A @ x) + (B * u)
            #print("New state: " + str(x.T))

        plt.subplot(2, 1, 1)
        plt.plot(x0, x1)
        plt.subplot(2, 1, 2)
        plt.plot(time, x0)
        plt.plot(time, x1)
        plt.show()

except KeyboardInterrupt:
    print("Done!")