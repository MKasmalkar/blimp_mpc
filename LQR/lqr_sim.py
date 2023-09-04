import control as ct
import numpy as np
import matplotlib.pyplot as plt

# A = np.matrix([[1, 2],
#                [3, 4]])

# B = np.matrix([[2],
#                [3]])

A = np.matrix([[-.2, 0.1],
               [0.8, -0.9]])

B = np.matrix([[0],
                [1]])

Q = np.matrix([[1, 0],
               [0, 1]])

R = np.matrix([[1]])

K, S, E = ct.dlqr(A, B, Q, R)

TIME_STOP = 20
TIME_STEP = 0.01
time = np.arange(0, TIME_STOP, TIME_STEP)
idx = 0

x = np.array([[5000],
              [3000]])

x0_vals = np.empty(time.shape)
x1_vals = np.empty(time.shape)

for t in time:
    u = -np.matmul(K, x)
    x = x + (np.matmul(A, x) + B*u) * TIME_STEP

    x0_vals[idx] = int(x[0])
    x1_vals[idx] = int(x[1])
    idx = idx + 1

plt.plot(time, x0_vals, label="x0")
plt.plot(time, x1_vals, label="x1")

#plt.quiver(x0_vals[:-1],
#           x1_vals[:-1],
#           x0_vals[1:] - x0_vals[:-1],
#           x1_vals[1:] - x1_vals[:-1], scale_units='xy', angles='xy', scale=1)

plt.show()