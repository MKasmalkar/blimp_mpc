import numpy as np
import scipy
import matplotlib.pyplot as plt
from rta.blimp import Blimp
import random

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

TIME_STOP = 30
time_vec = np.arange(0, TIME_STOP, dT)

x = np.zeros((12, 1))
u = np.matrix([[random.uniform(-1, 1)],
               [random.uniform(-1, 1)],
               [random.uniform(-1, 1)],
               [random.uniform(-1, 1)]])

y0_vals = []
y1_vals = []
y2_vals = []

for t in time_vec:
    x = A_dis @ x + B_dis @ u
    y = C @ x + D @ u

    y0_vals.append(float(y[0]))
    y1_vals.append(float(y[1]))
    y2_vals.append(float(y[2]))

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(y0_vals, y1_vals, y2_vals)
plt.show()