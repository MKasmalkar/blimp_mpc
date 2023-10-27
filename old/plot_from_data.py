import numpy as np
import matplotlib.pyplot as plt
import control
from rta.blimp import Blimp
import scipy
import sys
import traceback
import csv
from operators import *
from parameters import *
from controllers import *

if len(sys.argv) < 2:
    print("Please run with input file as argument")
    sys.exit(0)

with open('logs/' + sys.argv[1], 'r') as infile:
    reader = csv.reader(infile)
    data_list = list(reader)[1:]
    data_float = [[float(i) for i in j] for j in data_list]
    data_T = np.array(data_float)
    data = data_T.T

    time_vec = data[0, :]
    state = data[1:13, :]
    state_dot = data[13:25, :]
    u = data[25:29, :]
    error = data[29:33, :]

    n = len(time_vec)-1

    ## Set up figure

    fig = plt.figure()
    plt.ion()

    # ax_3d = fig.add_subplot(211, projection='3d')
    # ax_3d.grid()

    ax_err = fig.add_subplot(221)
    ax_zd = fig.add_subplot(222)
    ax_v = fig.add_subplot(223)
    ax_w = fig.add_subplot(224)

    plt.subplots_adjust(hspace=0.5)
    plt.subplots_adjust(wspace=0.5)

    # ax_3d.cla()
    # ax_3d.scatter(state[0, 0:n], state[1, 0:n], state[2, 0:n], color='blue', s=100)
    # ax_3d.scatter(state[0, n], state[1, n], state[2, n], color='m', s=200)
    # ax_3d.invert_yaxis()
    # ax_3d.invert_zaxis()

    # x_span = ax_3d.get_xlim()[1] - ax_3d.get_xlim()[0]
    # y_span = ax_3d.get_ylim()[1] - ax_3d.get_ylim()[0]
    # z_span = ax_3d.get_zlim()[1] - ax_3d.get_zlim()[0]

    # blimp_vector_scaling = abs(min(x_span,
    #                                     y_span,
    #                                     z_span)) * 0.2

    # blimp_x_vector = R_b__n(state[3, n], state[4, n], state[5, n]) \
    #                 @ np.array([blimp_vector_scaling, 0, 0]).T
    # blimp_y_vector = R_b__n(state[3, n], state[4, n], state[5, n]) \
    #                 @ np.array([0, blimp_vector_scaling, 0]).T
    # blimp_z_vector = R_b__n(state[3, n], state[4, n], state[5, n]) \
    #                 @ np.array([0, 0, blimp_vector_scaling]).T

    # qx = ax_3d.quiver(state[0, n], state[1, n], state[2, n], \
    #         blimp_x_vector[0], blimp_x_vector[1], blimp_x_vector[2], \
    #         color='r')
    # qx.ShowArrowHead = 'on'
    # qy = ax_3d.quiver(state[0, n], state[1, n], state[2, n], \
    #         blimp_y_vector[0], blimp_y_vector[1], blimp_y_vector[2], \
    #         color='g')
    # qy.ShowArrowHead = 'on'
    # qz = ax_3d.quiver(state[0, n], state[1, n], state[2, n], \
    #         blimp_z_vector[0], blimp_z_vector[1], blimp_z_vector[2], \
    #         color='b')
    # qz.ShowArrowHead = 'on'

    # ax_3d.set_xlabel('x')
    # ax_3d.set_ylabel('y')
    # ax_3d.set_zlabel('z')
    # ax_3d.set_title('Trajectory')

    ax_err.cla()
    ax_err.plot(time_vec[0:n], error[0, 0:n])
    ax_err.plot(time_vec[0:n], error[1, 0:n])
    ax_err.plot(time_vec[0:n], error[2, 0:n])
    ax_err.plot(time_vec[0:n], error[3, 0:n])
    ax_err.set_xlabel('Time (sec)')
    ax_err.set_ylabel('Error')
    ax_err.legend(['x', 'y', 'z', 'psi'])
    ax_err.set_title('Tracking Error')
    ax_err.set_yticks(np.arange(-0.3, 0.4, 0.1))
    ax_err.set_xticks(np.arange(0, 135, 15))

    ax_zd.cla()
    ax_zd.plot(time_vec[0:n], state[3, 0:n] * 180/np.pi)
    ax_zd.plot(time_vec[0:n], state[4, 0:n] * 180/np.pi)
    ax_zd.legend(['phi', 'theta'])
    ax_zd.set_title('Pitch/roll')
    ax_zd.set_xlabel('Time (sec)')
    ax_zd.set_ylabel('Degrees')
    ax_zd.set_yticks(np.arange(-16, 16, 4))
    ax_zd.set_xticks(np.arange(0, 135, 15))

    ax_v.cla()
    ax_v.plot(time_vec[0:n], state[6, 0:n])
    ax_v.plot(time_vec[0:n], state[7, 0:n])
    ax_v.plot(time_vec[0:n], state[8, 0:n])
    ax_v.legend(['vx', 'vy', 'vz'])
    ax_v.set_title('Velocity')
    ax_v.set_xlabel('Time (sec)')
    ax_v.set_ylabel('m/s')
    ax_v.set_yticks(np.arange(-0.6, 0.4, 0.1))
    ax_v.set_xticks(np.arange(0, 135, 15))

    ax_w.cla()
    ax_w.plot(time_vec[0:n], state[9, 0:n])
    ax_w.plot(time_vec[0:n], state[10, 0:n])
    ax_w.plot(time_vec[0:n], state[11, 0:n])
    ax_w.set_yticks(np.arange(-0.1, 0.5, 0.1))
    ax_w.legend(['wx', 'wy', 'wz'])
    ax_w.set_title('Omega')
    ax_w.set_xlabel('Time (sec)')
    ax_w.set_ylabel('rad/s')
    ax_w.set_yticks(np.arange(-0.1, 0.6, 0.1))
    ax_w.set_xticks(np.arange(0, 135, 15))

    plt.draw()
    plt.show(block=True)