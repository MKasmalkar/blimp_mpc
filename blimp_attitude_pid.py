import numpy as np
import matplotlib.pyplot as plt
import control
from rta.blimp import Blimp
import scipy
import sys
from operators import *
from parameters import *
from controllers import *

## Constants
N = 12
dT = 0.05

## Simulation

my_blimp = Blimp()
B_lin = my_blimp.B

# Time
TRACKING_TIME = 20
SETTLE_TIME = 100

tracking_time = np.arange(0, TRACKING_TIME, dT)
settle_time = np.arange(TRACKING_TIME, TRACKING_TIME + SETTLE_TIME, dT)

time_vec = np.concatenate((tracking_time, settle_time))

# Initial conditions
x0 = 0
y0 = 0
z0 = 0

phi0 = 0
theta0 = 10*np.pi/180
psi0 = 0

v_x0 = 0.0
v_y0 = 0.0
v_z0 = 0.0

w_x0 = 0
w_y0 = 0
w_z0 = 0

# State vector
eta_bn_n = np.array([[x0, y0, z0, phi0, theta0, psi0]]).T
nu_bn_b = np.array([[v_x0, v_y0, v_z0, w_x0, w_y0, w_z0]]).T

state = np.empty((N, len(time_vec)))
state_dot = np.empty((N, len(time_vec)))

state[:, 0] = np.vstack((eta_bn_n, nu_bn_b)).reshape(N)
state_dot[:, 0] = np.zeros(N)

fx_history = np.empty(len(time_vec))
fz_history = np.empty(len(time_vec))

# Controls
ctrl = NestedPID(dT)

# Set up figure

fig = plt.figure()
ax_3d = fig.add_subplot(322, projection='3d')
plt.ion()
ax_3d.grid()

ax_or = fig.add_subplot(321, projection='3d')
ax_or.grid()

ax_pos = fig.add_subplot(324)
ax_att = fig.add_subplot(323)

ax_v = fig.add_subplot(325)

ax_forces = fig.add_subplot(326)

plt.subplots_adjust(hspace=1.0)

try:
    for n in range(len(time_vec) - 1):
        # print()
        # print("Time: " + str(time_vec[n]))

        # Extract state variables
        x = state[0, n]
        y = state[1, n]
        z = state[2, n]

        phi = state[3, n]
        theta = state[4, n]
        psi = state[5, n]
        
        v_x__b = state[6, n]
        v_y__b = state[7, n]
        v_z__b = state[8, n]

        w_x__b = state[9, n]
        w_y__b = state[10, n]
        w_z__b = state[11, n]

        x_dot = state_dot[0, n]
        y_dot = state_dot[1, n]
        z_dot = state_dot[2, n]

        phi_dot = state_dot[3, n]
        theta_dot = state_dot[4, n]
        psi_dot = state_dot[5, n]
        
        u = ctrl.get_ctrl_input(state[:, n].reshape((12,1)))
        fx_history[n] = u[0]
        fz_history[n] = u[2]

        tau_B = np.array([u[0], u[1], u[2], -r_z_tg__b * u[1], r_z_tg__b * u[0], u[3]])

        # Restoration torque
        fg_B = R_b__n_inv(phi, theta, psi) @ fg_n
        g_CB = -np.block([[np.zeros((3, 1))],
                        [np.reshape(np.cross(r_gb__b, fg_B), (3, 1))]])

        # Update state
        eta_bn_n_dot = np.block([[R_b__n(phi, theta, psi),    np.zeros((3, 3))],
                               [np.zeros((3, 3)),            T(phi, theta)]]) @ nu_bn_b
        
        nu_bn_b_dot = np.reshape(-M_CB_inv @ (C(M_CB, nu_bn_b) @ nu_bn_b + \
                           D_CB @ nu_bn_b + g_CB - tau_B), (6, 1))

        eta_bn_n = eta_bn_n + eta_bn_n_dot * dT
        nu_bn_b = nu_bn_b + nu_bn_b_dot * dT

        state[:, n+1] = np.vstack((eta_bn_n, nu_bn_b)).reshape(N)
        state_dot[:, n+1] = ((state[:, n+1] - state[:, n]) / dT).reshape(N)
        
        ax_3d.cla()
        ax_3d.scatter(state[0, 0:n], state[1, 0:n], state[2, 0:n], color='blue', s=100)
        ax_3d.scatter(state[0, n], state[1, n], state[2, n], color='m', s=200)
        ax_3d.invert_yaxis()
        ax_3d.invert_zaxis()

        x_span = ax_3d.get_xlim()[1] - ax_3d.get_xlim()[0]
        y_span = ax_3d.get_ylim()[1] - ax_3d.get_ylim()[0]
        z_span = ax_3d.get_zlim()[1] - ax_3d.get_zlim()[0]

        ax_v.cla()
        ax_v.plot(time_vec[0:n], state[9, 0:n])
        ax_v.plot(time_vec[0:n], state[10, 0:n])
        ax_v.plot(time_vec[0:n], state[11, 0:n])
        ax_v.set_xlabel('Time')
        ax_v.legend(['w_x', 'w_y', 'w_z'])

        ax_or.cla()
        blimp_vector_scaling = 2
        
        blimp_x_vector = R_b__n(eta_bn_n[3], eta_bn_n[4], eta_bn_n[5]) \
                        @ np.array([blimp_vector_scaling, 0, 0]).T
        blimp_y_vector = R_b__n(eta_bn_n[3], eta_bn_n[4], eta_bn_n[5]) \
                        @ np.array([0, blimp_vector_scaling, 0]).T
        blimp_z_vector = R_b__n(eta_bn_n[3], eta_bn_n[4], eta_bn_n[5]) \
                        @ np.array([0, 0, blimp_vector_scaling]).T
    
        qx = ax_or.quiver(0, 0, 0, \
                blimp_x_vector[0], blimp_x_vector[1], blimp_x_vector[2], \
                color='r')
        qx.ShowArrowHead = 'on'
        qy = ax_or.quiver(0, 0, 0, \
                blimp_y_vector[0], blimp_y_vector[1], blimp_y_vector[2], \
                color='g')
        qy.ShowArrowHead = 'on'
        qz = ax_or.quiver(0, 0, 0, \
                blimp_z_vector[0], blimp_z_vector[1], blimp_z_vector[2], \
                color='b')
        qz.ShowArrowHead = 'on'

        ax_or.set_xlim(-1.5, 1.5)
        ax_or.set_ylim(-1.5, 1.5)
        ax_or.set_zlim(-1.5, 1.5)
        ax_or.invert_yaxis()
        ax_or.invert_zaxis()

        ax_3d.set_xlabel('x')
        ax_3d.set_ylabel('y')
        ax_3d.set_zlabel('z')
        ax_3d.set_title('Trajectory')
        # ax_3d.set_xlim(-0.001, 0.001)
        # ax_3d.set_ylim(-0.001, 0.001)
        # ax_3d.set_zlim(-0.001, 0.001)

        ax_pos.cla()
        ax_pos.plot(time_vec[0:n], state[0, 0:n])
        ax_pos.plot(time_vec[0:n], state[1, 0:n])
        ax_pos.plot(time_vec[0:n], state[2, 0:n])
        ax_pos.legend(['x', 'y', 'z'])

        ax_att.cla()
        ax_att.plot(time_vec[0:n], state[3, 0:n] * 180/np.pi)
        ax_att.plot(time_vec[0:n], state[4, 0:n] * 180/np.pi)
        ax_att.plot(time_vec[0:n], state[5, 0:n] * 180/np.pi)
        ax_att.legend(['phi', 'theta', 'psi'])
        
        ax_forces.cla()
        ax_forces.plot(time_vec[0:n], fx_history[0:n])
        ax_forces.plot(time_vec[0:n], fz_history[0:n])
        ax_forces.legend(['fx', 'fz'])

        plt.draw()
        plt.pause(0.001)

except KeyboardInterrupt:
    plt.draw()
    plt.pause(0.01)
    plt.show(block=True)

plt.show(block=True)