import numpy as np
import scipy
import matplotlib.pyplot as plt
from rta.blimp import Blimp
import random
from TrajTrackingGurobi import TrajTrackingGurobi
import traceback
import sys
import csv
from parameters import *
import time

np.set_printoptions(suppress=True)

dT = 0.05

# Time
TRACKING_TIME = 20
SETTLE_TIME = 100

tracking_time = np.arange(0, TRACKING_TIME, dT)
settle_time = np.arange(TRACKING_TIME, TRACKING_TIME + SETTLE_TIME, dT)

time_vec = np.concatenate((tracking_time, settle_time))

# Trajectory definition
f = 0.05
At = 1

x0 = At
y0 = 0
z0 = 0

phi0 = 0
theta0 = 0
psi0 = np.pi/2

v_x0 = 0.0
v_y0 = 0.0
v_z0 = 0.0

w_x0 = 0
w_y0 = 0
w_z0 = 0

traj_x = np.concatenate((At * np.cos(2*np.pi*f*tracking_time), At*np.ones(len(settle_time))))
traj_y = np.concatenate((At * np.sin(2*np.pi*f*tracking_time), np.zeros(len(settle_time))))
traj_z = np.concatenate((tracking_time * -1/10, -TRACKING_TIME * 1/10 * np.ones(len(settle_time))))
traj_psi = np.concatenate((psi0 + 2*np.pi*f*tracking_time, (psi0 + 2*np.pi) * np.ones(len(settle_time))))

traj_x_dot = np.concatenate((-2*np.pi*f*At*np.sin(2*np.pi*f*tracking_time), np.zeros(len(settle_time))))
traj_y_dot = np.concatenate((2*np.pi*f*At*np.cos(2*np.pi*f*tracking_time), np.zeros(len(settle_time))))
traj_z_dot = np.concatenate((-1/10 * np.ones(len(tracking_time)), np.zeros(len(settle_time))))
traj_psi_dot = np.concatenate((2*np.pi*f * np.ones(len(tracking_time)), np.zeros(len(settle_time))))

traj_x_ddot = np.concatenate((-(2*np.pi*f)**2*At*np.cos(2*np.pi*f*tracking_time), np.zeros(len(settle_time))))
traj_y_ddot = np.concatenate((-(2*np.pi*f)**2*At*np.sin(2*np.pi*f*tracking_time), np.zeros(len(settle_time))))
traj_z_ddot = np.concatenate((np.zeros(len(tracking_time)), np.zeros(len(settle_time))))
traj_psi_ddot = np.concatenate((np.zeros(len(tracking_time)), np.zeros(len(settle_time))))

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

Cmat = np.matrix([[0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                  [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]])

D = np.zeros((8, 4))

P = np.identity(8)
Q = np.identity(8)
R = np.identity(4)

xmin = np.matrix([[-np.inf],
                  [-np.inf],
                  [-np.inf],
                  [-np.inf],
                  [-np.inf],
                  [-np.inf],
                  [-np.inf],   # x
                  [-np.inf],   # y
                  [-np.inf],   # z
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
                  [np.inf],   # x
                  [np.inf],   # y
                  [np.inf],   # z
                  [np.inf],
                  [np.inf],
                  [np.inf]
                  ])

umin = np.matrix([[-np.inf],
                  [-np.inf],
                  [-np.inf],
                  [-np.inf]])

umax = np.matrix([[np.inf],
                  [np.inf],
                  [np.inf],
                  [np.inf]])

time_horizon = 10
blimp_controller = TrajTrackingGurobi(A_dis,
                                      B_dis,
                                      Cmat,
                                      D,
                                      P,
                                      Q,
                                      R,
                                      time_horizon,
                                      xmin,
                                      xmax,
                                      umin,
                                      umax)

N = 12

# State vector
eta_bn_n = np.array([[x0, y0, z0, phi0, theta0, psi0]]).T
nu_bn_b = np.array([[v_x0, v_y0, v_z0, w_x0, w_y0, w_z0]]).T

state = np.empty((N, len(time_vec)))
state_l = np.empty((N, len(time_vec)))
state_nl = np.empty((N, len(time_vec)))

state_dot = np.empty((N, len(time_vec)))
state_dot_l = np.empty((N, len(time_vec)))
state_dot_nl = np.empty((N, len(time_vec)))

state[:, 0] = np.vstack((eta_bn_n, nu_bn_b)).reshape(N)

state_nl[:, 0] = np.vstack((eta_bn_n, nu_bn_b)).reshape(N)
state_l[:, 0] = np.vstack((eta_bn_n, nu_bn_b)).reshape(N)
state_dot[:, 0] = np.zeros(N)
state_dot_l[:, 0] = np.zeros(N)
state_dot_nl[:, 0] = np.zeros(N)

u_log = np.empty((4, len(time_vec)))

error = np.empty((2, 4, len(time_vec)))

## Set up figure

fig = plt.figure()
plt.ion()

ax_3d = fig.add_subplot(211, projection='3d')
ax_3d.grid()

ax_err = fig.add_subplot(323)
ax_zd = fig.add_subplot(324)
ax_v = fig.add_subplot(325)
ax_w = fig.add_subplot(326)

plt.subplots_adjust(wspace=0.5)
plt.subplots_adjust(hspace=1)

try:
    for n in range(len(time_vec)):
        
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

        error[0, 0, n] = x - traj_x[n]
        error[0, 1, n] = y - traj_y[n]
        error[0, 2, n] = z - traj_z[n]
        error[0, 3, n] = psi - traj_psi[n]

        modified_state = np.array([[v_x__b],
                                   [v_y__b],
                                   [v_z__b],
                                   [w_x__b],
                                   [w_y__b],
                                   [w_z__b],
                                   [x],
                                   [y],
                                   [z],
                                   [phi],
                                   [theta],
                                   [psi]])
        
        v_ref_n = np.array([traj_x_dot[n:n+time_horizon],
                          traj_y_dot[n:n+time_horizon],
                          traj_z_dot[n:n+time_horizon]])
        v_ref_b = R_b__n_inv(phi, theta, psi) @ v_ref_n

        reference = np.array([traj_x[n:n+time_horizon],
                              traj_y[n:n+time_horizon],
                              traj_z[n:n+time_horizon],
                              traj_psi[n:n+time_horizon],
                              v_ref_b[0, :],
                              v_ref_b[1, :],
                              v_ref_b[2, :],
                              traj_psi_dot[n:n+time_horizon]]).T

        u = blimp_controller.get_tracking_ctrl(modified_state,
                                               reference)

        # State vector

        tau_B = np.array([u[0],
                            u[1],
                            u[2],
                            -r_z_tg__b * u[1],
                            r_z_tg__b * u[0],
                            u[3]]).reshape((6,1))
        
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

        # print(np.round(eta_bn_n.reshape(6), 3))

        state_nl[:, n+1] = np.vstack((eta_bn_n, nu_bn_b)).reshape(N)
        
        modified_state_new = np.asarray(A_dis @ modified_state + B_dis @ u)
        state_l[:, n+1] = np.array([[modified_state_new[6]],
                                  [modified_state_new[7]],
                                  [modified_state_new[8]],
                                  [modified_state_new[9]],
                                  [modified_state_new[10]],
                                  [modified_state_new[11]],
                                  [modified_state_new[0]],
                                  [modified_state_new[1]],
                                  [modified_state_new[2]],
                                  [modified_state_new[3]],
                                  [modified_state_new[4]],
                                  [modified_state_new[5]]]).reshape(12)
        
        state_dot_lin = A @ state_l[:,n] + B @ u
        
        state[:, n+1] = state_l[:, n+1]

        state_dot_l[:, n+1] = ((state_l[:, n+1] - state_l[:, n]) / dT).reshape(N)
        state_dot_nl[:, n+1] = np.vstack((eta_bn_n_dot, nu_bn_b_dot)).reshape(12)

        state_dot[:, n+1] = ((state[:, n+1] - state[:, n]) / dT).reshape(N)
        
        # print("State (linear):")
        # print(np.round(state_l[:, n+1].reshape(12), 3))
        # print("State (nonlinear):")
        # print(np.round(state_nl[:, n+1].reshape(12), 3))
        print("State dot (discrete):")
        print(np.round(state_dot_l[:, n+1].reshape(12), 3))
        print("State dot (nonlinear):")
        print(np.round(state_dot_nl[:, n+1].reshape(12), 3))
        print("State dot (used):")
        print(np.round(state_dot[:, n+1].reshape(12), 3))
        print("State dot (linear):")
        print(np.round(state_dot_lin[:, n+1].reshape(12), 3))
        input()

        ax_3d.cla()
        ax_3d.scatter(state[0, 0:n], state[1, 0:n], state[2, 0:n], color='blue', s=100)
        ax_3d.scatter(state[0, n], state[1, n], state[2, n], color='m', s=200)
        ax_3d.scatter(traj_x, traj_y, traj_z, color='g')
        ax_3d.invert_yaxis()
        ax_3d.invert_zaxis()

        x_span = ax_3d.get_xlim()[1] - ax_3d.get_xlim()[0]
        y_span = ax_3d.get_ylim()[1] - ax_3d.get_ylim()[0]
        z_span = ax_3d.get_zlim()[1] - ax_3d.get_zlim()[0]

        blimp_vector_scaling = abs(min(x_span,
                                            y_span,
                                            z_span)) * 0.2
        
        blimp_x_vector = R_b__n(eta_bn_n[3], eta_bn_n[4], eta_bn_n[5]) \
                        @ np.array([blimp_vector_scaling, 0, 0]).T
        blimp_y_vector = R_b__n(eta_bn_n[3], eta_bn_n[4], eta_bn_n[5]) \
                        @ np.array([0, blimp_vector_scaling, 0]).T
        blimp_z_vector = R_b__n(eta_bn_n[3], eta_bn_n[4], eta_bn_n[5]) \
                        @ np.array([0, 0, blimp_vector_scaling]).T

        qx = ax_3d.quiver(eta_bn_n[0], eta_bn_n[1], eta_bn_n[2], \
                blimp_x_vector[0], blimp_x_vector[1], blimp_x_vector[2], \
                color='r')
        qx.ShowArrowHead = 'on'
        qy = ax_3d.quiver(eta_bn_n[0], eta_bn_n[1], eta_bn_n[2], \
                blimp_y_vector[0], blimp_y_vector[1], blimp_y_vector[2], \
                color='g')
        qy.ShowArrowHead = 'on'
        qz = ax_3d.quiver(eta_bn_n[0], eta_bn_n[1], eta_bn_n[2], \
                blimp_z_vector[0], blimp_z_vector[1], blimp_z_vector[2], \
                color='b')
        qz.ShowArrowHead = 'on'

        ax_3d.set_xlabel('x')
        ax_3d.set_ylabel('y')
        ax_3d.set_zlabel('z')
        ax_3d.set_title('Trajectory')

        ax_err.cla()
        ax_err.plot(time_vec[0:n], error[0, 0, 0:n])
        ax_err.plot(time_vec[0:n], error[0, 1, 0:n])
        ax_err.plot(time_vec[0:n], error[0, 2, 0:n])
        ax_err.plot(time_vec[0:n], error[0, 3, 0:n])
        ax_err.set_xlabel('Time')
        ax_err.set_ylabel('Error')
        ax_err.legend(['x', 'y', 'z', 'psi'])
        ax_err.set_title('Tracking Error')
        
        ax_zd.cla()
        ax_zd.plot(time_vec[0:n], state[3, 0:n] * 180/np.pi)
        ax_zd.plot(time_vec[0:n], state[4, 0:n] * 180/np.pi)
        ax_zd.legend(['phi', 'theta'])
        ax_zd.set_title('Pitch/roll')
        ax_zd.set_xlabel('Time')
        ax_zd.set_ylabel('Degrees')

        ax_v.cla()
        ax_v.plot(time_vec[0:n], state[6, 0:n])
        ax_v.plot(time_vec[0:n], state[7, 0:n])
        ax_v.plot(time_vec[0:n], state[8, 0:n])
        ax_v.legend(['vx', 'vy', 'vz'])
        ax_v.set_title('Velocity')
        ax_v.set_xlabel('Time')
        ax_v.set_ylabel('m/s')

        ax_w.cla()
        ax_w.plot(time_vec[0:n], state[9, 0:n])
        ax_w.plot(time_vec[0:n], state[10, 0:n])
        ax_w.plot(time_vec[0:n], state[11, 0:n])
        ax_w.legend(['wx', 'wy', 'wz'])
        ax_w.set_title('Omega')
        ax_w.set_xlabel('Time')
        ax_w.set_ylabel('rad/s')

        plt.draw()
        plt.pause(0.00000000001)
    
except KeyboardInterrupt:
    plt.draw()
    plt.pause(0.01)
    plt.show(block=True)

except Exception as ex:
    print(traceback.format_exc())
    
finally:
    with open('logs/' + sys.argv[1], 'w', newline='') as outfile:
        writer = csv.writer(outfile)

        writer.writerow(['time',
                         'x', 'y', 'z', 'phi', 'theta', 'psi', 'vx', 'vy', 'vz', 'wx', 'wy', 'wz',
                         'xdot', 'ydot', 'zdot', 'phidot', 'thetadot', 'psidot', 'vxdot', 'vydot', 'vzdot', 'wxdot', 'wydot', 'wzdot',
                         'fx', 'fy', 'fz', 'tauz',
                         'x_error', 'y_error', 'z_error', 'psi_error'])
    
        time_history = time_vec.reshape((2400, 1))
        state_history = state.T
        state_dot_history = state_dot.T
        u_history = u_log.T
        error_history = error[0, :].T

        data = np.hstack((time_history,
                          state_history,
                          state_dot_history,
                          u_history,
                          error_history))
        writer.writerows(data)

    plt.show(block=True)