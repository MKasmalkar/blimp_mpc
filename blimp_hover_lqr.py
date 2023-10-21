import numpy as np
import matplotlib.pyplot as plt
import control
from rta.blimp import Blimp
import scipy
import sys

##  Operators
def H(r):
    return np.block([ [np.identity(3),         S(r).T],
                      [np.zeros((3, 3)),        np.identity(3)]])

def S(r):
    return np.block([[  0,       -r[2],    r[1]],
                     [r[2],     0,      -r[0]],
                     [-r[1],    r[0],      0]])

def C(M, nu):
    dimM = np.shape(M)[0]

    M11 = M[0:int(dimM/2), 0:int(dimM/2)]
    M12 = M[0:int(dimM/2), int(dimM/2):dimM]
    M21 = M[int(dimM/2):dimM, 0:int(dimM/2)]
    M22 = M[int(dimM/2):dimM, int(dimM/2):dimM]

    dimNu = np.shape(nu)[0]
    nu1 = nu[0:int(dimNu/2)]
    nu2 = nu[int(dimNu/2):dimNu]

    return np.block([[ np.zeros((3, 3)),    -S(M11@nu1 + M12@nu2)],
                     [-S(M11@nu1 + M12@nu2), -S(M21@nu1 + M22@nu2)]])

## Rotation matrices

def R_b__n(phi, theta, psi):
    phi = phi.item()
    theta = theta.item()
    psi = psi.item()

    x_rot = np.array([[1,         0,           0],
                      [0,       np.cos(phi),   -np.sin(phi)],
                      [0,       np.sin(phi),    np.cos(phi)]])

    y_rot = np.array([[np.cos(theta),      0,        np.sin(theta)],
                      [0,         1,           0],
                      [-np.sin(theta),     0,        np.cos(theta)]])
    
    z_rot = np.array([[np.cos(psi),    -np.sin(psi),       0],
                      [np.sin(psi),     np.cos(psi),       0],
                      [0,          0,           1]])

    # World-to-body
    return z_rot @ y_rot @ x_rot

def R_b__n_inv(phi, theta, psi):
    phi = phi.item()
    theta = theta.item()
    psi = psi.item()

    return np.array([[np.cos(psi)*np.cos(theta), np.cos(theta)*np.sin(psi), -np.sin(theta)],
                     [np.cos(psi)*np.sin(phi)*np.sin(theta) - np.cos(phi)*np.sin(psi), np.cos(phi)*np.cos(psi) + np.sin(phi)*np.sin(psi)*np.sin(theta), np.cos(theta)*np.sin(phi)],
                     [np.sin(phi)*np.sin(psi) + np.cos(phi)*np.cos(psi)*np.sin(theta), np.cos(phi)*np.sin(psi)*np.sin(theta) - np.cos(psi)*np.sin(phi), np.cos(phi)*np.cos(theta)]])

def T(phi, theta):
    phi = phi.item()
    theta = theta.item()

    return np.array([[1,     np.sin(phi)*np.tan(theta),      np.cos(phi)*np.tan(theta)],
                     [0,          np.cos(phi),                   -np.sin(phi)],
                     [0,     np.sin(phi)/np.cos(theta),      np.cos(phi)/np.cos(theta)]])


## Constants
N = 12
dT = 0.05

## Zero dynamics compensation

# Linearized blimp model

my_blimp = Blimp()
A_lin = my_blimp.jacobian_np(np.zeros((12, 1)))
B_lin = my_blimp.B

A_dis = scipy.linalg.expm(A_lin) #np.eye(12) + dT * A
B_int = np.zeros((12,12))
for i in range(10000):
    dTau = dT / 10000
    tau = i * dTau
    B_int += scipy.linalg.expm(A_lin * tau) * dTau
B_dis = B_int @ B_lin #np.linalg.inv(A) @ (A_dis - np.eye(12)) @ B

## Blimp model parameters

# Center of gravity to center of buoyancy
r_z_gb__b = 0.08705
r_gb__b = np.array([0, 0, r_z_gb__b]).T
r_z_tg__b = 0.13 - r_z_gb__b

## Inertia matrix

m_Ax = 0.3566
m_Ay = m_Ax
m_Az = 0.645

I_Ax = 0.0
I_Ay = 0.0
I_Az = 0.0

M_A_CB = np.diag([m_Ax, m_Ay, m_Az, I_Ax, I_Ay, I_Az])

m_RB = 0.1049
I_RBx = 0.5821
I_RBy = I_RBx
I_RBz = I_RBx

M_RB_CG = np.diag([m_RB, m_RB, m_RB, I_RBx, I_RBy, I_RBz])

M_RB_CB = H(r_gb__b).T @ M_RB_CG @ H(r_gb__b)

M_CB = M_RB_CB + M_A_CB

M_CB_inv = np.linalg.inv(M_CB)

m_x = m_RB + m_Ax
m_y = m_RB + m_Ay
m_z = m_RB + m_Az

I_x = I_RBx + m_RB * r_z_gb__b**2 + I_Ax
I_y = I_RBy + m_RB * r_z_gb__b**2 + I_Ay
I_z = I_RBz + I_Az

g_acc = 9.8
fg_n = m_RB * np.array([0, 0, g_acc]).T
f_g = fg_n[2]

## Aerodynamic damping
D_vx__CB = 0.0115
D_vy__CB = D_vx__CB
D_vz__CB = 0.0480
D_vxy__CB = D_vx__CB

D_wx__CB = 0.00980
D_wy__CB = D_wx__CB
D_wz__CB = D_wx__CB
D_omega_xy__CB = D_wx__CB
D_omega_z__CB = D_wz__CB

D_CB = np.diag([D_vx__CB, D_vy__CB, D_vz__CB, D_wx__CB, D_wy__CB, D_wz__CB])

## Simulation

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
theta0 = 5 * np.pi/180
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

w_y_error_int = 0
w_x_error_int = 0

w_y_error_prev = 0
w_x_error_prev = 0

th_error_int = 0
th_error_prev = 0

phi_error_int = 0
phi_error_prev = 0

fx_history = np.empty(len(time_vec))
fz_history = np.empty(len(time_vec))

Q = np.array([[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],         # x
              [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],         # y
              [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],         # z
              [0, 0, 0, 1000, 0, 0, 0, 0, 0, 0, 0, 0],      # phi
              [0, 0, 0, 0, 1000, 0, 0, 0, 0, 0, 0, 0],      # theta
              [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],         # psi
              [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],         # v_x
              [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],         # v_y
              [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],         # v_z
              [0, 0, 0, 0, 0, 0, 0, 0, 0, 1000, 0, 0],      # w_x
              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1000, 0],      # w_y
              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]])        # w_z

R = np.eye(4) * 1

# Using Bryson's Rule

max_acceptable_theta = 5 * np.pi/180
max_acceptable_phi = 5 * np.pi/180
max_acceptable_wy = 0.1
max_acceptable_wx = 0.1

Q = np.array([[0.000001, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
              [0, 0.000001, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0.000001, 0, 0, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 1/max_acceptable_phi**2, 0, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 1/max_acceptable_theta**2, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0.000001, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0.000001, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0.000001, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0, 0.000001, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0, 0, 1/max_acceptable_wx**2, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1/max_acceptable_wy**2, 0],
              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.000001]])

# Q[7][11] = 1
# Q[11][7] = 1
# Q[9][11] = 1
# Q[11][9] = 1
# Q[10][8] = 1
# Q[8][10] = 1

max_acceptable_fx = 5
max_acceptable_fy = 5
max_acceptable_fz = 5

R = np.array([[1/max_acceptable_fx**2, 0, 0 ,0],
              [0, 1/max_acceptable_fy**2, 0, 0],
              [0, 0, 1/max_acceptable_fz**2, 0],
              [0, 0, 0, 1]])

# Set up figure

fig = plt.figure()
ax_3d = fig.add_subplot(322, projection='3d')
plt.ion()
ax_3d.grid()

ax_or = fig.add_subplot(321, projection='3d')
ax_or.grid()

ax_xy = fig.add_subplot(323)

ax_zd = fig.add_subplot(324)

ax_v = fig.add_subplot(325)

ax_w = fig.add_subplot(326)

plt.subplots_adjust(hspace=1.0)

try:
    for n in range(len(time_vec) - 1):
        if time_vec[n] > 50:
            input()
            sys.exit()
        
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

        w_x_dot = state_dot[9, n]
        w_y_dot = state_dot[10, n]
        w_z_dot = state_dot[11, n]

        # # Set x, y, z = 0
        # state[0, n] = 0
        # state[1, n] = 0
        # state[2, n] = 0
        # eta_bn_n[0] = 0
        # eta_bn_n[1] = 0
        # eta_bn_n[2] = 0
        
        # Set vx, vy, vz = 0
        # state[6, n] = 0
        # state[7, n] = 0
        # state[8, n] = 0
        # nu_bn_b[0] = 0
        # nu_bn_b[1] = 0
        # nu_bn_b[2] = 0

        # Set wx, wz = 0
        # state[9] = 0
        # state[11] = 0
        # nu_bn_b[3] = 0
        # nu_bn_b[5] = 0

        # Set phi, psi = 0
        # state[3] = 0
        # state[5] = 0
        # eta_bn_n[3] = 0
        # eta_bn_n[5] = 0

        # A_lin = np.array([
        #     [0, 1, 0, 0],
        #     [-0.154, -0.0168, 3.9e-4, 0.495*v_x__b],
        #     [0.00304, 3.33e-4, -0.0249, -0.00979*v_x__b],
        #     [0, 0.615*v_x__b, 0, -0.064]
        # ])

        # B_lin = np.array([
        #     [0, 0],
        #     [0.0398, 0],
        #     [2.1661, 0],
        #     [0, 1.3335]
        # ])

        # max_allowable_theta = 0.05
        # max_allowable_wy = 0.02
        # max_allowable_vx = 0.5
        # max_allowable_vz = 0.5
        
        # Q = np.array([
        #     [1/max_allowable_theta**2, 0, 0, 0],
        #     [0, 1/max_allowable_wy**2, 0, 0],
        #     [0, 0, 1/max_allowable_vx**2, 0],
        #     [0, 0, 0, 1/max_allowable_vz**2]
        # ])
        # R = np.eye(2)

        # K = control.lqr(A_lin, B_lin, Q, R)[0]

        # f_out = -K @ np.array([theta, theta_dot, v_x__b, v_z__b]).reshape((4,1))
        # u = np.array([f_out[0].item(), 0, f_out[1].item(), 0]).reshape(4,1)

        max_allowable_theta = 0.05
        max_allowable_phi = 0.05
        
        max_allowable_wy = 0.02
        max_allowable_wx = 0.02

        max_allowable_vx = 0.5
        max_allowable_vy = 0.5

        max_allowable_vz = 0.5

        Q = np.array([
            [1/max_allowable_theta**2, 0, 0, 0, 0, 0, 0],
            [0, 1/max_allowable_wy**2, 0, 0, 0, 0, 0],
            [0, 0, 1/max_allowable_phi**2, 0, 0, 0, 0],
            [0, 0, 0, 1/max_allowable_wx**2, 0, 0, 0],
            [0, 0, 0, 0, 1/max_allowable_vx**2, 0, 0],
            [0, 0, 0, 0, 0, 1/max_allowable_vy**2, 0],
            [0, 0, 0, 0, 0, 0, 1/max_allowable_vz**2]
        ])

        R = np.eye(3)

        A_lin = np.array([
            [0, np.cos(phi), -1*w_y__b*np.sin(phi), 0, 0, 0, 0],
            [-0.154*np.cos(theta), 0.00979*v_z__b-0.0168, 0, 0, 0.495*v_z__b+3.9e-4, 0, 0.495*v_x__b+0.00979*w_y__b],
            [-(w_y__b*np.sin(phi))/(np.sin(theta)**2-1), np.sin(phi)*np.tan(theta), w_y__b*np.cos(phi)*np.tan(theta), 1, 0, 0, 0],
            [0.154*np.sin(phi)*np.sin(theta), 0, -0.154*np.cos(phi)*np.cos(theta), 0.00979*v_z__b-0.0168, 0, -0.495*v_z__b-3.9e-4, 0.00979*w_x__b-0.495*v_y__b],
            [0, -1.62*v_z__b, 0, 0,-0.0249, 0, -1.62*w_y__b],
            [0, 0, 0, 1.62*v_z__b, 0, -0.0249, 1.62*w_x__b],
            [0, 0.615*v_x__b+0.0244*w_y__b, 0, 0.0244*w_x__b-0.615*v_y__b, 0.615*w_y__b, -0.615*w_x__b, -0.064]
        ])
        
        B_lin = np.array([
            [0, 0, 0],
            [0.0398, 0, 0],
            [0, 0, 0],
            [0, -0.0398, 0],
            [2.17, 0, 0],
            [0, 2.17, 0],
            [0, 0, 1.33]
        ])

        K = control.lqr(A_lin, B_lin, Q, R)[0]
        f_out = -K @ np.array([theta, w_y__b, phi, w_x__b, v_x__b, v_y__b, v_z__b]).reshape((7, 1))
        u_swing = np.array([f_out[0].item(),
                            f_out[1].item(),
                            f_out[2].item(), 0]).reshape((4, 1))
        u = u_swing

        f_x = u[0].item()
        f_z = u[2].item()

        fx_history[n] = f_x
        fz_history[n] = f_z
    
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
        state_dot[:, n+1] = np.vstack((eta_bn_n_dot, nu_bn_b_dot)).reshape(N)

        # state_dot[:, n+1] = A_lin @ state[:, n].reshape(12) + (B_lin @ u).reshape(12)
        # state[:, n+1] = state[:, n] + dT * state_dot[:, n]

        ax_3d.cla()
        ax_3d.scatter(state[0, 0:n], state[1, 0:n], state[2, 0:n], color='blue', s=100)
        ax_3d.scatter(eta_bn_n[0], eta_bn_n[1], eta_bn_n[2], color='m', s=200)
        ax_3d.invert_yaxis()
        ax_3d.invert_zaxis()

        ax_or.cla()

        x_mag = 2
        y_mag = 2
        z_mag = 2
        
        blimp_x_vector = R_b__n(eta_bn_n[3], eta_bn_n[4], eta_bn_n[5]) \
                        @ np.array([x_mag, 0, 0]).T
        blimp_y_vector = R_b__n(eta_bn_n[3], eta_bn_n[4], eta_bn_n[5]) \
                        @ np.array([0, y_mag, 0]).T
        blimp_z_vector = R_b__n(eta_bn_n[3], eta_bn_n[4], eta_bn_n[5]) \
                        @ np.array([0, 0, z_mag]).T
    
        qx = ax_or.quiver(eta_bn_n[0], eta_bn_n[1], eta_bn_n[2], \
                blimp_x_vector[0], blimp_x_vector[1], blimp_x_vector[2], \
                color='r')
        qx.ShowArrowHead = 'on'
        qy = ax_or.quiver(eta_bn_n[0], eta_bn_n[1], eta_bn_n[2], \
                blimp_y_vector[0], blimp_y_vector[1], blimp_y_vector[2], \
                color='g')
        qy.ShowArrowHead = 'on'
        qz = ax_or.quiver(eta_bn_n[0], eta_bn_n[1], eta_bn_n[2], \
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

        ax_xy.cla()
        ax_xy.plot(time_vec[0:n], state[0, 0:n])
        ax_xy.plot(time_vec[0:n], state[1, 0:n])
        ax_xy.plot(time_vec[0:n], state[2, 0:n])
        ax_xy.legend(['x', 'y', 'z'])
        ax_xy.set_title('Position')

        ax_zd.cla()
        ax_zd.plot(time_vec[0:n], state[3, 0:n] * 180/np.pi)
        ax_zd.plot(time_vec[0:n], state[4, 0:n] * 180/np.pi)
        ax_zd.plot(time_vec[0:n], state[5, 0:n] * 180/np.pi)
        ax_zd.set_ylim(-20, 20)
        ax_zd.legend(['phi', 'theta', 'psi'])
        ax_zd.set_title('Pitch/roll')

        ax_v.cla()
        ax_v.plot(time_vec[0:n], state[6, 0:n])
        ax_v.plot(time_vec[0:n], state[7, 0:n])
        ax_v.plot(time_vec[0:n], state[8, 0:n])
        ax_v.legend(['vx', 'vy', 'vz'])
        ax_v.set_title('Velocity')

        ax_w.cla()
        ax_w.plot(time_vec[0:n], state[9, 0:n])
        ax_w.plot(time_vec[0:n], state[10, 0:n])
        ax_w.plot(time_vec[0:n], state[11, 0:n])
        ax_w.legend(['wx', 'wy', 'wz'])
        ax_w.set_title('Omega')

        plt.draw()
        plt.pause(0.001)

except KeyboardInterrupt:
    sys.exit()