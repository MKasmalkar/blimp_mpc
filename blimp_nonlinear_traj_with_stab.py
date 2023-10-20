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

# Zero dynamics compensator parameters
zd_Q = np.eye(12)
zd_R = np.eye(4)

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

# x0 = 9.85601983e-01
# y0 = -1.55297831e-01
# z0 = -1.95001036e+00

# phi0 = 1.17189905e-02
# theta0 = -1.45800250e-01
# psi0 = 7.69690160e+00

# v_x0 = 2.95591953e-01
# v_y0 = 1.02004348e-03
# v_z0 = -1.44491356e-01

# w_x0 = 4.48550691e-02
# w_y0 = 4.15662901e-02
# w_z0 = 3.10362917e-01

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

# traj_x = np.zeros(len(time_vec))
# traj_y = np.zeros(len(time_vec))
# traj_z = np.zeros(len(time_vec))
# traj_psi = np.zeros(len(time_vec))

# traj_x_dot = np.zeros(len(time_vec))
# traj_y_dot = np.zeros(len(time_vec))
# traj_z_dot = np.zeros(len(time_vec))
# traj_psi_dot = np.zeros(len(time_vec))

# traj_x_ddot = np.zeros(len(time_vec))
# traj_y_ddot = np.zeros(len(time_vec))
# traj_z_ddot = np.zeros(len(time_vec))
# traj_psi_ddot = np.zeros(len(time_vec))

# State vector
eta_bn_n = np.array([[x0, y0, z0, phi0, theta0, psi0]]).T
nu_bn_b = np.array([[v_x0, v_y0, v_z0, w_x0, w_y0, w_z0]]).T

state = np.empty((N, len(time_vec)))
state_dot = np.empty((N, len(time_vec)))

state[:, 0] = np.vstack((eta_bn_n, nu_bn_b)).reshape(N)
state_dot[:, 0] = np.zeros(N)

state[:, 390] = np.array([9.85601983e-01,
                        -1.55297831e-01,
                        -1.95001036e+00,  
                        1.17189905e-02, 
                        -1.45800250e-01, 
                         7.69690160e+00,  
                         2.95591953e-01,  
                         1.02004348e-03, 
                        -1.44491356e-01, 
                         4.48550691e-02, 
                         4.15662901e-02, 
                         3.10362917e-01])

eta_bn_n = state[0:6, 390].reshape((6, 1))
nu_bn_b = state[6:12, 390].reshape((6, 1))

state_dot[:, 390] = np.array([ 5.12175211e-02,  
                            3.09243929e-01,
                            -9.99919352e-02,
                            -1.15550288e-03, 
                            3.78695897e-02,  
                            3.14162015e-01,  
                            5.48175380e-03,  
                            2.96498902e-04, 
                            1.11918405e-02, 
                            -4.41363447e-03, 
                             7.97562719e-04,  
                             1.77348477e-03])

error = np.empty((2, 4, len(time_vec)))

## Swing reducing controller

v_x_error_int = 0
v_y_error_int = 0
theta_error_int = 0
phi_error_int = 0

# velocity setpoints
k_x = 1
k_y = 1

# Theta setpoint
kp_x = 1
ki_x = 1

# Phi setpoint
kp_y = 1
ki_y = 1

# f_x computation
kfx_th = 1
kfx_w = 1

# f_y computation
kfy_phi = 1
kfy_w = 1

# z correction
kz = 0

# psi correction
kpsi = 0


# LQR Using Bryson's Rule

max_acceptable_theta = 5 * np.pi/180
max_acceptable_phi = 5 * np.pi/180
max_acceptable_wy = 0.1
max_acceptable_wx = 0.1

Q = np.array([[10, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
              [0, 10, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 10, 0, 0, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 1/max_acceptable_phi**2, 0, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 1/max_acceptable_theta**2, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0, 0, 1/max_acceptable_wx**2, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1/max_acceptable_wy**2, 0],
              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]])

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


K = control.lqr(A_lin, B_lin, Q, R)[0]

## Set up figure

fig = plt.figure()
ax_3d = fig.add_subplot(211, projection='3d')
plt.ion()
ax_3d.grid()

ax_err = fig.add_subplot(223)

ax_zd = fig.add_subplot(224)

plt.subplots_adjust(wspace=0.5)

try:
    for n in range(390, len(time_vec) - 1):
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

        ## Feedback-linearized tracking controller

        # Compute input to integrator chain
        A = np.array([[v_y__b*(- np.cos(phi)*np.cos(psi)*psi_dot + np.sin(phi)*np.sin(psi)*phi_dot + ((np.cos(phi)*np.sin(psi) - np.cos(psi)*np.sin(phi)*np.sin(theta))*(D_vxy__CB*I_x + m_RB*m_z*r_z_gb__b*v_z__b))/(I_x*m_y - m_RB**2*r_z_gb__b**2) + np.cos(phi)*np.cos(psi)*np.sin(theta)*phi_dot + np.cos(psi)*np.cos(theta)*np.sin(phi)*theta_dot - np.sin(phi)*np.sin(psi)*np.sin(theta)*psi_dot) + w_z__b*(((I_x*(m_x*v_x__b + m_RB*r_z_gb__b*w_y__b))/(I_x*m_y - m_RB**2*r_z_gb__b**2) - (m_RB*r_z_gb__b*(I_y*w_y__b + m_RB*r_z_gb__b*v_x__b))/(I_x*m_y - m_RB**2*r_z_gb__b**2))*(np.cos(phi)*np.sin(psi) - np.cos(psi)*np.sin(phi)*np.sin(theta)) + np.cos(psi)*np.cos(theta)*((I_y*(m_y*v_y__b - m_RB*r_z_gb__b*w_x__b))/(I_y*m_x - m_RB**2*r_z_gb__b**2) + (m_RB*r_z_gb__b*(I_x*w_x__b - m_RB*r_z_gb__b*v_y__b))/(I_y*m_x - m_RB**2*r_z_gb__b**2))) - v_z__b*((D_vz__CB*(np.sin(phi)*np.sin(psi) + np.cos(phi)*np.cos(psi)*np.sin(theta)))/m_z - np.cos(phi)*np.sin(psi)*phi_dot - np.cos(psi)*np.sin(phi)*psi_dot + np.cos(psi)*np.sin(phi)*np.sin(theta)*phi_dot + np.cos(phi)*np.sin(psi)*np.sin(theta)*psi_dot - np.cos(phi)*np.cos(psi)*np.cos(theta)*theta_dot + (m_RB*r_z_gb__b*(m_y*v_y__b - m_RB*r_z_gb__b*w_x__b)*(np.cos(phi)*np.sin(psi) - np.cos(psi)*np.sin(phi)*np.sin(theta)))/(I_x*m_y - m_RB**2*r_z_gb__b**2) - (m_RB*r_z_gb__b*np.cos(psi)*np.cos(theta)*(m_x*v_x__b + m_RB*r_z_gb__b*w_y__b))/(I_y*m_x - m_RB**2*r_z_gb__b**2)) - w_x__b*(((m_y*v_y__b - m_RB*r_z_gb__b*w_x__b)*(np.sin(phi)*np.sin(psi) + np.cos(phi)*np.cos(psi)*np.sin(theta)))/m_z + ((np.cos(phi)*np.sin(psi) - np.cos(psi)*np.sin(phi)*np.sin(theta))*(I_x*m_z*v_z__b - D_omega_xy__CB*m_RB*r_z_gb__b))/(I_x*m_y - m_RB**2*r_z_gb__b**2) + (I_z*m_RB*r_z_gb__b*np.cos(psi)*np.cos(theta)*w_z__b)/(I_y*m_x - m_RB**2*r_z_gb__b**2)) + w_y__b*(((m_x*v_x__b + m_RB*r_z_gb__b*w_y__b)*(np.sin(phi)*np.sin(psi) + np.cos(phi)*np.cos(psi)*np.sin(theta)))/m_z - (np.cos(psi)*np.cos(theta)*(I_y*m_z*v_z__b - D_omega_xy__CB*m_RB*r_z_gb__b))/(I_y*m_x - m_RB**2*r_z_gb__b**2) + (I_z*m_RB*r_z_gb__b*w_z__b*(np.cos(phi)*np.sin(psi) - np.cos(psi)*np.sin(phi)*np.sin(theta)))/(I_x*m_y - m_RB**2*r_z_gb__b**2)) - v_x__b*(np.cos(theta)*np.sin(psi)*psi_dot + np.cos(psi)*np.sin(theta)*theta_dot + (np.cos(psi)*np.cos(theta)*(D_vxy__CB*I_y + m_RB*m_z*r_z_gb__b*v_z__b))/(I_y*m_x - m_RB**2*r_z_gb__b**2)) + (f_g*m_RB*r_z_gb__b**2*np.cos(theta)*np.sin(phi)*(np.cos(phi)*np.sin(psi) - np.cos(psi)*np.sin(phi)*np.sin(theta)))/(I_x*m_y - m_RB**2*r_z_gb__b**2) + (f_g*m_RB*r_z_gb__b**2*np.cos(psi)*np.cos(theta)*np.sin(theta))/(I_y*m_x - m_RB**2*r_z_gb__b**2)],
                      [v_z__b*((D_vz__CB*(np.cos(psi)*np.sin(phi) - np.cos(phi)*np.sin(psi)*np.sin(theta)))/m_z - np.cos(phi)*np.cos(psi)*phi_dot + np.sin(phi)*np.sin(psi)*psi_dot + np.cos(phi)*np.cos(psi)*np.sin(theta)*psi_dot + np.cos(phi)*np.cos(theta)*np.sin(psi)*theta_dot - np.sin(phi)*np.sin(psi)*np.sin(theta)*phi_dot + (m_RB*r_z_gb__b*(m_y*v_y__b - m_RB*r_z_gb__b*w_x__b)*(np.cos(phi)*np.cos(psi) + np.sin(phi)*np.sin(psi)*np.sin(theta)))/(I_x*m_y - m_RB**2*r_z_gb__b**2) + (m_RB*r_z_gb__b*np.cos(theta)*np.sin(psi)*(m_x*v_x__b + m_RB*r_z_gb__b*w_y__b))/(I_y*m_x - m_RB**2*r_z_gb__b**2)) - w_z__b*(((I_x*(m_x*v_x__b + m_RB*r_z_gb__b*w_y__b))/(I_x*m_y - m_RB**2*r_z_gb__b**2) - (m_RB*r_z_gb__b*(I_y*w_y__b + m_RB*r_z_gb__b*v_x__b))/(I_x*m_y - m_RB**2*r_z_gb__b**2))*(np.cos(phi)*np.cos(psi) + np.sin(phi)*np.sin(psi)*np.sin(theta)) - np.cos(theta)*np.sin(psi)*((I_y*(m_y*v_y__b - m_RB*r_z_gb__b*w_x__b))/(I_y*m_x - m_RB**2*r_z_gb__b**2) + (m_RB*r_z_gb__b*(I_x*w_x__b - m_RB*r_z_gb__b*v_y__b))/(I_y*m_x - m_RB**2*r_z_gb__b**2))) - v_y__b*(np.cos(psi)*np.sin(phi)*phi_dot + np.cos(phi)*np.sin(psi)*psi_dot + ((np.cos(phi)*np.cos(psi) + np.sin(phi)*np.sin(psi)*np.sin(theta))*(D_vxy__CB*I_x + m_RB*m_z*r_z_gb__b*v_z__b))/(I_x*m_y - m_RB**2*r_z_gb__b**2) - np.cos(phi)*np.sin(psi)*np.sin(theta)*phi_dot - np.cos(psi)*np.sin(phi)*np.sin(theta)*psi_dot - np.cos(theta)*np.sin(phi)*np.sin(psi)*theta_dot) + w_x__b*(((m_y*v_y__b - m_RB*r_z_gb__b*w_x__b)*(np.cos(psi)*np.sin(phi) - np.cos(phi)*np.sin(psi)*np.sin(theta)))/m_z + ((np.cos(phi)*np.cos(psi) + np.sin(phi)*np.sin(psi)*np.sin(theta))*(I_x*m_z*v_z__b - D_omega_xy__CB*m_RB*r_z_gb__b))/(I_x*m_y - m_RB**2*r_z_gb__b**2) - (I_z*m_RB*r_z_gb__b*np.cos(theta)*np.sin(psi)*w_z__b)/(I_y*m_x - m_RB**2*r_z_gb__b**2)) - w_y__b*(((m_x*v_x__b + m_RB*r_z_gb__b*w_y__b)*(np.cos(psi)*np.sin(phi) - np.cos(phi)*np.sin(psi)*np.sin(theta)))/m_z + (np.cos(theta)*np.sin(psi)*(I_y*m_z*v_z__b - D_omega_xy__CB*m_RB*r_z_gb__b))/(I_y*m_x - m_RB**2*r_z_gb__b**2) + (I_z*m_RB*r_z_gb__b*w_z__b*(np.cos(phi)*np.cos(psi) + np.sin(phi)*np.sin(psi)*np.sin(theta)))/(I_x*m_y - m_RB**2*r_z_gb__b**2)) - v_x__b*(- np.cos(psi)*np.cos(theta)*psi_dot + np.sin(psi)*np.sin(theta)*theta_dot + (np.cos(theta)*np.sin(psi)*(D_vxy__CB*I_y + m_RB*m_z*r_z_gb__b*v_z__b))/(I_y*m_x - m_RB**2*r_z_gb__b**2)) - (f_g*m_RB*r_z_gb__b**2*np.cos(theta)*np.sin(phi)*(np.cos(phi)*np.cos(psi) + np.sin(phi)*np.sin(psi)*np.sin(theta)))/(I_x*m_y - m_RB**2*r_z_gb__b**2) + (f_g*m_RB*r_z_gb__b**2*np.cos(theta)*np.sin(psi)*np.sin(theta))/(I_y*m_x - m_RB**2*r_z_gb__b**2)],
                      [w_x__b*((np.cos(theta)*np.sin(phi)*(I_x*m_z*v_z__b - D_omega_xy__CB*m_RB*r_z_gb__b))/(I_x*m_y - m_RB**2*r_z_gb__b**2) - (np.cos(phi)*np.cos(theta)*(m_y*v_y__b - m_RB*r_z_gb__b*w_x__b))/m_z + (I_z*m_RB*r_z_gb__b*np.sin(theta)*w_z__b)/(I_y*m_x - m_RB**2*r_z_gb__b**2)) + w_y__b*((np.sin(theta)*(I_y*m_z*v_z__b - D_omega_xy__CB*m_RB*r_z_gb__b))/(I_y*m_x - m_RB**2*r_z_gb__b**2) + (np.cos(phi)*np.cos(theta)*(m_x*v_x__b + m_RB*r_z_gb__b*w_y__b))/m_z - (I_z*m_RB*r_z_gb__b*np.cos(theta)*np.sin(phi)*w_z__b)/(I_x*m_y - m_RB**2*r_z_gb__b**2)) - v_x__b*(np.cos(theta)*theta_dot - (np.sin(theta)*(D_vxy__CB*I_y + m_RB*m_z*r_z_gb__b*v_z__b))/(I_y*m_x - m_RB**2*r_z_gb__b**2)) - w_z__b*(np.cos(theta)*np.sin(phi)*((I_x*(m_x*v_x__b + m_RB*r_z_gb__b*w_y__b))/(I_x*m_y - m_RB**2*r_z_gb__b**2) - (m_RB*r_z_gb__b*(I_y*w_y__b + m_RB*r_z_gb__b*v_x__b))/(I_x*m_y - m_RB**2*r_z_gb__b**2)) + (np.sin(theta)*(I_y*m_y*v_y__b - m_RB**2*r_z_gb__b**2*v_y__b + I_x*m_RB*r_z_gb__b*w_x__b - I_y*m_RB*r_z_gb__b*w_x__b))/(I_y*m_x - m_RB**2*r_z_gb__b**2)) - v_z__b*(np.cos(theta)*np.sin(phi)*phi_dot + np.cos(phi)*np.sin(theta)*theta_dot + (D_vz__CB*np.cos(phi)*np.cos(theta))/m_z + (m_RB*r_z_gb__b*np.sin(theta)*(m_x*v_x__b + m_RB*r_z_gb__b*w_y__b))/(I_y*m_x - m_RB**2*r_z_gb__b**2) - (m_RB*r_z_gb__b*np.cos(theta)*np.sin(phi)*(m_y*v_y__b - m_RB*r_z_gb__b*w_x__b))/(I_x*m_y - m_RB**2*r_z_gb__b**2)) - v_y__b*(- np.cos(phi)*np.cos(theta)*phi_dot + np.sin(phi)*np.sin(theta)*theta_dot + (np.cos(theta)*np.sin(phi)*(D_vxy__CB*I_x + m_RB*m_z*r_z_gb__b*v_z__b))/(I_x*m_y - m_RB**2*r_z_gb__b**2)) - (f_g*m_RB*r_z_gb__b**2*np.sin(theta)**2)/(I_y*m_x - m_RB**2*r_z_gb__b**2) + (f_g*m_RB*r_z_gb__b**2*np.sin(phi)**2*(np.sin(theta)**2 - 1))/(I_x*m_y - m_RB**2*r_z_gb__b**2)],
                      [w_y__b*((np.cos(phi)*phi_dot)/np.cos(theta) + (np.cos(phi)*(I_x*w_x__b - m_RB*r_z_gb__b*v_y__b))/(I_z*np.cos(theta)) - (np.sin(phi)*(D_omega_xy__CB*m_x - m_RB*m_z*r_z_gb__b*v_z__b))/(np.cos(theta)*(I_y*m_x - m_RB**2*r_z_gb__b**2)) - (np.sin(phi)*np.sin(theta)*theta_dot)/(np.sin(theta)**2 - 1)) - w_z__b*((np.sin(phi)*phi_dot)/np.cos(theta) + (D_omega_z__CB*np.cos(phi))/(I_z*np.cos(theta)) - (np.cos(phi)*np.sin(theta)*theta_dot)/np.cos(theta)**2 + (np.sin(phi)*(I_x*m_x*w_x__b - m_RB**2*r_z_gb__b**2*w_x__b - m_RB*m_x*r_z_gb__b*v_y__b + m_RB*m_y*r_z_gb__b*v_y__b))/(np.cos(theta)*(I_y*m_x - m_RB**2*r_z_gb__b**2))) - v_x__b*((np.cos(phi)*(m_y*v_y__b - m_RB*r_z_gb__b*w_x__b))/(I_z*np.cos(theta)) - (np.sin(phi)*(m_x*m_z*v_z__b + D_vxy__CB*m_RB*r_z_gb__b))/(np.cos(theta)*(I_y*m_x - m_RB**2*r_z_gb__b**2))) - w_x__b*((np.cos(phi)*(I_y*w_y__b + m_RB*r_z_gb__b*v_x__b))/(I_z*np.cos(theta)) - (I_z*m_x*np.sin(phi)*w_z__b)/(np.cos(theta)*(I_y*m_x - m_RB**2*r_z_gb__b**2))) + (np.cos(phi)*v_y__b*(m_x*v_x__b + m_RB*r_z_gb__b*w_y__b))/(I_z*np.cos(theta)) - (m_x*np.sin(phi)*v_z__b*(m_x*v_x__b + m_RB*r_z_gb__b*w_y__b))/(np.cos(theta)*(I_y*m_x - m_RB**2*r_z_gb__b**2)) - (f_g*m_x*r_z_gb__b*np.sin(phi)*np.sin(theta))/(np.cos(theta)*(I_y*m_x - m_RB**2*r_z_gb__b**2))]
                     ])
        
        Binv = np.array([[(np.cos(psi)*np.cos(theta)*(I_y*m_x - m_RB**2*r_z_gb__b**2))/(I_y - m_RB*r_z_gb__b*r_z_tg__b), (np.cos(theta)*np.sin(psi)*(I_y*m_x - m_RB**2*r_z_gb__b**2))/(I_y - m_RB*r_z_gb__b*r_z_tg__b), -(np.sin(theta)*(I_y*m_x - m_RB**2*r_z_gb__b**2))/(I_y - m_RB*r_z_gb__b*r_z_tg__b), 0],
                         [-((I_x*m_y - m_RB**2*r_z_gb__b**2)*(np.cos(phi)*np.sin(psi) - np.cos(psi)*np.sin(phi)*np.sin(theta)))/(I_x - m_RB*r_z_gb__b*r_z_tg__b), ((I_x*m_y - m_RB**2*r_z_gb__b**2)*(np.cos(phi)*np.cos(psi) + np.sin(phi)*np.sin(psi)*np.sin(theta)))/(I_x - m_RB*r_z_gb__b*r_z_tg__b), (np.cos(theta)*np.sin(phi)*(I_x*m_y - m_RB**2*r_z_gb__b**2))/(I_x - m_RB*r_z_gb__b*r_z_tg__b), 0],
                         [m_z*np.sin(phi)*np.sin(psi) + m_z*np.cos(phi)*np.cos(psi)*np.sin(theta), m_z*np.cos(phi)*np.sin(psi)*np.sin(theta) - m_z*np.cos(psi)*np.sin(phi), m_z*np.cos(phi)*np.cos(theta), 0],
                         [(I_z*np.cos(psi)*np.cos(theta)*np.sin(phi)*(m_RB*r_z_gb__b - m_x*r_z_tg__b))/(np.cos(phi)*(I_y - m_RB*r_z_gb__b*r_z_tg__b)), (I_z*np.cos(theta)*np.sin(phi)*np.sin(psi)*(m_RB*r_z_gb__b - m_x*r_z_tg__b))/(np.cos(phi)*(I_y - m_RB*r_z_gb__b*r_z_tg__b)), -(I_z*np.sin(phi)*np.sin(theta)*(m_RB*r_z_gb__b - m_x*r_z_tg__b))/(np.cos(phi)*(I_y - m_RB*r_z_gb__b*r_z_tg__b)), (I_z*np.cos(theta))/np.cos(phi)]
                        ])
        
        zeta1 = np.array([[x],
                          [y],
                          [z],
                          [psi]])
        zeta2 = np.array([[x_dot],
                          [y_dot],
                          [z_dot],
                          [psi_dot]])

        yd = np.array([[traj_x[n]],
                       [traj_y[n]],
                       [traj_z[n]],
                       [traj_psi[n]]])
        yd_dot = np.array([[traj_x_dot[n]],
                           [traj_y_dot[n]],
                           [traj_z_dot[n]],
                           [traj_psi_dot[n]]])
        yd_ddot = np.array([[traj_x_ddot[n]],
                            [traj_y_ddot[n]],
                            [traj_z_ddot[n]],
                            [traj_psi_ddot[n]]])
        
        e1 = zeta1 - yd
        e2 = zeta2 - yd_dot
        
        error[0, :, n] = e1.reshape(4)
        error[1, :, n] = e2.reshape(4)

        k1 = 10
        k2 = 10

        q = -k1 * e1 - k2 * e2 + yd_ddot

        u_traj = Binv @ (q - A)

        ## Swing-reducing controller
        
        # # Compute setpoints
        # x_error = e1[0].item()
        # y_error = e1[1].item()
        # z_error = e1[2].item()
        # psi_error = e1[3].item()

        # v_sp_b = R_b__n_inv(phi, theta, psi) @ np.array([x_error, y_error, z_error]).reshape((3, 1))
        # v_x_sp = -k_x * v_sp_b[0].item()
        # v_y_sp = -k_y * v_sp_b[1].item()

        # # Update v_x error and integral of v_x error
        # delta_v_x = v_x__b - v_x_sp
        # v_x_error_int += delta_v_x * dT
        
        # # Compute theta setpoint
        # theta_sp = np.arcsin(D_vxy__CB * r_z_tg__b / (r_z_gb__b * m_RB * g_acc) * v_x_sp) \
        #             - kp_x * delta_v_x - ki_x * v_x_error_int
        
        # # Update theta error and integral of theta error
        # theta_error = theta - theta_sp
        # theta_error_int += theta_error * dT
        # wy_error = w_y__b - 0

        # # Compute f_x
        # f_x = r_z_gb__b / r_z_tg__b * m_RB * g_acc * np.sin(theta_sp) \
        #         - kfx_th * theta_error \
        #         - kfx_w * wy_error
        
        # # Update v_y_error and integral of v_y error
        # delta_v_y = v_y__b - v_y_sp
        # v_y_error_int += delta_v_y * dT

        # # Compute phi setpoint
        # phi_sp = np.arcsin(D_vxy__CB * r_z_tg__b / (r_z_gb__b * m_RB * g_acc) * v_y_sp) \
        #             - kp_y * delta_v_y - ki_y * v_y_error_int
        
        # # Update phi error and integral of phi error
        # phi_error = phi - phi_sp
        # phi_error_int += phi_error * dT
        # wx_error = w_x__b - 0

        # # Compute f_y
        # f_y = r_z_gb__b / r_z_tg__b * m_RB * g_acc * np.sin(phi_sp) \
        #         - kfy_phi * phi_error \
        #         - kfy_w * wx_error
        
        # # z correction
        # f_z = -kz * v_sp_b[2].item()

        # # psi correction
        # tau_z = -kpsi * psi_error
        
        # # Swing reducing controller input
        # u_swing = np.array([f_x, f_y, f_z, tau_z]).reshape((4, 1))

        # # Compute total input

        # k_traj_wx = 100
        # k_traj_wy = 100
        # k_traj = np.diag([min(1, max(1 - abs((abs(theta) % np.pi * k_traj_wy * w_y__b)) / (np.pi/20), 0)),
        #                   min(1, max(1 - abs((abs(phi) % np.pi * k_traj_wx * w_x__b)) / (np.pi/20), 0)),
        #                   1,
        #                   1])
        
        # k_swing = np.eye(4) - k_traj

        # k_lqr = 0.05
        # print(np.around(u_traj.reshape(4), 3), '\t', np.around(k_lqr * u_lqr.reshape(4), 3))
        
        if time_vec[n] < 20:
            # u = u_traj + k_lqr * u_lqr
            u = u_traj
        else:
            # LQR
            lqr_err = np.array([[x - traj_x[n]],
                                [y - traj_y[n]],
                                [z - traj_z[n]],
                                [phi],
                                [theta],
                                [psi - traj_psi[n]],
                                [v_x__b],
                                [v_y__b],
                                [v_z__b],
                                [w_x__b],
                                [w_y__b],
                                [w_z__b]])
            
            u_lqr = -K @ lqr_err
            u = u_traj + 0.01 * u_lqr
        
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

        # print("Inputs:")
        # print("Fx: " + str(u[0]))
        # print("Fy: " + str(u[1]))
        # print("Fz: " + str(u[2]))
        # print("Tz: " + str(u[3]))
        # print()

        # print("Position:")
        # print("X: " + str(eta_bn_n[0]))
        # print("Y: " + str(eta_bn_n[1]))
        # print("Z: " + str(eta_bn_n[2]))
        # print("Phi: " + str(eta_bn_n[3]))
        # print("Theta: " + str(eta_bn_n[4]))
        # print("Psi: " + str(eta_bn_n[5]))
        # print()

        # print("Error: ")
        # print("E1:\n" + str(e1))
        # print("E2:\n" + str(e2))
        
        # print("Rates:")
        # print("X: " + str(nu_bn_b[0]))
        # print("Y: " + str(nu_bn_b[1]))
        # print("Z: " + str(nu_bn_b[2]))
        # print("Phi: " + str(nu_bn_b[3]))
        # print("Theta: " + str(nu_bn_b[4]))
        # print("Psi: " + str(nu_bn_b[5]))
        
        ax_3d.cla()
        ax_3d.scatter(state[0, 0:n], state[1, 0:n], state[2, 0:n], color='blue', s=100)
        ax_3d.scatter(eta_bn_n[0], eta_bn_n[1], eta_bn_n[2], color='m', s=200)
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
                color='magenta')
        qx.ShowArrowHead = 'on'
        qy = ax_3d.quiver(eta_bn_n[0], eta_bn_n[1], eta_bn_n[2], \
                blimp_y_vector[0], blimp_y_vector[1], blimp_y_vector[2], \
                color='magenta')
        qy.ShowArrowHead = 'on'
        qz = ax_3d.quiver(eta_bn_n[0], eta_bn_n[1], eta_bn_n[2], \
                blimp_z_vector[0], blimp_z_vector[1], blimp_z_vector[2], \
                color='magenta')
        qz.ShowArrowHead = 'on'

        ax_3d.set_xlabel('x')
        ax_3d.set_ylabel('y')
        ax_3d.set_zlabel('z')
        ax_3d.set_title('Trajectory')
        # ax_3d.set_xlim(-0.001, 0.001)
        # ax_3d.set_ylim(-0.001, 0.001)
        # ax_3d.set_zlim(-0.001, 0.001)

        ax_err.cla()
        ax_err.plot(time_vec[0:n], error[0, 0, 0:n])
        ax_err.plot(time_vec[0:n], error[0, 1, 0:n])
        ax_err.plot(time_vec[0:n], error[0, 2, 0:n])
        ax_err.plot(time_vec[0:n], error[0, 3, 0:n])
        ax_err.set_xlabel('Time')
        ax_err.set_ylabel('Error')
        ax_err.legend(['x', 'y', 'z', 'psi'])
        ax_err.set_title('Error')
        ax_err.set_ylim(-0.5, 0.5)

        ax_zd.cla()
        ax_zd.plot(time_vec[0:n], state[3, 0:n] * 180/np.pi)
        ax_zd.plot(time_vec[0:n], state[4, 0:n] * 180/np.pi)
        # ax_zd.plot(time_vec[0:n], state[9, 0:n])
        # ax_zd.plot(time_vec[0:n], state[10, 0:n])
        # ax_zd.legend(['phi', 'theta', 'wx', 'wy'])
        ax_zd.set_ylim(-20, 20)
        ax_zd.legend(['phi', 'theta'])
        ax_zd.set_title('Pitch/roll')

        plt.draw()
        plt.pause(0.001)
        
except KeyboardInterrupt:
    plt.draw()
    plt.pause(0.01)
    plt.show(block=True)

plt.show(block=True)