import numpy as np
import matplotlib.pyplot as plt
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

def T(phi, theta):
    phi = phi.item()
    theta = theta.item()

    return np.array([[1,     np.sin(phi)*np.tan(theta),      np.cos(phi)*np.tan(theta)],
                     [0,          np.cos(phi),                   -np.sin(phi)],
                     [0,     np.sin(phi)/np.cos(theta),      np.cos(phi)/np.cos(theta)]])

## System parameters

N = 12

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
dT = 0.01
TIMEOUT = 20

time_vec = np.arange(0, TIMEOUT, dT)

# Initial conditions
x0 = 1
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

# State vector
eta_bn_n = np.array([[x0, y0, z0, phi0, theta0, psi0]]).T
nu_bn_b = np.array([[v_x0, v_y0, v_z0, w_x0, w_y0, w_z0]]).T

state = np.empty((N, len(time_vec)))
state_dot = np.empty((N, len(time_vec)))

state[:, 0] = np.vstack((eta_bn_n, nu_bn_b)).reshape(N)
state_dot[:, 0] = np.zeros(N)

error = np.empty((2, 4, len(time_vec)))

# Trajectory definition
f = 0.05

traj_x = np.cos(2*np.pi*f*time_vec)
traj_y = np.sin(2*np.pi*f*time_vec)
traj_z = time_vec * -1/10
traj_psi = psi0 + 2*np.pi*f*time_vec

traj_x_dot = -2*np.pi*f*np.sin(2*np.pi*f*time_vec)
traj_y_dot = 2*np.pi*f*np.cos(2*np.pi*f*time_vec)
traj_z_dot = -1/10 * np.ones(len(time_vec))
traj_psi_dot = 2*np.pi*f * np.ones(len(time_vec))

traj_x_ddot = -(2*np.pi*f)**2*np.cos(2*np.pi*f*time_vec)
traj_y_ddot = -(2*np.pi*f)**2*np.sin(2*np.pi*f*time_vec)
traj_z_ddot = np.zeros(len(time_vec))
traj_psi_ddot = np.zeros(len(time_vec))

# Set up figure

fig = plt.figure()
ax_3d = fig.add_subplot(131, projection='3d')
plt.ion()
ax_3d.grid()

ax_err = fig.add_subplot(132)

ax_zd = fig.add_subplot(133)

plt.subplots_adjust(wspace=0.5)

try:
    for n in range(len(time_vec)):
        # print()
        # print("Time: " + str(time_vec[n]))

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

        u = Binv @ (q - A)

        tau_B = np.array([u[0], u[1], u[2], -r_z_tg__b * u[1], r_z_tg__b * u[0], u[3]])
        
        # Restoration torque
        fg_B = np.linalg.inv(R_b__n(phi, theta, psi)) @ fg_n
        g_CB = -np.block([[np.zeros((3, 1))],
                        [np.reshape(np.cross(r_gb__b, fg_B), (3, 1))]])

        # Update state
        eta_bn_n_dot = np.block([[R_b__n(phi, theta, psi),    np.zeros((3, 3))],
                                [np.zeros((3, 3)),            T(phi, theta)]]) @ nu_bn_b
        
        nu_bn_b_dot = np.reshape(-np.linalg.inv(M_CB) @ (C(M_RB_CB + M_A_CB, nu_bn_b) @ nu_bn_b + \
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

        ax_err.cla()
        ax_err.plot(time_vec[0:n], error[0, 0, 0:n])
        ax_err.plot(time_vec[0:n], error[0, 1, 0:n])
        ax_err.plot(time_vec[0:n], error[0, 2, 0:n])
        ax_err.plot(time_vec[0:n], error[0, 3, 0:n])
        ax_err.set_xlabel('Time')
        ax_err.set_ylabel('Error')
        ax_err.legend(['x', 'y', 'z', 'psi'])
        ax_err.set_title('Error')
        ax_err.set_aspect(1/ax_err.get_data_ratio())

        ax_zd.cla()
        ax_zd.plot(time_vec[0:n], state[3, 0:n])
        ax_zd.plot(time_vec[0:n], state[4, 0:n])
        ax_zd.plot(time_vec[0:n], state[9, 0:n])
        ax_zd.plot(time_vec[0:n], state[10, 0:n])
        ax_zd.legend(['phi', 'theta', 'wx', 'wy'])
        ax_zd.set_title('Pitch/roll')
        ax_zd.set_aspect(1/ax_zd.get_data_ratio())

        plt.draw()
        plt.pause(0.001)
        
except KeyboardInterrupt:
    plt.draw()
    plt.pause(0.01)
    plt.show(block=True)