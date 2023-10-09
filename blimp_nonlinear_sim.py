import numpy as np
import matplotlib.pyplot as plt
import time
from rta.blimp import Blimp
import random
from BlimpTrackingGurobi import BlimpTrackingGurobi
import scipy
import csv
import sys
import os

if len(sys.argv) < 2:
    print("Please run with output file name as first argument")

outfile = sys.argv[1]

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

# Center of gravity to center of buoyancy
r_zgb_b = 0.08705
r_gb_b = np.array([0, 0, r_zgb_b]).T
r_tzb_B = 0.13 - r_zgb_b

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

M_RB_CB = H(r_gb_b).T @ M_RB_CG @ H(r_gb_b)

M_CB = M_RB_CB + M_A_CB

g_acc = 9.8
fg_n = m_RB * np.array([0, 0, g_acc]).T

## Aerodynamic damping
D_vx_CB = 0.0115
D_vy_CB = D_vx_CB
D_vz_CB = 0.0480
D_wx_CB = 0.00980
D_wy_CB = D_wx_CB
D_wz_CB = D_wx_CB

D_CB = np.diag([D_vx_CB, D_vy_CB, D_vz_CB, D_wx_CB, D_wy_CB, D_wz_CB])

## Simulation

x0 = 0
y0 = 0
z0 = 0

phi0 = 0
theta0 = 0
psi0 = 0

v_x0 = 0.0
v_y0 = 0.0
v_z0 = 0.0

w_x0 = 0
w_y0 = 0
w_z0 = 0

eta_bn_n = np.array([[x0, y0, z0, phi0, theta0, psi0]]).T
nu_bn_b = np.array([[v_x0, v_y0, v_z0, w_x0, w_y0, w_z0]]).T

ax = plt.figure().add_subplot(projection='3d')
plt.ion()
ax.grid()

past_x_vals = []
past_y_vals = []
past_z_vals = []

# Controls

dT = 0.005

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

Cm = np.matrix([[0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
               [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
               [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0]])
D = np.zeros((3, 4))

P = np.identity(3) * 1
Q = np.identity(3) * 1
R = np.identity(4) * 10

N = 50

def distance_to_goal(state, goal):
    return np.linalg.norm(state - goal)

xmin = np.matrix([[-np.inf],
                  [-np.inf],
                  [-np.inf],
                  [-np.inf],
                  [-np.inf],
                  [-np.inf],
                  [-np.inf],   # x
                  [-np.inf],   # y
                  [-np.inf],   # z
                  [-np.inf],   # phi
                  [-np.inf],   # theta
                  [-np.inf]    # psi
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
                  [np.inf],   # phi
                  [np.inf],   # theta
                  [np.inf]    # psi
                  ])

umin = np.matrix([[-10],
                  [-10],
                  [-10],
                  [-np.inf]])

umax = np.matrix([[10],
                  [10],
                  [10],
                  [np.inf]])

blimp_controller = BlimpTrackingGurobi(A_dis,
                                       B_dis,
                                       Cm,
                                       D,
                                       P,
                                       Q,
                                       R,
                                       N,
                                       xmin,
                                       xmax,
                                       umin,
                                       umax)

DEADBAND = 0.75

TIMESTEPS_TO_SETTLE = 10
settling_timer = TIMESTEPS_TO_SETTLE

ref_idx = 0
ref_distance = 5
reference_points = [
                    np.array([ref_distance, ref_distance, ref_distance]),
                    np.array([ref_distance, -ref_distance, -ref_distance]),
                    np.array([-ref_distance, ref_distance, -ref_distance]),
                    np.array([-ref_distance, ref_distance, ref_distance])
                    ]

ref_vals_x = []
ref_vals_y = []
ref_vals_z = []

NUM_REF_PTS = 4

done = False
t = 0

time_vec = []
error_vals = []

u0_vals = []
u1_vals = []
u2_vals = []
u3_vals = []

times = []

TIMEOUT = np.inf

try:
    while not done:
        print()
        print("Time: " + str(t))

        x = eta_bn_n[0]
        y = eta_bn_n[1]
        z = eta_bn_n[2]

        phi = eta_bn_n[3]
        theta = eta_bn_n[4]
        psi = eta_bn_n[5]

        v_x = nu_bn_b[0]
        v_y = nu_bn_b[1]
        v_z = nu_bn_b[2]

        w_x = nu_bn_b[3]
        w_y = nu_bn_b[4]
        w_z = nu_bn_b[5]

        # Compute control action
        start_time = time.time()
        u = blimp_controller.get_tracking_ctrl(np.block([[nu_bn_b], [eta_bn_n]]),
                                              reference_points[ref_idx])
        times.append(time.time() - start_time)

        u0_vals.append(u[0].item())
        u1_vals.append(u[1].item())
        u2_vals.append(u[2].item())
        u3_vals.append(u[3].item())

        tau_B = np.array([[u[0], u[1], u[2], -r_tzb_B * u[1], r_tzb_B * u[0], u[3]]]).T

        # Restoration torque
        fg_B = np.linalg.inv(R_b__n(phi, theta, psi)) @ fg_n
        g_CB = -np.block([[np.zeros((3, 1))],
                        [np.reshape(np.cross(r_gb_b, fg_B), (3, 1))]])

        # Update state
        eta_bn_n_dot = np.block([[R_b__n(phi, theta, psi),    np.zeros((3, 3))],
                                [np.zeros((3, 3)),            T(phi, theta)]]) @ nu_bn_b
        
        nu_bn_b_dot = np.reshape(-np.linalg.inv(M_CB) @ (C(M_RB_CB + M_A_CB, nu_bn_b) @ nu_bn_b + \
                            D_CB @ nu_bn_b + g_CB - tau_B), (6, 1))
        
        eta_bn_n = eta_bn_n + eta_bn_n_dot * dT
        nu_bn_b = nu_bn_b + nu_bn_b_dot * dT

        print("Inputs:")
        print("Fx: " + str(u[0]))
        print("Fy: " + str(u[1]))
        print("Fz: " + str(u[2]))
        print("Tz: " + str(u[3]))
        print()

        print("Position:")
        print("X: " + str(eta_bn_n[0]))
        print("Y: " + str(eta_bn_n[1]))
        print("Z: " + str(eta_bn_n[2]))
        print("Phi: " + str(eta_bn_n[3]))
        print("Theta: " + str(eta_bn_n[4]))
        print("Psi: " + str(eta_bn_n[5]))
        print()
        
        print("Rates:")
        print("X: " + str(nu_bn_b[0]))
        print("Y: " + str(nu_bn_b[1]))
        print("Z: " + str(nu_bn_b[2]))
        print("Phi: " + str(nu_bn_b[3]))
        print("Theta: " + str(nu_bn_b[4]))
        print("Psi: " + str(nu_bn_b[5]))
        
        past_x_vals.append(eta_bn_n[0].item())
        past_y_vals.append(eta_bn_n[1].item())
        past_z_vals.append(eta_bn_n[2].item())

        ref_vals_x.append(reference_points[ref_idx][0])
        ref_vals_y.append(reference_points[ref_idx][1])
        ref_vals_z.append(reference_points[ref_idx][2])

        ax.cla()
        ax.scatter(past_x_vals, past_y_vals, past_z_vals, color='blue')
        ax.scatter(eta_bn_n[0], eta_bn_n[1], eta_bn_n[2], color='m', s=200)
        ax.scatter(ref_vals_x, ref_vals_y, ref_vals_z,
                   color='r', s=100)
        ax.invert_yaxis()
        ax.invert_zaxis()

        x_span = ax.get_xlim()[1] - ax.get_xlim()[0]
        y_span = ax.get_ylim()[1] - ax.get_ylim()[0]
        z_span = ax.get_zlim()[1] - ax.get_zlim()[0]
        blimp_x_vector_scaling = abs(min(x_span,
                                         y_span,
                                         z_span)) * 0.2
        
        
        blimp_x_vector = R_b__n(eta_bn_n[3], eta_bn_n[4], eta_bn_n[5]) \
                        @ np.array([blimp_x_vector_scaling, 0, 0]).T
    
        q = ax.quiver(eta_bn_n[0], eta_bn_n[1], eta_bn_n[2], \
                blimp_x_vector[0], blimp_x_vector[1], blimp_x_vector[2], \
                color='magenta')
        q.ShowArrowHead = 'on'

        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')

        plt.draw()
        plt.pause(0.001)
        
        current_position = np.block([eta_bn_n[0], eta_bn_n[1], eta_bn_n[2]])

        error = distance_to_goal(current_position,
                                 reference_points[ref_idx])
        error_vals.append(error)

        print("Reference: " + str(reference_points[ref_idx]))
        print("Current: " + str(current_position))
        print("Error: " + str(error))

        if error < DEADBAND:
            print(settling_timer)
            settling_timer -= 1
            if settling_timer == 0:
                settling_timer = TIMESTEPS_TO_SETTLE
                ref_idx = ref_idx + 1
                print("====================================")
                print("UPDATING REFERENCE")
                print("====================================")
                if ref_idx == 4:
                    done = True
        else:
            settling_timer = TIMESTEPS_TO_SETTLE

        time_vec.append(t)
        t = t + dT

        if t > TIMEOUT:
            done = True
        
except KeyboardInterrupt:
    ax.cla()
    ax.scatter(past_x_vals, past_y_vals, past_z_vals, color='blue')
    ax.scatter(eta_bn_n[0], eta_bn_n[1], eta_bn_n[2], color='m', s=200)
    ax.scatter(ref_vals_x, ref_vals_y, ref_vals_z,
               color='r', s=100)
    ax.invert_yaxis()
    ax.invert_zaxis()
    
    x_span = ax.get_xlim()[1] - ax.get_xlim()[0]
    y_span = ax.get_ylim()[1] - ax.get_ylim()[0]
    z_span = ax.get_zlim()[1] - ax.get_zlim()[0]
    blimp_x_vector_scaling = min(x_span,
                                    y_span,
                                    z_span) * 0.2
    blimp_x_vector = R_b__n(eta_bn_n[3], eta_bn_n[4], eta_bn_n[5]) \
                        @ np.array([blimp_x_vector_scaling, 0, 0]).T

    q = ax.quiver(eta_bn_n[0], eta_bn_n[1], eta_bn_n[2], \
            blimp_x_vector[0], blimp_x_vector[1], blimp_x_vector[2], \
            color='magenta')
    q.ShowArrowHead = 'on'

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')

    plt.draw()
    plt.pause(0.001)
        


with open('outputs/' + outfile, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile, delimiter=',')

    print("Writing to CSV...")
    writer.writerow(['Time',
                     'X', 'Y', 'Z',
                     'Reference X', 'Reference Y', 'Reference Z',
                     'Error',
                     'u0', 'u1', 'u2', 'u3',
                     'deltaT'])
    
    for i in range(len(time_vec)):
        print("Row " + str(i) + " out of " + str(len(time_vec)))
        writer.writerow([time_vec[i],
                         past_x_vals[i], past_y_vals[i], past_z_vals[i],
                         ref_vals_x[i], ref_vals_y[i], ref_vals_z[i],
                         error_vals[i],
                         u0_vals[i], u1_vals[i], u2_vals[i], u3_vals[i],
                         times[i]])
        
    print("Done writing to CSV!")

plt.show(block=True)