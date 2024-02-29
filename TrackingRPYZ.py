from BlimpController import BlimpController
import numpy as np
import control
from parameters import *
import sys

class TrackingRPYZ(BlimpController):

    def __init__(self, dT, skip_derivatives=True):
        super().__init__(dT)
                
        # Time
        TRACKING_TIME = 120

        tracking_time = np.arange(0, TRACKING_TIME, dT)

        # Trajectory definition
        self.At = 1

        self.traj_z = np.zeros(len(tracking_time) + 1)
        self.traj_phi = np.zeros(len(tracking_time) + 1)
        self.traj_theta = np.zeros(len(tracking_time) + 1)
        self.traj_psi = np.zeros(len(tracking_time) + 1)

        self.traj_z_dot = np.zeros(len(tracking_time) + 1)
        self.traj_phi_dot = np.zeros(len(tracking_time) + 1)
        self.traj_theta_dot = np.zeros(len(tracking_time) + 1)
        self.traj_psi_dot = np.zeros(len(tracking_time) + 1)

        self.traj_z_ddot = np.zeros(len(tracking_time) + 1)
        self.traj_phi_ddot = np.zeros(len(tracking_time) + 1)
        self.traj_theta_ddot = np.zeros(len(tracking_time) + 1)
        self.traj_psi_ddot = np.zeros(len(tracking_time) + 1)

    def init_sim(self, sim):
        sim.set_var('theta', np.pi/10)
        sim.set_var('psi', np.pi/10)
        sim.set_var('phi', np.pi/10)
        sim.set_var('z', 2)

    def get_ctrl_action(self, sim):

        sim.start_timer()

        n = sim.get_current_timestep()

        # Extract state variables
        z = sim.get_var('z')

        phi = sim.get_var('phi')
        theta = sim.get_var('theta')
        psi = sim.get_var('psi')
        
        w_x__b = sim.get_var('wx')
        w_y__b = sim.get_var('wy')
        
        z_dot = sim.get_var_dot('z')

        phi_dot = sim.get_var_dot('phi')
        theta_dot = sim.get_var_dot('theta')
        psi_dot = sim.get_var_dot('psi')

        ## Feedback-linearized tracking controller

        print("         [z\tphi\ttheta\tpsi]")
        print("States: " + str(np.array([z, phi, theta, psi]).reshape((1,4))))
        print("State_dot: " + str(np.array([z_dot, phi_dot, theta_dot, psi_dot]).reshape((1,4))))

        # Compute input to integrator chain
        A = np.array(
            [                    
                                                                                                                                                                                w_y__b*((m_RB*r_z_gb__b*np.cos(phi)*np.cos(theta)*w_y__b)/m_z - (D_omega_xy__CB*m_RB*r_z_gb__b*np.sin(theta))/(- m_RB**2*r_z_gb__b**2 + I_y*m_x) + (I_z*m_RB*r_z_gb__b*np.cos(theta)*np.sin(phi)*np.tan(phi)*w_y__b)/(- m_RB**2*r_z_gb__b**2 + I_x*m_y)) - w_x__b*((D_omega_xy__CB*m_RB*r_z_gb__b*np.cos(theta)*np.sin(phi))/(- m_RB**2*r_z_gb__b**2 + I_x*m_y) - (m_RB*r_z_gb__b*np.cos(phi)*np.cos(theta)*w_x__b)/m_z + (I_z*m_RB*r_z_gb__b*np.tan(phi)*np.sin(theta)*w_y__b)/(- m_RB**2*r_z_gb__b**2 + I_y*m_x)) + np.tan(phi)*w_y__b*((np.sin(theta)*(I_x*m_RB*r_z_gb__b*w_x__b - I_y*m_RB*r_z_gb__b*w_x__b))/(- m_RB**2*r_z_gb__b**2 + I_y*m_x) + np.cos(theta)*np.sin(phi)*((I_x*m_RB*r_z_gb__b*w_y__b)/(- m_RB**2*r_z_gb__b**2 + I_x*m_y) - (I_y*m_RB*r_z_gb__b*w_y__b)/(- m_RB**2*r_z_gb__b**2 + I_x*m_y))) - (f_g*m_RB*r_z_gb__b**2*np.sin(theta)**2)/(- m_RB**2*r_z_gb__b**2 + I_y*m_x) + (f_g*m_RB*r_z_gb__b**2*np.sin(phi)**2*(np.sin(theta)**2 - 1))/(- m_RB**2*r_z_gb__b**2 + I_x*m_y),
    w_y__b*(np.sin(phi)*(np.tan(theta)**2 + 1)*theta_dot + np.cos(phi)*np.tan(theta)*phi_dot + (I_z*m_y*np.tan(phi)*w_y__b)/(- m_RB**2*r_z_gb__b**2 + I_x*m_y) - (D_omega_xy__CB*m_x*np.sin(phi)*np.tan(theta))/(- m_RB**2*r_z_gb__b**2 + I_y*m_x) + (I_x*np.cos(phi)*np.tan(theta)*w_x__b)/I_z) - w_x__b*((D_omega_xy__CB*m_y)/(- m_RB**2*r_z_gb__b**2 + I_x*m_y) + (I_y*np.cos(phi)*np.tan(theta)*w_y__b)/I_z + (I_z*m_x*np.sin(phi)*np.tan(phi)*np.tan(theta)*w_y__b)/(- m_RB**2*r_z_gb__b**2 + I_y*m_x)) - np.tan(phi)*w_y__b*(np.sin(phi)*np.tan(theta)*((m_RB**2*r_z_gb__b**2*w_x__b)/(- m_RB**2*r_z_gb__b**2 + I_y*m_x) - (I_x*m_x*w_x__b)/(- m_RB**2*r_z_gb__b**2 + I_y*m_x)) + np.cos(phi)*(np.tan(theta)**2 + 1)*theta_dot - np.sin(phi)*np.tan(theta)*phi_dot - (m_RB**2*r_z_gb__b**2*w_y__b)/(- m_RB**2*r_z_gb__b**2 + I_x*m_y) + (I_y*m_y*w_y__b)/(- m_RB**2*r_z_gb__b**2 + I_x*m_y) - (D_omega_z__CB*np.cos(phi)*np.tan(theta))/I_z) - (f_g*m_y*r_z_gb__b*np.cos(theta)*np.sin(phi))/(- m_RB**2*r_z_gb__b**2 + I_x*m_y) - (f_g*m_x*r_z_gb__b*(np.sin(phi) - np.cos(theta)**2*np.sin(phi)))/(np.cos(theta)*(- m_RB**2*r_z_gb__b**2 + I_y*m_x)),
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        w_x__b*((I_y*np.sin(phi)*w_y__b)/I_z - (I_z*m_x*np.cos(phi)*np.tan(phi)*w_y__b)/(- m_RB**2*r_z_gb__b**2 + I_y*m_x)) - w_y__b*(np.sin(phi)*phi_dot + (D_omega_xy__CB*m_x*np.cos(phi))/(- m_RB**2*r_z_gb__b**2 + I_y*m_x) + (I_x*np.sin(phi)*w_x__b)/I_z) + np.tan(phi)*w_y__b*(np.cos(phi)*phi_dot - (D_omega_z__CB*np.sin(phi))/I_z + (np.cos(phi)*(- w_x__b*m_RB**2*r_z_gb__b**2 + I_x*m_x*w_x__b))/(- m_RB**2*r_z_gb__b**2 + I_y*m_x)) - (f_g*m_x*r_z_gb__b*np.cos(phi)*np.sin(theta))/(- m_RB**2*r_z_gb__b**2 + I_y*m_x),
                                                                                                                                                                                                                                                                                                                                                w_y__b*((np.cos(phi)*phi_dot)/np.cos(theta) - (np.sin(phi)*np.sin(theta)*theta_dot)/(np.sin(theta)**2 - 1) + (I_x*np.cos(phi)*w_x__b)/(I_z*np.cos(theta)) - (D_omega_xy__CB*m_x*np.sin(phi))/(np.cos(theta)*(- m_RB**2*r_z_gb__b**2 + I_y*m_x))) - w_x__b*((I_y*np.cos(phi)*w_y__b)/(I_z*np.cos(theta)) + (I_z*m_x*np.sin(phi)*np.tan(phi)*w_y__b)/(np.cos(theta)*(- m_RB**2*r_z_gb__b**2 + I_y*m_x))) + np.tan(phi)*w_y__b*((np.sin(phi)*phi_dot)/np.cos(theta) + (D_omega_z__CB*np.cos(phi))/(I_z*np.cos(theta)) - (np.cos(phi)*np.sin(theta)*theta_dot)/np.cos(theta)**2 + (np.sin(phi)*(- w_x__b*m_RB**2*r_z_gb__b**2 + I_x*m_x*w_x__b))/(np.cos(theta)*(- m_RB**2*r_z_gb__b**2 + I_y*m_x))) - (f_g*m_x*r_z_gb__b*np.sin(phi)*np.sin(theta))/(np.cos(theta)*(- m_RB**2*r_z_gb__b**2 + I_y*m_x))
    
            ]
        ).reshape((4,1))

        print("A: " + str(A))

        Binv = np.array([
            [                              0,                                                                                                  0,                       -(np.cos(phi)*(- m_RB**2*r_z_gb__b**2 + I_y*m_x))/(m_RB*r_z_gb__b - m_x*r_z_tg__b),                                                                                                                                                                      -(np.cos(theta)*np.sin(phi)*(- m_RB**2*r_z_gb__b**2 + I_y*m_x))/(m_RB*r_z_gb__b - m_x*r_z_tg__b)],
            [                              0,                                  (- m_RB**2*r_z_gb__b**2 + I_x*m_y)/(m_RB*r_z_gb__b - m_y*r_z_tg__b),                                                                                                      0,                                                                                                                                                                                  -(np.sin(theta)*(- m_RB**2*r_z_gb__b**2 + I_x*m_y))/(m_RB*r_z_gb__b - m_y*r_z_tg__b)],
            [m_z/(np.cos(phi)*np.cos(theta)), -(m_z*np.sin(phi)*(I_x - m_RB*r_z_gb__b*r_z_tg__b))/(np.cos(phi)*(m_RB*r_z_gb__b - m_y*r_z_tg__b)), -(m_z*np.sin(theta)*(I_y - m_RB*r_z_gb__b*r_z_tg__b))/(np.cos(theta)*(m_RB*r_z_gb__b - m_x*r_z_tg__b)), (m_z*np.sin(phi)*np.sin(theta)*(I_x*m_RB*r_z_gb__b - I_y*m_RB*r_z_gb__b - I_x*m_x*r_z_tg__b + I_y*m_y*r_z_tg__b + m_RB*m_x*r_z_gb__b*r_z_tg__b**2 - m_RB*m_y*r_z_gb__b*r_z_tg__b**2))/(np.cos(phi)*(m_RB*r_z_gb__b - m_x*r_z_tg__b)*(m_RB*r_z_gb__b - m_y*r_z_tg__b))],
            [                              0,                                                                                                  0,                                                                                       -I_z*np.sin(phi),                                                                                                                                                                                                                                       I_z*np.cos(phi)*np.cos(theta)],
        ])

        print("Binv: " + str(Binv))

        zeta1 = np.array([[z],
                          [phi],
                          [theta],
                          [psi]])
        zeta2 = np.array([[z_dot],
                          [phi_dot],
                          [theta_dot],
                          [psi_dot]])

        yd = np.array([[self.traj_z[n]],
                       [self.traj_phi[n]],
                       [self.traj_theta[n]],
                       [self.traj_psi[n]]])
        yd_dot = np.array([[self.traj_z_dot[n]],
                           [self.traj_phi_dot[n]],
                           [self.traj_theta_dot[n]],
                           [self.traj_psi_dot[n]]])
        yd_ddot = np.array([[self.traj_z_ddot[n]],
                            [self.traj_phi_ddot[n]],
                            [self.traj_theta_ddot[n]],
                            [self.traj_psi_ddot[n]]])
        
        e1 = zeta1 - yd
        e2 = zeta2 - yd_dot

        k1 = 10
        k2 = 10

        q = -k1 * e1 - k2 * e2 + yd_ddot

        u_traj = Binv @ (q - A)

        u = u_traj

        print("q: " + str(q.reshape((1,4))))

        print("  [x\ty\tz\tpsi]")
        print("u: " + str(u.reshape(1,4)))

        if (np.isnan(u).any()):
            print("nan detected in input")
            sys.exit(0)

        print()

        sim.end_timer()

        return u
    
    def get_trajectory(self):
        return (self.traj_phi, self.traj_theta, self.traj_psi)
    
    def get_error(self, sim):
        n = sim.get_current_timestep() + 1
        return np.array([
                sim.get_var_history('z') - self.traj_z[0:n],
                sim.get_var_history('phi') - self.traj_phi[0:n],
                sim.get_var_history('theta') - self.traj_theta[0:n],
                sim.get_var_history('psi') - self.traj_psi[0:n]
        ]).T
    