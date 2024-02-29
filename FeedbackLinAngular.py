from BlimpController import BlimpController
import numpy as np
import control
from PID import PID
from parameters import *
import sys

class FeedbackLinAngular(BlimpController):

    def __init__(self, dT, skip_derivatives=True):
        super().__init__(dT)
                
        # Time
        TRACKING_TIME = 20
        SETTLE_TIME = 100

        tracking_time = np.arange(0, TRACKING_TIME + SETTLE_TIME, dT)

        # Trajectory definition
        f = 0.05
        self.At = 1

        # self.traj_x = np.concatenate((self.At * np.cos(2*np.pi*f*tracking_time), self.At*np.ones(len(settle_time))))
        # self.traj_y = np.concatenate((self.At * np.sin(2*np.pi*f*tracking_time), np.zeros(len(settle_time))))
        # self.traj_z = np.concatenate((tracking_time * -1/10, -TRACKING_TIME * 1/10 * np.ones(len(settle_time))))
        # self.traj_psi = np.concatenate((psi0 + 2*np.pi*f*tracking_time, (psi0 + 2*np.pi) * np.ones(len(settle_time))))

        # self.traj_x_dot = np.concatenate((-2*np.pi*f*self.At*np.sin(2*np.pi*f*tracking_time), np.zeros(len(settle_time))))
        # self.traj_y_dot = np.concatenate((2*np.pi*f*self.At*np.cos(2*np.pi*f*tracking_time), np.zeros(len(settle_time))))
        # self.traj_z_dot = np.concatenate((-1/10 * np.ones(len(tracking_time)), np.zeros(len(settle_time))))
        # self.traj_psi_dot = np.concatenate((2*np.pi*f * np.ones(len(tracking_time)), np.zeros(len(settle_time))))

        # self.traj_x_ddot = np.concatenate((-(2*np.pi*f)**2*self.At*np.cos(2*np.pi*f*tracking_time), np.zeros(len(settle_time))))
        # self.traj_y_ddot = np.concatenate((-(2*np.pi*f)**2*self.At*np.sin(2*np.pi*f*tracking_time), np.zeros(len(settle_time))))
        # self.traj_z_ddot = np.concatenate((np.zeros(len(tracking_time)), np.zeros(len(settle_time))))
        # self.traj_psi_ddot = np.concatenate((np.zeros(len(tracking_time)), np.zeros(len(settle_time))))

        self.traj_x = np.zeros(len(tracking_time) + 1)
        self.traj_y = np.zeros(len(tracking_time) + 1)
        self.traj_z = np.zeros(len(tracking_time) + 1)
        self.traj_psi = np.zeros(len(tracking_time) + 1)

        self.traj_x_dot = np.zeros(len(tracking_time) + 1)
        self.traj_y_dot = np.zeros(len(tracking_time) + 1)
        self.traj_z_dot = np.zeros(len(tracking_time) + 1)
        self.traj_psi_dot = np.zeros(len(tracking_time) + 1)
        
        self.traj_x_ddot = np.zeros(len(tracking_time) + 1)
        self.traj_y_ddot = np.zeros(len(tracking_time) + 1)
        self.traj_z_ddot = np.zeros(len(tracking_time) + 1)
        self.traj_psi_ddot = np.zeros(len(tracking_time) + 1)

        # virtual input controllers

        Kpw = 0.01
        Kiw = 0.01
        Kdw = 0.01

        Kpz = 0.01
        Kiz = 0.01
        Kdz = 0.01

        self.wx_dot_pid = PID(Kpw, Kiw, Kdw, dT)
        self.wy_dot_pid = PID(Kpw, Kiw, Kdw, dT)
        self.wz_dot_pid = PID(Kpw, Kiw, Kdw, dT)
        self.vz_dot_pid = PID(Kpz, Kiz, Kdz, dT)

        self.prev_tau = np.zeros((4,1))

    def init_sim(self, sim):
        sim.set_var('x', self.At)

    def get_ctrl_action(self, sim):

        sim.start_timer()

        n = sim.get_current_timestep()

        # Extract state variables
        x = sim.get_var('x')
        y = sim.get_var('y')
        z = sim.get_var('z')

        phi = sim.get_var('phi')
        theta = sim.get_var('theta')
        psi = sim.get_var('psi')
        
        v_x__b = sim.get_var('vx')
        v_y__b = sim.get_var('vy')
        v_z__b = sim.get_var('vz')

        w_x__b = sim.get_var('wx')
        w_y__b = sim.get_var('wy')
        w_z__b = sim.get_var('wz')

        x_dot = sim.get_var_dot('x')
        y_dot = sim.get_var_dot('y')
        z_dot = sim.get_var_dot('z')

        phi_dot = sim.get_var_dot('phi')
        theta_dot = sim.get_var_dot('theta')
        psi_dot = sim.get_var_dot('psi')

        v_x_dot = sim.get_var_dot('vx')
        v_y_dot = sim.get_var_dot('vy')
        v_z_dot = sim.get_var_dot('vz')

        w_x_dot = sim.get_var_dot('wx')
        w_y_dot = sim.get_var_dot('wy')
        w_z_dot = sim.get_var_dot('wz')
        
        f_x__b = self.prev_tau[0].item()
        f_y__b = self.prev_tau[1].item()
        f_z__b = self.prev_tau[2].item()
        tau_z__b = self.prev_tau[3].item()

        ## Feedback-linearized tracking controller

        # Compute input to integrator chain
        A = np.array([[v_z__b*(np.cos(phi)*np.sin(psi)*phi_dot + np.cos(psi)*np.sin(phi)*psi_dot - np.cos(psi)*np.sin(phi)*np.sin(theta)*phi_dot - np.cos(phi)*np.sin(psi)*np.sin(theta)*psi_dot + np.cos(phi)*np.cos(psi)*np.cos(theta)*theta_dot) - v_x__b*(np.cos(theta)*np.sin(psi)*psi_dot + np.cos(psi)*np.sin(theta)*theta_dot) + v_y__b*(- np.cos(phi)*np.cos(psi)*psi_dot + np.sin(phi)*np.sin(psi)*phi_dot + np.cos(phi)*np.cos(psi)*np.sin(theta)*phi_dot + np.cos(psi)*np.cos(theta)*np.sin(phi)*theta_dot - np.sin(phi)*np.sin(psi)*np.sin(theta)*psi_dot) - ((np.cos(phi)*np.sin(psi) - np.cos(psi)*np.sin(phi)*np.sin(theta))*(f_y__b - w_z__b*(m_Axy*v_x__b + m_RB*v_x__b + m_RB*r_z_gb__b*w_y__b) - D_vxy__CB*v_y__b + v_z__b*w_x__b*(m_Az + m_RB)))/(m_Axy + m_RB) + (np.cos(psi)*np.cos(theta)*(f_x__b + w_z__b*(m_Axy*v_y__b + m_RB*v_y__b - m_RB*r_z_gb__b*w_x__b) - D_vxy__CB*v_x__b - v_z__b*w_y__b*(m_Az + m_RB)))/(m_Axy + m_RB)],
                      [v_x__b*(np.cos(psi)*np.cos(theta)*psi_dot - np.sin(psi)*np.sin(theta)*theta_dot) + v_z__b*(- np.cos(phi)*np.cos(psi)*phi_dot + np.sin(phi)*np.sin(psi)*psi_dot + np.cos(phi)*np.cos(psi)*np.sin(theta)*psi_dot + np.cos(phi)*np.cos(theta)*np.sin(psi)*theta_dot - np.sin(phi)*np.sin(psi)*np.sin(theta)*phi_dot) + v_y__b*(- np.cos(psi)*np.sin(phi)*phi_dot - np.cos(phi)*np.sin(psi)*psi_dot + np.cos(phi)*np.sin(psi)*np.sin(theta)*phi_dot + np.cos(psi)*np.sin(phi)*np.sin(theta)*psi_dot + np.cos(theta)*np.sin(phi)*np.sin(psi)*theta_dot) + ((np.cos(phi)*np.cos(psi) + np.sin(phi)*np.sin(psi)*np.sin(theta))*(f_y__b - w_z__b*(m_Axy*v_x__b + m_RB*v_x__b + m_RB*r_z_gb__b*w_y__b) - D_vxy__CB*v_y__b + v_z__b*w_x__b*(m_Az + m_RB)))/(m_Axy + m_RB) + (np.cos(theta)*np.sin(psi)*(f_x__b + w_z__b*(m_Axy*v_y__b + m_RB*v_y__b - m_RB*r_z_gb__b*w_x__b) - D_vxy__CB*v_x__b - v_z__b*w_y__b*(m_Az + m_RB)))/(m_Axy + m_RB)],
                      [v_y__b*(np.cos(phi)*np.cos(theta)*phi_dot - np.sin(phi)*np.sin(theta)*theta_dot) - v_z__b*(np.cos(theta)*np.sin(phi)*phi_dot + np.cos(phi)*np.sin(theta)*theta_dot) - np.cos(theta)*v_x__b*theta_dot - (np.sin(theta)*(f_x__b + w_z__b*(m_Axy*v_y__b + m_RB*v_y__b - m_RB*r_z_gb__b*w_x__b) - D_vxy__CB*v_x__b - v_z__b*w_y__b*(m_Az + m_RB)))/(m_Axy + m_RB) + (np.cos(theta)*np.sin(phi)*(f_y__b - w_z__b*(m_Axy*v_x__b + m_RB*v_x__b + m_RB*r_z_gb__b*w_y__b) - D_vxy__CB*v_y__b + v_z__b*w_x__b*(m_Az + m_RB)))/(m_Axy + m_RB)],
                    [(np.cos(phi)*np.cos(theta)*w_y__b*phi_dot - np.cos(theta)*np.sin(phi)*w_z__b*phi_dot + np.cos(phi)*np.sin(theta)*w_z__b*theta_dot + np.sin(phi)*np.sin(theta)*w_y__b*theta_dot)/np.cos(theta)**2]
        ])
        
        Binv = np.array([[-((m_Axy + m_RB)*(np.cos(phi)*np.sin(psi) - np.cos(psi)*np.sin(phi)*np.sin(theta)))/(m_RB*r_z_gb__b), ((m_Axy + m_RB)*(np.cos(phi)*np.cos(psi) + np.sin(phi)*np.sin(psi)*np.sin(theta)))/(m_RB*r_z_gb__b), (np.cos(theta)*np.sin(phi)*(m_Axy + m_RB))/(m_RB*r_z_gb__b), 0],
                         [-(np.cos(psi)*np.cos(theta)*(m_Axy + m_RB))/(m_RB*r_z_gb__b), -(np.cos(theta)*np.sin(psi)*(m_Axy + m_RB))/(m_RB*r_z_gb__b), (np.sin(theta)*(m_Axy + m_RB))/(m_RB*r_z_gb__b), 0],
                         [(np.cos(psi)*np.cos(theta)*np.sin(phi)*(m_Axy + m_RB))/(m_RB*r_z_gb__b*np.cos(phi)), (np.cos(theta)*np.sin(phi)*np.sin(psi)*(m_Axy + m_RB))/(m_RB*r_z_gb__b*np.cos(phi)), -(np.sin(phi)*np.sin(theta)*(m_Axy + m_RB))/(m_RB*r_z_gb__b*np.cos(phi)), np.cos(theta)/np.cos(phi)],
                         [np.sin(phi)*np.sin(psi) + np.cos(phi)*np.cos(psi)*np.sin(theta), np.cos(phi)*np.sin(psi)*np.sin(theta) - np.cos(psi)*np.sin(phi), np.cos(phi)*np.cos(theta), 0]
        ])
        
        zeta1 = np.array([[x],
                          [y],
                          [z],
                          [psi]])
        zeta2 = np.array([[x_dot],
                          [y_dot],
                          [z_dot],
                          [psi_dot]])

        yd = np.array([[self.traj_x[n]],
                       [self.traj_y[n]],
                       [self.traj_z[n]],
                       [self.traj_psi[n]]])
        yd_dot = np.array([[self.traj_x_dot[n]],
                           [self.traj_y_dot[n]],
                           [self.traj_z_dot[n]],
                           [self.traj_psi_dot[n]]])
        yd_ddot = np.array([[self.traj_x_ddot[n]],
                            [self.traj_y_ddot[n]],
                            [self.traj_z_ddot[n]],
                            [self.traj_psi_ddot[n]]])
        
        e1 = zeta1 - yd
        e2 = zeta2 - yd_dot

        k1 = 0.001
        k2 = 0.001

        q = -k1 * e1 - k2 * e2 + yd_ddot

        virtual_u = Binv @ (q - A)

        f_x_ref = v_x_dot*(m_Axy + m_RB) - w_z__b*(m_Axy*v_y__b + m_RB*v_y__b - m_RB*r_z_gb__b*w_x__b) + D_vxy__CB*v_x__b + m_RB*r_z_gb__b*w_y_dot + v_z__b*w_y__b*(m_Az + m_RB)
        f_y_ref = w_z__b*(m_Axy*v_x__b + m_RB*v_x__b + m_RB*r_z_gb__b*w_y__b) + v_y_dot*(m_Axy + m_RB) + D_vxy__CB*v_y__b - m_RB*r_z_gb__b*w_x_dot - v_z__b*w_x__b*(m_Az + m_RB)
        f_z_ref = w_x__b*(m_Axy*v_y__b + m_RB*v_y__b - m_RB*r_z_gb__b*w_x__b) - w_y__b*(m_Axy*v_x__b + m_RB*v_x__b + m_RB*r_z_gb__b*w_y__b) + v_z_dot*(m_Az + m_RB) + D_vz__CB*v_z__b
        tau_z_ref = D_wz__CB*w_z__b + I_Az*w_z_dot + I_RBz*w_z_dot

        f_y   = self.wy_dot_pid.get_ctrl(virtual_u[0].item() - w_x_dot)
        f_x   = self.wx_dot_pid.get_ctrl(virtual_u[1].item() - w_y_dot)
        tau_z = self.wz_dot_pid.get_ctrl(virtual_u[2].item() - w_z_dot)
        f_z   = self.vz_dot_pid.get_ctrl(virtual_u[3].item() - v_z_dot)

        print("v_x_dot: " + str(v_x_dot))
        print("w_dot_y: " + str(w_y_dot))
        print("w_dot_y error: " + str(virtual_u[1].item() - w_y_dot))
        print("fx: " + str(f_x))

        u = np.array([f_x, f_y, f_z, tau_z]).reshape((4,1))
        self.prev_tau = u

        if (np.any(np.isnan(u))):
            print("Detected NaN")
            sys.exit(0)

        sim.end_timer()
        
        return u
    
    def get_trajectory(self):
        return (self.traj_x, self.traj_y, self.traj_z)
    
    def get_error(self, sim):
        n = sim.get_current_timestep() + 1
        return np.array([
                sim.get_var_history('x') - self.traj_x[0:n],
                sim.get_var_history('y') - self.traj_y[0:n],
                sim.get_var_history('z') - self.traj_z[0:n],
                sim.get_var_history('psi') - self.traj_psi[0:n]
        ]).T
    