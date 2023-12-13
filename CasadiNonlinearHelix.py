import casadi as cs
import numpy as np
import time
from BlimpController import BlimpController
from parameters import *

class CasadiNonlinearHelix(BlimpController):

    def __init__(self, dT, skip_derivatives=True):
        super().__init__(dT)

        self.order = 12
        self.num_inputs = 4
        self.num_outputs = 6
        
        # Time
        TRACKING_TIME = 20
        SETTLE_TIME = 100

        tracking_time = np.arange(0, TRACKING_TIME, dT)
        settle_time = np.arange(TRACKING_TIME, TRACKING_TIME + SETTLE_TIME + 1, dT)

        time_vec = np.concatenate((tracking_time, settle_time))

        # Trajectory definition
        f = 0.05
        self.At = 1

        self.x0 = self.At
        y0 = 0
        z0 = 0

        phi0 = 0
        theta0 = 0
        self.psi0 = np.pi/2

        v_x0 = 0.0
        v_y0 = 0.0
        v_z0 = 0.0

        w_x0 = 0
        w_y0 = 0
        w_z0 = 0

        z_slope = -1/10

        self.traj_x = np.concatenate((self.At * np.cos(2*np.pi*f*tracking_time), self.At*np.ones(len(settle_time))))
        self.traj_y = np.concatenate((self.At * np.sin(2*np.pi*f*tracking_time), np.zeros(len(settle_time))))
        self.traj_z = np.concatenate((tracking_time * z_slope, TRACKING_TIME * z_slope * np.ones(len(settle_time))))
        self.traj_psi = np.concatenate((self.psi0 + 2*np.pi*f*tracking_time, (self.psi0 + 2*np.pi) * np.ones(len(settle_time))))
        
        self.target_phi = np.zeros(self.traj_x.shape)
        self.target_theta = np.zeros(self.traj_x.shape)

        self.P = np.eye(6)
        self.Q = np.eye(6)
        self.R = np.eye(4)
    
    def init_sim(self, sim):
        sim.set_var('x', self.x0)
        sim.set_var('psi', self.psi0)

        self.C = np.matrix([[0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]])
        
        self.N = 20 
        
        xmin = np.matrix([[-np.inf],
                        [-np.inf],
                        [-np.inf],
                        [-np.inf],
                        [-np.inf],
                        [-np.inf],
                        [-1.5],   # x
                        [-1.5],   # y
                        [-1.5],   # z
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
                        [1.5],   # x
                        [1.5],   # y
                        [5],   # z
                        [np.inf],
                        [np.inf],
                        [np.inf]
                        ])

        umin = np.matrix([[-2],
                        [-2],
                        [-2],
                        [-2]])

        umax = np.matrix([[2],
                        [2],
                        [2],
                        [2]])

        self.opti = cs.Opti()

        dT = self.opti.parameter()
        self.opti.set_value(dT, sim.dT)
        
        self.x = self.opti.variable(self.N+1, self.order)
        self.y = self.opti.variable(self.N+1, self.num_outputs)
        self.z = self.opti.variable(self.N+1, self.num_outputs)
        self.u = self.opti.variable(self.N, self.num_inputs)

        for k in range(self.N):
            self.opti.subject_to(self.y[k, :] == (self.C @ self.x[k, :].T).T)

            v_x_k = self.x[k, 0]
            v_y_k = self.x[k, 1]
            v_z_k = self.x[k, 2]
            w_x_k = self.x[k, 3]
            w_y_k = self.x[k, 4]
            w_z_k = self.x[k, 5]

            x_k = self.x[k, 6]
            y_k = self.x[k, 7]
            z_k = self.x[k, 8]
            phi_k = self.x[k, 9]
            the_k = self.x[k, 10]
            psi_k = self.x[k, 11]

            f_x__b = self.u[k, 0]
            f_y__b = self.u[k, 1]
            f_z__b = self.u[k, 2]
            tau_z__b = self.u[k, 3]

            x_dot = self.opti.variable()
            y_dot = self.opti.variable()
            z_dot = self.opti.variable()
            phi_dot = self.opti.variable()
            theta_dot = self.opti.variable()
            psi_dot = self.opti.variable()
            v_dot_x_bn__b = self.opti.variable()
            v_dot_y_bn__b = self.opti.variable()
            v_dot_z_bn__b = self.opti.variable()
            w_dot_x_bn__b = self.opti.variable()
            w_dot_y_bn__b = self.opti.variable()
            w_dot_z_bn__b = self.opti.variable()

            self.opti.subject_to(x_dot == v_z_k*(cs.sin(phi_k)*cs.sin(psi_k) + cs.cos(phi_k)*cs.cos(psi_k)*cs.sin(the_k)) - v_y_k*(cs.cos(phi_k)*cs.sin(psi_k) - cs.cos(psi_k)*cs.sin(phi_k)*cs.sin(the_k)) + v_x_k*cs.cos(psi_k)*cs.cos(the_k))
            self.opti.subject_to(y_dot == v_y_k*(cs.cos(phi_k)*cs.cos(psi_k) + cs.sin(phi_k)*cs.sin(psi_k)*cs.sin(the_k)) - v_z_k*(cs.cos(psi_k)*cs.sin(phi_k) - cs.cos(phi_k)*cs.sin(psi_k)*cs.sin(the_k)) + v_x_k*cs.cos(the_k)*cs.sin(psi_k))
            self.opti.subject_to(z_dot == v_z_k*cs.cos(phi_k)*cs.cos(the_k) - v_x_k*cs.sin(the_k) + v_y_k*cs.cos(the_k)*cs.sin(phi_k))
            self.opti.subject_to(phi_dot == w_x_k + w_z_k*cs.cos(phi_k)*cs.tan(the_k) + w_y_k*cs.sin(phi_k)*cs.tan(the_k))
            self.opti.subject_to(theta_dot == w_y_k*cs.cos(phi_k) - w_z_k*cs.sin(phi_k))
            self.opti.subject_to(psi_dot == (w_z_k*cs.cos(phi_k)) / cs.cos(the_k) + (w_y_k*cs.sin(phi_k)) / cs.cos(the_k))
            self.opti.subject_to(v_dot_x_bn__b == -(D_vxy__CB*v_x_k - f_x__b - w_z_k*(m_Axy*v_y_k + m_RB*v_y_k - m_RB*r_z_gb__b*w_x_k) + w_y_k*(m_Az*v_z_k + m_RB*v_z_k) + m_RB*r_z_gb__b*w_dot_y_bn__b)/(m_Axy + m_RB))
            self.opti.subject_to(v_dot_y_bn__b == (f_y__b - D_vxy__CB*v_y_k - w_z_k*(m_Axy*v_x_k + m_RB*v_x_k + m_RB*r_z_gb__b*w_y_k) + w_x_k*(m_Az*v_z_k + m_RB*v_z_k) + m_RB*r_z_gb__b*w_dot_x_bn__b)/(m_Axy + m_RB))
            self.opti.subject_to(v_dot_z_bn__b == (f_z__b - D_vz__CB*v_z_k + w_y_k*(m_Axy*v_x_k + m_RB*v_x_k + m_RB*r_z_gb__b*w_y_k) - w_x_k*(m_Axy*v_y_k + m_RB*v_y_k - m_RB*r_z_gb__b*w_x_k))/(m_Az + m_RB))
            self.opti.subject_to(w_dot_x_bn__b == -(D_omega_xy__CB*w_x_k - w_z_k*(w_y_k*(m_RB*r_z_gb__b**2 + I_RBxy) + I_Axy*w_y_k + m_RB*r_z_gb__b*v_x_k) + f_y__b*r_z_tg__b - v_z_k*(m_Axy*v_y_k + m_RB*v_y_k - m_RB*r_z_gb__b*w_x_k) + w_y_k*(I_Az*w_z_k + I_RBz*w_z_k) + v_y_k*(m_Az*v_z_k + m_RB*v_z_k) + (f_g*r_z_gb__b*cs.sin(phi_k - the_k))/2 - m_RB*r_z_gb__b*v_dot_y_bn__b + (f_g*r_z_gb__b*cs.sin(phi_k + the_k))/2)/(m_RB*r_z_gb__b**2 + I_Axy + I_RBxy))
            self.opti.subject_to(w_dot_y_bn__b == -(D_omega_xy__CB*w_y_k + w_z_k*(w_x_k*(m_RB*r_z_gb__b**2 + I_RBxy) + I_Axy*w_x_k - m_RB*r_z_gb__b*v_y_k) - f_x__b*r_z_tg__b + v_z_k*(m_Axy*v_x_k + m_RB*v_x_k + m_RB*r_z_gb__b*w_y_k) - w_x_k*(I_Az*w_z_k + I_RBz*w_z_k) - v_x_k*(m_Az*v_z_k + m_RB*v_z_k) + f_g*r_z_gb__b*cs.sin(the_k) + m_RB*r_z_gb__b*v_dot_x_bn__b)/(m_RB*r_z_gb__b**2 + I_Axy + I_RBxy))
            self.opti.subject_to(w_dot_z_bn__b == (tau_z__b - D_omega_z__CB*w_z_k - w_x_k*(w_y_k*(m_RB*r_z_gb__b**2 + I_RBxy) + I_Axy*w_y_k + m_RB*r_z_gb__b*v_x_k) + w_y_k*(w_x_k*(m_RB*r_z_gb__b**2 + I_RBxy) + I_Axy*w_x_k - m_RB*r_z_gb__b*v_y_k) + v_y_k*(m_Axy*v_x_k + m_RB*v_x_k + m_RB*r_z_gb__b*w_y_k) - v_x_k*(m_Axy*v_y_k + m_RB*v_y_k - m_RB*r_z_gb__b*w_x_k))/(I_Az + I_RBz))

            self.opti.subject_to(self.x[k+1, 0] == self.x[k, 0] + dT * v_dot_x_bn__b)
            self.opti.subject_to(self.x[k+1, 1] == self.x[k, 1] + dT * v_dot_y_bn__b)
            self.opti.subject_to(self.x[k+1, 2] == self.x[k, 2] + dT * v_dot_z_bn__b)
            self.opti.subject_to(self.x[k+1, 3] == self.x[k, 3] + dT * w_dot_x_bn__b)
            self.opti.subject_to(self.x[k+1, 4] == self.x[k, 4] + dT * w_dot_y_bn__b)
            self.opti.subject_to(self.x[k+1, 5] == self.x[k, 5] + dT * w_dot_z_bn__b)
            self.opti.subject_to(self.x[k+1, 6] == self.x[k, 6] + dT * x_dot)
            self.opti.subject_to(self.x[k+1, 7] == self.x[k, 7] + dT * y_dot)
            self.opti.subject_to(self.x[k+1, 8] == self.x[k, 8] + dT * z_dot)
            self.opti.subject_to(self.x[k+1, 9] == self.x[k, 9] + dT * phi_dot)
            self.opti.subject_to(self.x[k+1, 10] == self.x[k, 10] + dT * theta_dot)
            self.opti.subject_to(self.x[k+1, 11] == self.x[k, 11] + dT * psi_dot)
            
        # terminal cost
        self.obj1 = self.z[self.N, :] @ self.P @ self.z[self.N, :].T
        
        # running state/error cost
        self.obj2 = sum(self.z[k, :] @ self.Q @ self.z[k, :].T for k in range(self.N))
        
        # running input cost
        self.obj3 = sum(self.u[k, :] @ self.R @ self.u[k, :].T for k in range(self.N))
        
        self.obj = self.obj1 + self.obj2 + self.obj3

        self.opti.minimize(self.obj)

        p_opts = {'expand': False, 'print_time': False, 'verbose': False}
        s_opts = {'max_iter': 10, 'print_level': 0}
        self.opti.solver('ipopt', p_opts, s_opts)
       
    def get_ctrl_action(self, sim):

        sim.start_timer()

        n = sim.get_current_timestep()

        reference = np.array([
            self.traj_x[n:min(n+self.N, len(self.traj_x))],
            self.traj_y[n:min(n+self.N, len(self.traj_y))],
            self.traj_z[n:min(n+self.N, len(self.traj_z))],
            np.zeros(min(self.N, len(self.traj_x) - n)),
            np.zeros(min(self.N, len(self.traj_x) - n)),
            self.traj_psi[n:min(n+self.N, len(self.traj_psi))]
        ])

        state = sim.get_state()

        opti_k = self.opti.copy()
        
        opti_k.subject_to(self.x[0, :] == state.reshape((1,12)))

        for k in range(self.N):
            if k < reference.shape[1]:
                opti_k.subject_to(self.z[k, :] == self.y[k, :] - reference[:, k].reshape((1,6)))
         
        sol = opti_k.solve()

        u = np.array(sol.value(self.u)[0]).T

        #print(self.m.status)
        #print(np.round(self.u.X[0].T, 3))
        
        sim.end_timer()

        return u
        
    def get_trajectory(self):
        return (self.traj_x,
                self.traj_y,
                self.traj_z)
    
    def get_error(self, sim):
        n = sim.get_current_timestep() + 1
        return np.array([
            sim.get_var_history('x') - self.traj_x[0:n],
            sim.get_var_history('y') - self.traj_y[0:n],
            sim.get_var_history('z') - self.traj_z[0:n],
            sim.get_var_history('psi') - self.traj_psi[0:n]
        ]).T
