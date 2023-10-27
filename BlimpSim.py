import numpy as np
import scipy

from parameters import *
from operators import *

class BlimpSim():

    def __init__(self, dT):
        self.dT = dT

        self.idx = {'vx': 0,
                    'vy': 1,
                    'vz': 2,
                    'wx': 3,
                    'wy': 4,
                    'wz': 5,
                    'x': 6,
                    'y': 7,
                    'z': 8,
                    'phi': 9,
                    'theta': 10,
                    'psi': 11}

        self.current_timestep = 0

        self.time_vec = np.array([0])

        self.state = np.zeros((12,1))
        self.state_dot = np.zeros((12,1))
        self.u = np.zeros((4,1))

        self.state_history = np.zeros((1,12))
        self.state_dot_history = np.zeros((1,12))
        self.u_history = np.zeros((1, 4))
        
    def update_model_discrete(self, u):
        self.u = u

        self.update_A_dis()

        self.state = (self.A_dis @ self.state).reshape((12,1)) + (self.B_dis @ self.u).reshape((12,1))
        self.state_dot = (self.state_history[self.current_timestep] - self.state_history[self.current_timestep-1])/self.dT
        
        self.update_history()

    def update_model(self, u):
        pass

    def update_model_nonlinear(self, u):
        self.u = u

        tau_B = np.array([u[0],
                            u[1],
                            u[2],
                            -r_z_tg__b * u[1],
                            r_z_tg__b * u[0],
                            u[3]]).reshape((6,1))
        
        phi = self.state[self.idx['phi']]
        theta = self.state[self.idx['theta']]
        psi = self.state[self.idx['psi']]
        
        # Restoration torque
        fg_B = R_b__n_inv(phi, theta, psi) @ fg_n
        g_CB = -np.block([[np.zeros((3, 1))],
                        [np.reshape(np.cross(r_gb__b, fg_B), (3, 1))]])

        # Update state
        eta_bn_n_dot = np.block([[R_b__n(phi, theta, psi),    np.zeros((3, 3))],
                                [np.zeros((3, 3)),            T(phi, theta)]]) @ nu_bn_b
        
        nu_bn_b_dot = np.reshape(-M_CB_inv @ (C(M_CB, nu_bn_b) @ nu_bn_b + \
                            D_CB @ nu_bn_b + g_CB - tau_B), (6, 1))
        
        eta_bn_n = eta_bn_n + eta_bn_n_dot * self.dT
        nu_bn_b = nu_bn_b + nu_bn_b_dot * self.dT

        self.state_dot = np.vstack((nu_bn_b_dot, eta_bn_n_dot))
        self.state = np.vstack((nu_bn_b, eta_bn_n))

        self.update_history()

    def set_var(self, var, val):
        self.state[self.idx[var]] = val
        self.state_history[self.current_timestep] = self.state

    def set_var_dot(self, var, val):
        self.state_dot[self.idx[var]] = val
        self.state_dot_history[self.current_timestep] = self.state_dot

    def get_var(self, var):
        return self.state[self.idx[var]]
    
    def get_var_dot(self, var):
        return self.state_dot[self.idx[var]]
    
    def get_var_history(self, var):
        return self.state_history[:, self.idx[var]]
    
    def get_var_dot_history(self, var):
        return self.state_dot_history[:, self.idx[var]]
    
    def get_state(self):
        return self.state
    
    def get_body_x(self, length):
        return R_b__n(self.get_var('phi'),
                      self.get_var('theta'),
                      self.get_var('psi')) \
                        @ np.array([length, 0, 0]).T
    
    def get_body_y(self, length):
        return R_b__n(self.get_var('phi'),
                      self.get_var('theta'),
                      self.get_var('psi')) \
                        @ np.array([0, length, 0]).T
    
    def get_body_z(self, length):
        return R_b__n(self.get_var('phi'),
                      self.get_var('theta'),
                      self.get_var('psi')) \
                        @ np.array([0, 0, length]).T
    
    def get_time_vec(self):
        return self.time_vec

    def update_history(self):
        self.current_timestep += 1

        self.state_history = np.append(self.state_history, self.state.reshape((1,12)), axis=0)
        self.state_dot_history = np.append(self.state_dot_history, self.state_dot.reshape((1,12)), axis=0)
        self.u_history = np.append(self.u_history, self.u.reshape((1,4)), axis=0)
        self.time_vec = np.append(self.time_vec, self.current_timestep * self.dT)