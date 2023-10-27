import numpy as np
import scipy
from rta.blimp import Blimp

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

        self.blimp = Blimp()

        self.update_A_lin()
        self.update_A_dis()

        self.B_lin = self.blimp.B
        
        B_int = np.zeros((12,12))
        for i in range(10000):
            dTau = dT / 10000
            tau = i * dTau
            B_int += scipy.linalg.expm(self.A_lin * tau) * dTau
        self.B_dis = B_int @ self.B_lin #np.linalg.inv(A) @ (A_dis - np.eye(12)) @ B
    
    def update_model(self, u):
        pass

    def update_A_dis(self):
        self.update_A_lin()
        self.A_dis = scipy.linalg.expm(self.A_lin * self.dT)
        
    def update_A_lin(self):
        self.A_lin = self.blimp.jacobian_np(self.state.reshape((12,1)))

    def get_A_lin(self):
        self.update_A_lin()
        return self.A_lin
    
    def get_A_dis(self):
        self.update_A_dis()
        return self.A_dis
    
    def get_B_lin(self):
        return self.B_lin
    
    def get_B_dis(self):
        return self.B_dis

    def set_var(self, var, val):
        self.state[self.idx[var]] = val
        self.state_history[self.current_timestep] = self.state.reshape(12)

    def set_var_dot(self, var, val):
        self.state_dot[self.idx[var]] = val
        self.state_dot_history[self.current_timestep] = self.state_dot

    def get_var(self, var):
        return self.state[self.idx[var]].item()
    
    def get_var_dot(self, var):
        return self.state_dot[self.idx[var]].item()
    
    def get_var_history(self, var):
        return self.state_history[:, self.idx[var]]
    
    def get_var_dot_history(self, var):
        return self.state_dot_history[:, self.idx[var]]
    
    def get_state(self):
        return self.state
    
    def get_state_dot(self):
        return self.state_dot
    
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
    
    def get_current_timestep(self):
        return self.current_timestep
    
    def get_current_time(self):
        return self.current_timestep * self.dT

    def update_history(self):
        self.current_timestep += 1

        self.state_history = np.append(self.state_history, self.state.reshape((1,12)), axis=0)
        self.state_dot_history = np.append(self.state_dot_history, self.state_dot.reshape((1,12)), axis=0)
        self.u_history = np.append(self.u_history, self.u.reshape((1,4)), axis=0)
        self.time_vec = np.append(self.time_vec, self.current_timestep * self.dT)