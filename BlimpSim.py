import numpy as np
import scipy
from rta.blimp import Blimp
import time
import csv

from parameters import *
from operators import *

class BlimpSim():

    def __init__(self, dT):
        self.dT = dT

        self.state_idx = {'vx': 0,
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
        
        self.input_idx = {'fx': 0,
                          'fy': 1,
                          'fz': 2,
                          'tz': 3}

        self.current_timestep = 0

        self.time_vec = np.array([0])

        self.state = np.zeros((12,1))
        self.state_dot = np.zeros((12,1))
        self.u = np.zeros((4,1))

        self.state_history = np.zeros((1,12))
        self.state_dot_history = np.zeros((1,12))
        self.u_history = np.zeros((1, 4))

        # The time it took to compute the control input
        # that led to the current state
        self.solve_time_history = np.array([0])

        self.start_time = 0
        self.time_delta = 0

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

    def update_A_lin(self):
        psi = self.get_var('psi')

        self.A_lin = np.array([
            [-0.024918743228681705659255385398865, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, -0.024918743228681705659255385398865, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, -0.064008534471213351935148239135742, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, -0.016812636348731757607311010360718, 0, 0, 0, 0, 0, -0.15352534538760664872825145721436, 0, 0],
            [0, 0, 0, 0, -0.016812636348731757607311010360718, 0, 0, 0, 0, 0, -0.15352534538760664872825145721436, 0],
            [0, 0, 0, 0, 0, -0.016835595258726243628188967704773, 0, 0, 0, 0, 0, 0],
            [np.cos(psi), -1.0*np.sin(psi), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [np.sin(psi), np.cos(psi), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 1.0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1.0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1.0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1.0, 0, 0, 0, 0, 0, 0]
        ])

    def update_A_dis(self):
        self.update_A_lin()
        self.A_dis = scipy.linalg.expm(self.A_lin * self.dT)
    
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
    
    def start_timer(self):
        self.start_time = time.process_time_ns()

    def end_timer(self):
        self.time_delta = time.process_time_ns() - self.start_time
        
    def set_var(self, var, val):
        self.state[self.state_idx[var]] = val
        self.state_history[self.current_timestep] = self.state.reshape(12)

    def set_var_dot(self, var, val):
        self.state_dot[self.state_idx[var]] = val
        self.state_dot_history[self.current_timestep] = self.state_dot

    def get_var(self, var):
        return self.state[self.state_idx[var]].item()
    
    def get_var_dot(self, var):
        return self.state_dot[self.state_idx[var]].item()
    
    def get_var_history(self, var):
        return self.state_history[:, self.state_idx[var]]
    
    def get_var_dot_history(self, var):
        return self.state_dot_history[:, self.state_idx[var]]
    
    def get_full_u_history(self):
        return self.u_history

    def get_u_history(self, var):
        return self.u_history[:, self.input_idx[var]]
    
    def get_state(self):
        return self.state
    
    def get_state_history(self):
        return self.state_history
    
    def get_state_dot_history(self):
        return self.state_dot_history
    
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
    
    def get_solve_time_history(self):
        return self.solve_time_history
    
    def get_current_timestep(self):
        return self.current_timestep
    
    def get_current_time(self):
        return self.current_timestep * self.dT

    def update_history(self):
        self.current_timestep += 1

        self.solve_time_history = np.append(self.solve_time_history, self.time_delta)
        self.state_history = np.append(self.state_history, self.state.reshape((1,12)), axis=0)
        self.state_dot_history = np.append(self.state_dot_history, self.state_dot.reshape((1,12)), axis=0)
        self.u_history = np.append(self.u_history, self.u.reshape((1,4)), axis=0)
        self.time_vec = np.append(self.time_vec, self.current_timestep * self.dT)

    def load_data(self, filename):
        with open('logs/' + filename, 'r') as infile:
            reader = csv.reader(infile)
            data_list = list(reader)[1:]
            data_float = [[float(i) for i in j] for j in data_list]
            data_np = np.array(data_float)

            self.time_vec = data_np[:, 0]

            self.current_timestep = int(data_np[1, 34].item())

            self.state_history = data_np[:, 1:13]
            self.state = self.state_history[self.current_timestep, :]
   
            self.state_dot_history = data_np[:, 13:25]
            self.state_dot = self.state_dot_history[self.current_timestep, :]

            self.u_history = data_np[:, 25:29]
            self.u = self.u_history[self.current_timestep, :]

            self.solve_time_history = data_np[:, 33]
            self.dT = data_np[0, 34]