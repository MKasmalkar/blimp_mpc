import casadi as cs
import numpy as np
import time
from BlimpController import BlimpController
from parameters import *

class CasadiHelix(BlimpController):

    def __init__(self, dT):
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
    
    def init_sim(self, sim):
        # Get A matrix corresponding to zero state vector equilibrium position
        self.A = sim.get_A_dis()
        
        sim.set_var('x', self.x0)
        sim.set_var('psi', self.psi0)

        self.B = sim.get_B_dis()

        self.C = np.matrix([[0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]])
        self.D = np.zeros((self.num_outputs, self.num_inputs))

        self.P = np.identity(self.num_outputs)
        self.Q = np.identity(self.num_outputs)
        self.R = np.identity(self.num_inputs)
        
        self.N = 250 
        
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

        A = self.opti.parameter(self.A.shape[0], self.A.shape[1])
        self.opti.set_value(A, self.A)

        B = self.opti.parameter(self.B.shape[0], self.B.shape[1])
        self.opti.set_value(B, self.B)
        
        C = self.opti.parameter(self.C.shape[0], self.C.shape[1])
        self.opti.set_value(C, self.C)
        
        P = self.opti.parameter(self.P.shape[0], self.P.shape[1])
        self.opti.set_value(P, self.P)

        Q = self.opti.parameter(self.Q.shape[0], self.Q.shape[1])
        self.opti.set_value(Q, self.Q)

        R = self.opti.parameter(self.R.shape[0], self.R.shape[1])
        self.opti.set_value(R, self.R)
        
        self.x = self.opti.variable(self.N+1, self.order)
        self.y = self.opti.variable(self.N+1, self.num_outputs)
        self.z = self.opti.variable(self.N+1, self.num_outputs)
        self.u = self.opti.variable(self.N, self.num_inputs)
        
        for k in range(self.N):
            self.opti.subject_to(self.y[k, :] == (self.C @ self.x[k, :].T).T)
            self.opti.subject_to(self.x[k+1, :] == (A @ self.x[k, :].T + B @ self.u[k, :].T).T)
            # self.opti.subject_to(self.opti.bounded(xmin.T, self.x[k, :], xmax.T))
            # self.opti.subject_to(self.opti.bounded(umin.T, self.u[k, :], umax.T))
            
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

        sim_state = sim.get_state()

        # A matrix is generated assuming psi = 0
        # need to perform some rotations to account for this

        psi = sim_state[11].item()

        v_b = np.array([
            [sim_state[0].item()],
            [sim_state[1].item()],
            [sim_state[2].item()]
        ])

        v_n = R_b__n(0, 0, psi) @ v_b

        state = np.array([
            v_n[0].item(),
            v_n[1].item(),
            v_n[2].item(),
            sim_state[3].item(),
            sim_state[4].item(),
            sim_state[5].item(),
            sim_state[6].item(),
            sim_state[7].item(),
            sim_state[8].item(),
            sim_state[9].item(),
            sim_state[10].item(),
            psi
        ])

        opti_k = self.opti.copy()
        
        opti_k.subject_to(self.x[0, :] == state.reshape((1,12)))

        for k in range(self.N):
            if k < reference.shape[1]:
                opti_k.subject_to(self.z[k, :] == self.y[k, :] - reference[:, k].reshape((1,6)))
         
        sol = opti_k.solve()

        u_orig = np.array(sol.value(self.u)[0]).T

        u_rot = R_b__n_inv(0, 0, psi) @ np.array([u_orig[0], u_orig[1], u_orig[2]]).T

        u = np.array([
            [u_rot[0].item()],
            [u_rot[1].item()],
            [u_rot[2].item()],
            [u_orig[3].item()]
        ])

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
