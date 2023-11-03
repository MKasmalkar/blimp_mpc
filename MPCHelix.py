import gurobipy as gp
from gurobipy import GRB
import numpy as np
import time
from BlimpController import BlimpController
from parameters import *

class MPCHelix(BlimpController):

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
        A_dis = sim.get_A_dis()
        
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

        xmin = np.matrix([[-np.inf],
                        [-np.inf],
                        [-np.inf],
                        [-np.inf],
                        [-np.inf],
                        [-np.inf],
                        [-np.inf],   # x
                        [-np.inf],   # y
                        [-np.inf],   # z
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
                        [np.inf],   # x
                        [np.inf],   # y
                        [np.inf],   # z
                        [np.inf],
                        [np.inf],
                        [np.inf]
                        ])

        umin = np.matrix([[-np.inf],
                        [-np.inf],
                        [-np.inf],
                        [-np.inf]])

        umax = np.matrix([[np.inf],
                        [np.inf],
                        [np.inf],
                        [np.inf]])
        
        self.N = 250

        self.env = gp.Env(empty=True)
        self.env.setParam('OutputFlag', 0)
        self.env.setParam('LogToConsole', 0)
        self.env.start()

        self.m = gp.Model(env=self.env)

        self.x = self.m.addMVar(shape=(self.N+1, self.order),
                        lb=xmin.T, ub=xmax.T, name='x')
        self.y = self.m.addMVar(shape=(self.N+1, self.num_outputs),
                        lb=-GRB.INFINITY, ub=GRB.INFINITY, name='y')
        self.z = self.m.addMVar(shape=(self.N+1, self.num_outputs),
                        lb=-GRB.INFINITY, ub=GRB.INFINITY, name='z')
        self.u = self.m.addMVar(shape=(self.N, self.num_inputs),
                        lb=umin.T, ub=umax.T, name='u')
        
        self.ic_constraint = self.m.addConstr(self.x[0, :] == np.zeros((1, 12)).flatten(), name='ic')

        for k in range(self.N):
            self.m.addConstr(self.y[k, :] == self.C @ self.x[k, :])
            self.m.addConstr(self.x[k+1, :] - self.B @ self.u[k, :] == A_dis @ self.x[k, :],
                                name='dynamics' + str(k))
            
        self.error_constraints = []
        for k in range(self.N):
            self.error_constraints.append(
                # z = error
                # z = y - reference
                # => y - z = reference
                self.m.addConstr(self.y[k, :] - self.z[k, :] == np.zeros((1, self.num_outputs)).flatten(),
                                 name='error' + str(k)))

        # terminal cost
        obj1 = self.z[self.N, :] @ self.P @ self.z[self.N, :]
        
        # running state/error cost
        obj2 = sum(self.z[k, :] @ self.Q @ self.z[k, :] for k in range(self.N))
        
        # running input cost
        obj3 = sum(self.u[k, :] @ self.R @ self.u[k, :] for k in range(self.N))

        self.m.setObjective(obj1 + obj2 + obj3)

        self.m.update()
        
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

        self.ic_constraint.rhs = state

        for k in range(self.N):
            if k < reference.shape[1]:
                self.error_constraints[k].rhs = reference[:, k]
            else:
                self.m.remove(self.error_constraints[k])
            
        self.m.optimize()

        u_orig = self.u.X[0].T

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
