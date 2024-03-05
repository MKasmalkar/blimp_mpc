import gurobipy as gp
from gurobipy import GRB
import numpy as np
import time
from BlimpController import BlimpController
from parameters import *

class WaypointTrackingMPC(BlimpController):

    def __init__(self, dT):
        super().__init__(dT)

        self.order = 12
        self.num_inputs = 4
        self.num_outputs = 6
        
        self.ref_idx = 0
        self.reference_points = [
                            np.array([5, 5, 5, 0]),
                            np.array([5, -5, -5, -np.pi/2]),
                            np.array([-5, -5, 2.5, np.pi]),
                            np.array([-5, 5, -2.5, -np.pi])
                            ]
        self.NUM_REF_PTS = 4

        self.DEADBAND = 1

        self.TIMESTEPS_TO_SETTLE = 5 / dT
        self.settling_timer = self.TIMESTEPS_TO_SETTLE

    def init_sim(self, sim):
        self.error_history = np.array([[sim.get_var('x') - self.reference_points[0][0],
                                      sim.get_var('y') - self.reference_points[0][1],
                                      sim.get_var('z') - self.reference_points[0][2],
                                      sim.get_var('psi') - self.reference_points[0][3]]]).reshape((1, 4))
        
        # Get A matrix corresponding to zero state vector equilibrium position
        A_dis = sim.get_A_dis()
        self.B = sim.get_B_dis()

        self.C = np.matrix([[0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]])
        self.D = np.zeros((self.num_outputs, self.num_inputs))

        self.P = np.identity(self.num_outputs)
        self.Q = np.identity(self.num_outputs) * 1000
        self.R = np.identity(self.num_inputs) * 100000

        max_allowable_phi = 0.05
        max_allowable_theta = 0.05
        max_allowable_psi_deviation = 0.05

        max_allowable_x_deviation = 0.4
        max_allowable_y_deviation = 0.4
        max_allowable_z_deviation = 0.4

        self.Q[0, 0] = 1/max_allowable_x_deviation**2
        self.Q[1, 1] = 1/max_allowable_y_deviation**2
        self.Q[2, 2] = 1/max_allowable_z_deviation**2

        self.P[0, 0] = 1/max_allowable_x_deviation**2
        self.P[1, 1] = 1/max_allowable_y_deviation**2
        self.P[2, 2] = 1/max_allowable_z_deviation**2

        self.Q[3, 3] = 1/max_allowable_phi**2
        self.Q[4, 4] = 1/max_allowable_theta**2
        self.Q[5, 5] = 1/max_allowable_psi_deviation**2

        self.P[3, 3] = 1/max_allowable_phi**2
        self.P[4, 4] = 1/max_allowable_theta**2
        self.P[5, 5] = 1/max_allowable_psi_deviation**2

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

        reference = np.array([
            np.ones(self.N) * self.reference_points[self.ref_idx][0],
            np.ones(self.N) * self.reference_points[self.ref_idx][1],
            np.ones(self.N) * self.reference_points[self.ref_idx][2],
            np.zeros(self.N),
            np.zeros(self.N),
            np.ones(self.N) * self.reference_points[self.ref_idx][3],
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

        x = sim.get_var('x')
        y = sim.get_var('y')
        z = sim.get_var('z')

        self.error_history = np.vstack((self.error_history, 
            np.array([
                x - self.reference_points[self.ref_idx][0],
                y - self.reference_points[self.ref_idx][1],
                z - self.reference_points[self.ref_idx][2],
                psi - self.reference_points[self.ref_idx][3],
            ]).reshape((1,4))
        ))

        error = self.distance_to_goal([x, y, z, psi],
                                      self.reference_points[self.ref_idx])
        if error < self.DEADBAND:
            self.settling_timer -= 1
            if self.settling_timer == 0:
                self.settling_timer = self.TIMESTEPS_TO_SETTLE
                self.ref_idx = (self.ref_idx + 1) % self.NUM_REF_PTS
        else:
            self.settling_timer = self.TIMESTEPS_TO_SETTLE

        return u
    
    def get_trajectory(self):
        return (np.array([pt[0] for pt in self.reference_points]),
                np.array([pt[1] for pt in self.reference_points]),
                np.array([pt[2] for pt in self.reference_points]))

    def distance_to_goal(self, state, goal):
        return np.linalg.norm(state - goal)
    