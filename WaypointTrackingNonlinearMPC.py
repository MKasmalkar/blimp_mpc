import gurobipy as gp
from gurobipy import GRB
import numpy as np
import time
from BlimpController import BlimpController

class WaypointTrackingMPC(BlimpController):

    def __init__(self, dT):
        super().__init__(dT)

                
        self.ref_idx = 0
        self.reference_points = [
                            np.array([5, 5, 5]),
                            np.array([5, -5, -5]),
                            np.array([-5, -5, 2.5]),
                            np.array([-5, 5, -2.5])
                            ]
        self.NUM_REF_PTS = 4

        self.DEADBAND = 1

        self.TIMESTEPS_TO_SETTLE = 10
        self.settling_timer = self.TIMESTEPS_TO_SETTLE
    
    def distance_to_goal(self, state, goal):
        return np.linalg.norm(state - goal)

    def init_sim(self, sim):
        self.A = sim.get_A_dis()
        self.B = sim.get_B_dis()

        self.C = np.matrix([[0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0]])
        self.D = np.zeros((3, 4))

        self.P = np.identity(3) * 10
        self.Q = np.identity(3) * 10
        self.R = np.identity(4) * 10
 
        xmin = np.matrix([[-np.inf],
                        [-np.inf],
                        [-np.inf],
                        [-np.inf],
                        [-np.inf],
                        [-np.inf],
                        [-10],   # x
                        [-10],   # y
                        [-10],   # z
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
                        [10],   # x
                        [10],   # y
                        [10],   # z
                        [np.inf],
                        [np.inf],
                        [np.inf]
                        ])

        umin = np.matrix([[-0.005],
                        [-0.005],
                        [-0.005],
                        [-np.inf]])

        umax = np.matrix([[0.005],
                        [0.005],
                        [0.005],
                        [np.inf]])
        
        self.N = 10

        self.order = 12
        self.num_inputs = 4
        self.num_outputs = 3

        
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
            self.m.addConstr(self.x[k+1, :] == self.A @ self.x[k, :] + self.B @ self.u[k, :])

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

        error = self.distance_to_goal(self.C @ sim.get_state(),
                                      self.reference_points[self.ref_idx])
        if error < self.DEADBAND:
            self.settling_timer -= 1
            if self.settling_timer == 0:
                self.settling_timer = self.TIMESTEPS_TO_SETTLE
                self.ref_idx = (self.ref_idx + 1) % self.NUM_REF_PTS
        else:
            self.settling_timer = self.TIMESTEPS_TO_SETTLE
        
        reference = self.reference_points[self.ref_idx]

        sim_state = sim.get_state()
        state = np.array([
            sim_state[0].item(),
            sim_state[1].item(),
            sim_state[2].item(),
            sim_state[3].item(),
            sim_state[4].item(),
            sim_state[5].item(),
            sim_state[6].item(),
            sim_state[7].item(),
            sim_state[8].item(),
            sim_state[9].item(),
            sim_state[10].item(),
            sim_state[11].item(),
        ])
        self.ic_constraint.rhs = state

        for k in range(self.N):
            self.error_constraints[k].rhs = reference
        
        self.m.optimize()

        # print(self.m.status)
        # print(self.u.X)
        return np.matrix(self.u.X[0]).T

    def get_trajectory(self):
        return (self.reference_points[self.ref_idx][0].item(),
                self.reference_points[self.ref_idx][1].item(),
                self.reference_points[self.ref_idx][2].item())