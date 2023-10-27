from typing_extensions import override
from ModelPredictiveController import ModelPredictiveController
import gurobipy as gp
from gurobipy import GRB
import numpy as np
import time

class TrajTrackingGurobi(ModelPredictiveController):

    def __init__(self, base_dir, N):
        super().__init__(base_dir)

        self.N = N

        self.order = self.A.shape[0]

    def __init__(self, A, B, C, D, P, Q, R, N,
                 xmin,
                 xmax,
                 umin,
                 umax):
        
        self.A = A
        self.B = B
        self.C = C
        self.D = D
        self.P = P
        self.Q = Q
        self.R = R
        self.N = N

        self.order = self.A.shape[0]
        self.num_inputs = self.B.shape[1]
        self.num_outputs = self.C.shape[0]

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

        self.times = {}

    def print_matrix(self, matrix, name):
        print(name + ":")
        print(matrix)
        print()

    def start_timer(self):
        self.start_time = time.time()

    def check_time(self, message, p=False):
        deltaT = time.time() - self.start_time
        
        if p:
            print(message + ": " + str(deltaT))

        if message not in self.times.keys():
            self.times[message] = [deltaT]
        else:
            self.times[message].append(deltaT)
            
        self.start_time = time.time()

    @override
    def get_tracking_ctrl(self,
                          current_state,
                          reference):

        self.start_timer()

        self.ic_constraint.rhs = current_state.flatten()

        for k in range(self.N):
            self.error_constraints[k].rhs = reference[k]
        
        self.m.optimize()

        self.check_time('deltaT')

        # print(self.m.status)
        # print(self.u.X)
        return np.matrix(self.u.X[0]).T
