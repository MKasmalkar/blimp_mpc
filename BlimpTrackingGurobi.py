from typing_extensions import override
from ModelPredictiveController import ModelPredictiveController
import gurobipy as gp
from gurobipy import GRB
import numpy as np
import time

class BlimpTrackingGurobi(ModelPredictiveController):

    def __init__(self, base_dir, N):
        super().__init__(base_dir)

        self.N = N

        self.order = self.A.shape[0]

    def __init__(self, A, B, C, D, P, Q, R, N):
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
                          reference,
                          xmin,
                          xmax,
                          umin,
                          umax):

        self.start_timer()
        with gp.Env(empty=True) as env:
            env.setParam('OutputFlag', 0)
            env.setParam('LogToConsole', 0)
            env.start()

            self.check_time("Benchmark 1")

            with gp.Model(env=env) as m:
                self.check_time("Benchmark 2")

                x = m.addMVar(shape=(self.N+1, self.order),
                             lb=xmin.T, ub=xmax.T, name='x')
                y = m.addMVar(shape=(self.N+1, self.num_outputs),
                             lb=-GRB.INFINITY, ub=GRB.INFINITY, name='y')
                z = m.addMVar(shape=(self.N+1, self.num_outputs),
                             lb=-GRB.INFINITY, ub=GRB.INFINITY, name='z')
                u = m.addMVar(shape=(self.N, self.num_inputs),
                             lb=umin.T, ub=umax.T, name='u')
                
                self.check_time("Benchmark 3")

                m.addConstr(x[0, :] == current_state.T)

                self.check_time("Benchmark 4")
                
                yr = reference.T
        
                for k in range(self.N):
                    m.addConstr(y[k, :] == self.C @ x[k, :])
                    m.addConstr(z[k, :] == y[k, :] - yr)
                    m.addConstr(x[k+1, :] == self.A @ x[k, :] + self.B @ u[k, :])

                self.check_time("Benchmark 5")

                # terminal cost
                obj1 = z[self.N, :] @ self.P @ z[self.N, :]
                
                # running state/error cost
                obj2 = sum(z[k, :] @ self.Q @ z[k, :] for k in range(self.N))
                
                # running input cost
                obj3 = sum(u[k, :] @ self.R @ u[k, :] for k in range(self.N))

                m.setObjective(obj1 + obj2 + obj3)

                self.check_time("Benchmark 6")

                sol = m.optimize()
                
                self.check_time("Benchmark 7", True)

                return np.matrix(u.X[0]).T
