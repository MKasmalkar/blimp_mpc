from typing_extensions import override
from ModelPredictiveController import ModelPredictiveController
import gurobipy as gp
from gurobipy import GRB
import numpy as np

class ExampleMPC(ModelPredictiveController):

    def __init__(self, base_dir, N):
        super().__init__(base_dir)

        self.N = N

        self.order = self.A.shape[0]

    def print_matrix(self, matrix, name):
        print(name + ":")
        print(matrix)
        print()

    @override
    def get_control_vector(self, current_state):
        
        with gp.Env(empty=True) as env:
            env.setParam('OutputFlag', 0)
            env.setParam('LogToConsole', 0)
            env.start()

            with gp.Model(env=env) as m:

                xmin = np.tile(-5, (self.N+1, 1))
                xmax = np.tile(5, (self.N+1, 1))
                umin = np.tile(-0.5, (self.N, 1))
                umax = np.tile(0.5, (self.N, 1))

                # rows = time steps
                # columns = state/input values
                x = m.addMVar(shape=(self.N+1, self.order),
                            lb=xmin, ub=xmax, name='x')
                z = m.addMVar(shape=(self.N+1, self.order),
                            lb=-GRB.INFINITY, name='z')
                u = m.addMVar(shape=(self.N, 1),
                            lb=umin, ub=umax, name='u')

                m.addConstr(x[0, :] == current_state.T)
                
                xr = np.array([0, 0])

                # add running cost
                for k in range(self.N):
                    m.addConstr(z[k, :] == x[k, :] - xr)
                    m.addConstr(x[k+1, :] == self.A @ x[k, :] + self.B @ u[k, :])
                
                obj1 = sum(z[k, :] @ self.Q @ z[k, :] for k in range(self.N+1))
                obj2 = sum(u[k, :] @ self.R @ u[k, :] for k in range(self.N))
                m.setObjective(obj1 + obj2, GRB.MINIMIZE)
                m.optimize()

                return u.X[0]
