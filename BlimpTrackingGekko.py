from typing_extensions import override
from ModelPredictiveController import ModelPredictiveController
from gekko import GEKKO
import numpy as np

class BlimpTrackingGekko(ModelPredictiveController):

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

    def print_matrix(self, matrix, name):
        print(name + ":")
        print(matrix)
        print()

    @override
    def get_tracking_ctrl(self,
                          current_state,
                          reference,
                          xmin,
                          xmax,
                          umin,
                          umax):

        m = GEKKO()

        # rows = time steps
        # columns = state/input values
        x = m.Array(m.Var, (self.N+1, self.order), lb=xmin, ub=xmax)
        y = m.Array(m.Var, (self.N+1, self.num_outputs))
        z = m.Array(m.Var, (self.N+1, self.num_outputs))
        u = m.Array(m.Var, (self.N, self.num_inputs), lb=umin, ub=umax)     

        A = m.Param(self.A)
        B = m.Param(self.B)
        C = m.Param(self.C)
        P = m.Param(self.P)
        Q = m.Param(self.Q)
        R = m.Param(self.R)

        # Initial condition
        m.Equation(x[0, :] == current_state.T)

        yr = reference.T
        
        # add running cost
        for k in range(self.N):
            m.Equation(y[k, :] == C @ x[k, :].T)
            m.Equation(z[k, :] == y[k, :] - yr)
            m.Equation(x[k+1, :] == (A @ x[k, :].T + B @ u[k, :].T).T)
            
        # terminal cost
        obj1 = z[self.N, :] @ P @ z[self.N, :].T

        # running state/error cost
        obj2 = sum([z[k, :] @ Q @ z[k, :].T for k in range(self.N)])
        
        # running input cost
        obj3 = sum([u[k, :] @ R @ u[k, :].T for k in range(self.N)])

        m.Minimize(obj1 + obj2 + obj3)

        m.options.IMODE = 7

        m.solve()
        
        return np.matrix(u[0]).T
