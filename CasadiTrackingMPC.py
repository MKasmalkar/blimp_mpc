from typing_extensions import override
from ModelPredictiveController import ModelPredictiveController
import casadi as cs
import numpy as np

class CasadiTrackingMPC(ModelPredictiveController):

    def __init__(self, base_dir, N):
        super().__init__(base_dir)

        self.N = N

        self.order = self.A.shape[0]

    def print_matrix(self, matrix, name):
        print(name + ":")
        print(matrix)
        print()

    @override
    def get_tracking_ctrl(self, current_state, reference):

        opti = cs.Opti()

        xmin = np.tile(-11, (1, self.N+1))
        xmax = np.tile(11, (1, self.N+1))
        umin = np.tile(-5, (1, self.N))
        umax = np.tile(5, (1, self.N))

        # rows = time steps
        # columns = state/input values
        x = opti.variable(self.N+1, self.order)
        y = opti.variable(self.N+1, self.C.shape[1])
        z = opti.variable(self.N+1, self.order)
        u = opti.variable(self.N, 1)

        A = opti.parameter(self.A.shape[0], self.A.shape[1])
        opti.set_value(A, self.A)

        B = opti.parameter(self.B.shape[0], self.B.shape[1])
        opti.set_value(B, self.B)

        C = opti.parameter(self.C.shape[0], self.C.shape[1])
        opti.set_value(C, self.C)

        Q = opti.parameter(self.Q.shape[0], self.Q.shape[1])
        opti.set_value(Q, self.Q)

        R = opti.parameter(self.R.shape[0], self.R.shape[1])
        opti.set_value(R, self.R)

        opti.subject_to(x[0, :] == current_state.T)

        yr = reference.T
        
        # add running cost
        for k in range(self.N):
            opti.subject_to(y[k, :] == (C @ x[k, :].T).T)
            opti.subject_to(z[k, :] == y[k, :] - yr)
            opti.subject_to(x[k+1, :] == (A @ x[k, :].T + B @ u[k, :].T).T)
            opti.subject_to(opti.bounded(xmin, x[k, :], xmax))
            opti.subject_to(opti.bounded(umin, u[k, :], umax))
        
        obj1 = sum(z[k, :] @ Q @ z[k, :].T for k in range(self.N+1))
        obj2 = sum(u[k, :] @ R @ u[k, :].T for k in range(self.N))
        opti.minimize(obj1 + obj2)

        p_opts = {'expand': False, 'print_time': False, 'verbose': False}
        s_opts = {'max_iter': 100, 'print_level': 0}
        opti.solver('ipopt', p_opts, s_opts)
        
        sol = opti.solve()
        # print(sol.value(u))

        return sol.value(u)[0]
