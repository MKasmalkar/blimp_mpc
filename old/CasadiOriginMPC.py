from typing_extensions import override
from ModelPredictiveController import ModelPredictiveController
import casadi as cs
import numpy as np

class CasadiOriginMPC(ModelPredictiveController):

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

        opti = cs.Opti()

        xmin = np.tile(-5, (1, self.N+1))
        xmax = np.tile(5, (1, self.N+1))
        umin = np.tile(-0.5, (1, self.N))
        umax = np.tile(0.5, (1, self.N))

        # rows = time steps
        # columns = state/input values
        x = opti.variable(self.N+1, self.order)
        z = opti.variable(self.N+1, self.order)
        u = opti.variable(self.N, 1)

        A = opti.parameter(self.A.shape[0], self.A.shape[1])
        opti.set_value(A, self.A)

        B = opti.parameter(self.B.shape[0], self.B.shape[1])
        opti.set_value(B, self.B)

        Q = opti.parameter(self.Q.shape[0], self.Q.shape[1])
        opti.set_value(Q, self.Q)

        R = opti.parameter(self.R.shape[0], self.R.shape[1])
        opti.set_value(R, self.R)

        opti.subject_to(x[0, :] == current_state.T)
        
        xr = np.matrix([0, 0])

        # add running cost
        for k in range(self.N):
            opti.subject_to(z[k, :] == x[k, :] - xr)
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
