from typing_extensions import override
from ModelPredictiveController import ModelPredictiveController
import casadi as cs
import numpy as np

class BlimpTrackingMPC(ModelPredictiveController):

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

        opti = cs.Opti()

        # rows = time steps
        # columns = state/input values
        x = opti.variable(self.N+1, self.order)
        y = opti.variable(self.N+1, self.num_outputs)
        z = opti.variable(self.N+1, self.num_outputs)
        u = opti.variable(self.N, self.num_inputs)

        A = opti.parameter(self.A.shape[0], self.A.shape[1])
        opti.set_value(A, self.A)

        B = opti.parameter(self.B.shape[0], self.B.shape[1])
        opti.set_value(B, self.B)

        C = opti.parameter(self.C.shape[0], self.C.shape[1])
        opti.set_value(C, self.C)

        P = opti.parameter(self.P.shape[0], self.P.shape[1])
        opti.set_value(P, self.P)

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
            opti.subject_to(opti.bounded(xmin.T, x[k, :], xmax.T))
            opti.subject_to(opti.bounded(umin.T, u[k, :], umax.T))
        
        # terminal cost
        obj1 = z[self.N, :] @ P @ z[self.N, :].T
        
        # running state/error cost
        obj2 = sum(z[k, :] @ Q @ z[k, :].T for k in range(self.N))
        
        # running input cost
        obj3 = sum(u[k, :] @ R @ u[k, :].T for k in range(self.N))

        opti.minimize(obj1 + obj2 + obj3)

        p_opts = {'expand': False, 'print_time': False, 'verbose': False}
        s_opts = {'max_iter': 100, 'print_level': 0}
        opti.solver('ipopt', p_opts, s_opts)
        
        sol = opti.solve()

        return np.matrix(sol.value(u)[0]).T
