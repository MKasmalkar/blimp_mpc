from LinearBlimpSim import LinearBlimpSim
from rta.blimp import Blimp
import numpy as np
import scipy

class DiscreteBlimpSim(LinearBlimpSim):

    def __init__(self, dT):
        super().__init__(dT)

        self.blimp = Blimp()
        self.update_A_dis()

        B_int = np.zeros((12,12))
        for i in range(10000):
            dTau = dT / 10000
            tau = i * dTau
            B_int += scipy.linalg.expm(self.A_lin * tau) * dTau
        self.B_dis = B_int @ self.B_lin

    def update_model(self, u):
        self.u = u

        self.update_A_lin()
        self.state_dot = (self.A_lin @ self.state).reshape((12,1)) + (self.B_lin @ self.u).reshape((12,1))
        self.state = self.state + self.state_dot*self.dT

        self.update_history()
    
    def update_A_dis(self):
        self.update_A_lin()
        self.A_dis = scipy.linalg.expm(self.A_lin)