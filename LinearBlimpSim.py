from BlimpSim import BlimpSim
from rta.blimp import Blimp
import numpy as np
import scipy

class LinearBlimpSim(BlimpSim):

    def __init__(self, dT):
        super().__init__(dT)

        self.blimp = Blimp()
        self.update_A_lin()
        self.B_lin = self.blimp.B

    def update_model(self, u):
        self.u = u

        self.update_A_lin()
        self.state_dot = (self.A_lin @ self.state).reshape((12,1)) + (self.B_lin @ self.u).reshape((12,1))
        self.state = self.state + self.state_dot*self.dT

        self.update_history()
        
    def update_A_lin(self):
        self.A_lin = self.blimp.jacobian_np(self.state)