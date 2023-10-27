from LinearBlimpSim import LinearBlimpSim
from rta.blimp import Blimp
import numpy as np
import scipy

class DiscreteBlimpSim(LinearBlimpSim):

    def __init__(self, dT):
        super().__init__(dT)

    def update_model(self, u):
        self.u = u

        self.update_A_lin()
        self.state_dot = (self.A_lin @ self.state).reshape((12,1)) + (self.B_lin @ self.u).reshape((12,1))
        self.state = self.state + self.state_dot*self.dT

        self.update_history()