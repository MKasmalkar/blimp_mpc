from LinearBlimpSim import LinearBlimpSim
from rta.blimp import Blimp
import numpy as np
import scipy

class DiscreteBlimpSim(LinearBlimpSim):

    def __init__(self, dT):
        super().__init__(dT)

    def update_model(self, u):
        self.u = u

        self.update_A_dis()
        self.state = np.asarray(self.A_dis @ self.state.reshape((12,1)) + self.B_dis @ self.u.reshape((4,1)))
        self.state_dot = (self.state.reshape((12,1)) - self.state_history[self.current_timestep].reshape((12,1))) / self.dT
        
        self.update_history()