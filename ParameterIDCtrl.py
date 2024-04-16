from BlimpController import BlimpController
import numpy as np

class ParameterIDCtrl(BlimpController):

    def get_ctrl_action(self, sim):
        return np.array([0.0, 0.0, 0.05, 0.0])
    
    def get_error(self, sim):
        n = sim.get_current_timestep() + 1
        return np.array([
            sim.get_var_history('x'),
            sim.get_var_history('y'),
            sim.get_var_history('z'),
            sim.get_var_history('psi')
        ]).T