from BlimpController import BlimpController
import numpy as np

class TestController(BlimpController):

    def get_ctrl_action(self, sim):
        return np.array([
            0.05, 0, 0, 0
        ])