from typing_extensions import override
from ModelPredictiveController import ModelPredictiveController
import control as ct

class LQR(ModelPredictiveController):

    def __init__(self, base_dir):
        super().__init__(base_dir)
        self.K, self.S, self.E = ct.dlqr(self.A,
                                         self.B,
                                         self.Q,
                                         self.R)
    
    @override
    def get_control_vector(self, x):
        return -self.K @ x