import os
import numpy as np

class ModelPredictiveController:

    def __init__(self, base_dir):
        self.A, self.B, self.C, self.D, self.P, self.Q, self.R = ModelPredictiveController.load_dynamics(base_dir)
        
    @staticmethod
    def load_dynamics(base_dir):
        A = np.matrix(ModelPredictiveController.read_matrix_from_file(os.path.join(base_dir, 'A.csv')))
        B = np.matrix(ModelPredictiveController.read_matrix_from_file(os.path.join(base_dir, 'B.csv')))
        C = np.matrix(ModelPredictiveController.read_matrix_from_file(os.path.join(base_dir, 'C.csv')))
        D = np.matrix(ModelPredictiveController.read_matrix_from_file(os.path.join(base_dir, 'D.csv')))
        P = np.matrix(ModelPredictiveController.read_matrix_from_file(os.path.join(base_dir, 'P.csv')))
        Q = np.matrix(ModelPredictiveController.read_matrix_from_file(os.path.join(base_dir, 'Q.csv')))
        R = np.matrix(ModelPredictiveController.read_matrix_from_file(os.path.join(base_dir, 'R.csv')))
        
        # A = dynamics matrix
        # B = input matrix
        # C = output matrix
        # D = feedthrough matrix
        # P = terminal cost matrix
        # Q = running state cost matrix
        # R = running input cost matrix

        return A, B, C, D, P, Q, R

    @staticmethod
    def read_matrix_from_file(file):
        with open(file, 'r') as file:
            return [[float(n) for n in line.split(',')] for line in file]
    
    def get_control_vector(self, x):
        pass

    def get_tracking_ctrl(self, x, xr):
        pass