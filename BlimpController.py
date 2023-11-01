import numpy as np
import csv

class BlimpController():

    def __init__(self, dT):
        self.dT = dT
        self.error_history = None

    def get_ctrl_action(self, sim):
        pass

    def init_sim(self, sim):
        pass

    def get_trajectory(self):
        pass

    def get_error(self, sim):
        return self.error_history

    def load_data(self, filename):
        with open('logs/' + filename, 'r') as infile:
            reader = csv.reader(infile)
            data_list = list(reader)[1:]
            data_float = [[float(i) for i in j] for j in data_list]
            data_np = np.array(data_float)

            self.dT = data_np[0, 34]
            self.error_history = data_np[:, 29:33]