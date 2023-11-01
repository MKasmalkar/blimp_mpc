import csv
import numpy as np

class BlimpLogger:

    def __init__(self, filename):
        self.filename = filename

    def log(self, sim, ctrl):
        with open('logs/' + self.filename, 'w', newline='') as outfile:
            writer = csv.writer(outfile)
       
            writer.writerow(['time',
                             'x', 'y', 'z', 'phi', 'theta', 'psi', 'vx', 'vy', 'vz', 'wx', 'wy', 'wz',
                             'xdot', 'ydot', 'zdot', 'phidot', 'thetadot', 'psidot', 'vxdot', 'vydot', 'vzdot', 'wxdot', 'wydot', 'wzdot',
                             'fx', 'fy', 'fz', 'tauz',
                             'x_error', 'y_error', 'z_error', 'psi_error',
                             'solve_time',
                             'metadata'])
            
            # want timesteps x data elements

            n = sim.get_current_timestep() + 1

            time_history = sim.get_time_vec().reshape((n, 1))
            state_history = sim.get_state_history()
            state_dot_history = sim.get_state_dot_history()
            u_history = sim.get_full_u_history()
            error_history = ctrl.get_error(sim)
            solve_times = sim.get_solve_time_history().reshape((n, 1))
            
            metadata = np.pad(np.array([sim.dT, sim.get_current_timestep()]), ((0, n-2))).reshape((n,1))

            data = np.hstack((time_history,
                              state_history,
                              state_dot_history,
                              u_history,
                              error_history,
                              solve_times,
                              metadata))
            
            for row in range(data.shape[0]):
                writer.writerow(np.asarray(data[row, :]).flatten())

            print("Wrote to " + str(self.filename))