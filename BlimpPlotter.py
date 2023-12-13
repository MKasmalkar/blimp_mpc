from tracemalloc import start
import matplotlib.pyplot as plt
import numpy as np
import time

class BlimpPlotter():
    
    def __init__(self):
        self.plotting = False
        self.window_closed = False

    def init_plot(self, title, waveforms=True, disable_plotting=False):
        if disable_plotting:
            self.plotting = False
            return

        self.fig = plt.figure(title, figsize=(13.5, 7))
        plt.ion()

        self.waveforms = waveforms

        if waveforms:
            gs = self.fig.add_gridspec(3,4)

            self.ax_3d = self.fig.add_subplot(gs[0:2, 3], projection='3d')
            self.ax_3d.grid()
            self.ax_or = self.fig.add_subplot(gs[2, 3], projection='3d')
            self.ax_or.grid()

            self.ax_pos = self.fig.add_subplot(gs[0, 0])
            self.ax_vel = self.fig.add_subplot(gs[1, 0])
            self.ax_ang = self.fig.add_subplot(gs[0, 1])
            self.ax_w = self.fig.add_subplot(gs[1, 1])
            self.ax_err_pos = self.fig.add_subplot(gs[2, 0])
            self.ax_err_ang = self.fig.add_subplot(gs[2, 1])
            self.ax_force = self.fig.add_subplot(gs[0, 2])
            self.ax_solve = self.fig.add_subplot(gs[1, 2])

            plt.subplots_adjust(wspace=0.5)
            plt.subplots_adjust(hspace=0.75)
        else:
            self.ax_3d = self.fig.add_subplot(projection='3d')
            self.ax_3d.grid()

        self.fig.canvas.mpl_connect('close_event', self.on_close)

        self.plotting = True
        self.window_closed = False

    def update_plot(self, sim, ctrl):
        
        if not self.plotting: return
    
        self.ax_3d.cla()
        
        self.ax_3d.scatter(sim.get_var_history('x'),
                           sim.get_var_history('y'),
                           sim.get_var_history('z'),
                           color='blue',
                           s=100)
        self.ax_3d.scatter(sim.get_var('x'),
                           sim.get_var('y'),
                           sim.get_var('z'),
                           color='m',
                           s=200)
        # self.ax_3d.set_zlim([-2, 2])
    
        traj = ctrl.get_trajectory()
        
        if traj != None:
            traj_x = traj[0]
            traj_y = traj[1]
            traj_z = traj[2]
            self.ax_3d.scatter(traj_x,
                               traj_y,
                               traj_z,
                               color='g')
    
        self.ax_3d.invert_yaxis()
        self.ax_3d.invert_zaxis()
        self.ax_3d.set_xlabel('x')
        self.ax_3d.set_ylabel('y')
        self.ax_3d.set_zlabel('z')
        self.ax_3d.set_title('Trajectory')
    
        if self.waveforms:
            self.ax_or.cla()

            blimp_x_vector = sim.get_body_x(2)
            blimp_y_vector = sim.get_body_y(2)
            blimp_z_vector = sim.get_body_z(2)
            
            qx = self.ax_or.quiver(0, 0, 0, \
                    blimp_x_vector[0], blimp_x_vector[1], blimp_x_vector[2], \
                    color='r')
            qx.ShowArrowHead = 'on'
            qy = self.ax_or.quiver(0, 0, 0, \
                    blimp_y_vector[0], blimp_y_vector[1], blimp_y_vector[2], \
                    color='g')
            qy.ShowArrowHead = 'on'
            qz = self.ax_or.quiver(0, 0, 0, \
                    blimp_z_vector[0], blimp_z_vector[1], blimp_z_vector[2], \
                    color='b')
            qz.ShowArrowHead = 'on'

            self.ax_or.set_xlim(-1.5, 1.5)
            self.ax_or.set_ylim(-1.5, 1.5)
            self.ax_or.set_zlim(-1.5, 1.5)
            self.ax_or.invert_yaxis()
            self.ax_or.invert_zaxis()
            self.ax_or.set_title('Orientation')
            self.ax_or.set_xlabel('x')
            self.ax_or.set_ylabel('y')
            self.ax_or.set_zlabel('z')
        
            self.ax_pos.cla()
            self.ax_pos.plot(sim.get_time_vec(), sim.get_var_history('x'))
            self.ax_pos.plot(sim.get_time_vec(), sim.get_var_history('y'))
            self.ax_pos.plot(sim.get_time_vec(), sim.get_var_history('z'))
            self.ax_pos.legend(['x', 'y', 'z'])
            self.ax_pos.set_ylabel('m')
            self.ax_pos.set_xlabel('Time (sec)')
            self.ax_pos.set_title('Position')

            self.ax_vel.cla()
            self.ax_vel.plot(sim.get_time_vec(), sim.get_var_history('vx'))
            self.ax_vel.plot(sim.get_time_vec(), sim.get_var_history('vy'))
            self.ax_vel.plot(sim.get_time_vec(), sim.get_var_history('vz'))
            self.ax_vel.legend(['vx', 'vy', 'vz'])
            self.ax_vel.set_ylabel('m/s')
            self.ax_vel.set_xlabel('Time (sec)')
            self.ax_vel.set_title('Velocity')

            self.ax_ang.cla()
            self.ax_ang.plot(sim.get_time_vec(), sim.get_var_history('phi') * 180/np.pi)
            self.ax_ang.plot(sim.get_time_vec(), sim.get_var_history('theta') * 180/np.pi)
            # self.ax_ang.plot(sim.get_time_vec(), sim.get_var_history('psi') * 180/np.pi)
            # self.ax_ang.legend(['phi', 'theta', 'psi'])
            self.ax_ang.legend(['phi', 'theta'])
            self.ax_ang.set_ylabel('deg')
            self.ax_ang.set_xlabel('Time (sec)')
            self.ax_ang.set_title('Angles')

            self.ax_w.cla()
            self.ax_w.plot(sim.get_time_vec(), sim.get_var_history('wx') * 180/np.pi)
            self.ax_w.plot(sim.get_time_vec(), sim.get_var_history('wy') * 180/np.pi)
            self.ax_w.plot(sim.get_time_vec(), sim.get_var_history('wz') * 180/np.pi)
            self.ax_w.legend(['wx', 'wy', 'wz'])
            self.ax_w.set_ylabel('deg/s')
            self.ax_w.set_xlabel('Time (sec)')
            self.ax_w.set_title('Angular Velocity')

            self.ax_err_pos.cla()
            self.ax_err_ang.cla()

            error = ctrl.get_error(sim)
            if error is not None:
                error_x = error[:, 0]
                error_y = error[:, 1]
                error_z = error[:, 2]
                error_psi = error[:, 3]

                self.ax_err_pos.plot(sim.get_time_vec(), error_x)
                self.ax_err_pos.plot(sim.get_time_vec(), error_y)
                self.ax_err_pos.plot(sim.get_time_vec(), error_z)

                self.ax_err_ang.plot(sim.get_time_vec(), error_psi * 180/np.pi)

            self.ax_err_pos.legend(['x', 'y', 'z'])
            self.ax_err_pos.set_ylabel('m')
            self.ax_err_pos.set_xlabel('Time (sec)')
            self.ax_err_pos.set_title('Position error')
            
            self.ax_err_ang.legend(['psi'])
            self.ax_err_ang.set_ylabel('deg')
            self.ax_err_ang.set_xlabel('Time (sec)')
            self.ax_err_ang.set_title('Angle error')

            self.ax_force.cla()
            self.ax_force.plot(sim.get_time_vec(), sim.get_u_history('fx'))
            self.ax_force.plot(sim.get_time_vec(), sim.get_u_history('fy'))
            self.ax_force.plot(sim.get_time_vec(), sim.get_u_history('fz'))
            self.ax_force.plot(sim.get_time_vec(), sim.get_u_history('tz'))
            self.ax_force.legend(['fx', 'fy', 'fz', 'tz'])
            self.ax_force.set_ylabel('N or N-m')
            self.ax_force.set_xlabel('Time (sec)')
            self.ax_force.set_title('Inputs')

            self.ax_solve.cla()
            self.ax_solve.plot(sim.get_time_vec(), sim.get_solve_time_history())
            self.ax_solve.set_ylabel('ns')
            self.ax_solve.set_xlabel('Time (sec)')
            self.ax_solve.set_title('Solve time')
        
        plt.draw()
        plt.pause(0.0000001)

    def block(self):
        plt.show(block=True)

    def on_close(self, a):
        self.window_closed = True

    def window_was_closed(self):
        return self.window_closed