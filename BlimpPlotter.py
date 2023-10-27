import matplotlib.pyplot as plt
import numpy as np

class BlimpPlotter():
    
    def __init__(self):
        self.plotting = False

    def init_plot(self, title):
        self.fig = plt.figure(title, figsize=(8, 6))
        plt.ion()

        self.ax_or = self.fig.add_subplot(321, projection='3d')
        self.ax_or.grid()
        self.ax_3d = self.fig.add_subplot(322, projection='3d')
        self.ax_3d.grid()
        
        self.ax_pos = self.fig.add_subplot(323)
        self.ax_ang = self.fig.add_subplot(324)
        self.ax_vel = self.fig.add_subplot(325)
        self.ax_w = self.fig.add_subplot(326)

        plt.subplots_adjust(wspace=0.25)
        plt.subplots_adjust(hspace=0.75)

        self.fig.canvas.mpl_connect('close_event', self.on_close)

        self.plotting = True
        self.window_closed = False

    def update_plot(self, sim, ctrl):

        if not self.plotting: return

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
        self. ax_or.invert_zaxis()

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

        self.ax_pos.cla()
        self.ax_pos.plot(sim.get_time_vec(), sim.get_var_history('x'))
        self.ax_pos.plot(sim.get_time_vec(), sim.get_var_history('y'))
        self.ax_pos.plot(sim.get_time_vec(), sim.get_var_history('z'))
        self.ax_pos.legend(['x', 'y', 'z'])
        self.ax_pos.set_ylabel('m')
        self.ax_pos.set_title('Position')

        self.ax_vel.cla()
        self.ax_vel.plot(sim.get_time_vec(), sim.get_var_history('vx'))
        self.ax_vel.plot(sim.get_time_vec(), sim.get_var_history('vy'))
        self.ax_vel.plot(sim.get_time_vec(), sim.get_var_history('vz'))
        self.ax_vel.legend(['vx', 'vy', 'vz'])
        self.ax_vel.set_ylabel('m/s')
        self.ax_vel.set_title('Velocity')

        self.ax_ang.cla()
        self.ax_ang.plot(sim.get_time_vec(), sim.get_var_history('phi') * 180/np.pi)
        self.ax_ang.plot(sim.get_time_vec(), sim.get_var_history('theta') * 180/np.pi)
        self.ax_ang.plot(sim.get_time_vec(), sim.get_var_history('psi') * 180/np.pi)
        self.ax_ang.legend(['phi', 'theta', 'psi'])
        self.ax_ang.set_ylabel('deg')
        self.ax_ang.set_title('Angles')

        self.ax_w.cla()
        self.ax_w.plot(sim.get_time_vec(), sim.get_var_history('wx') * 180/np.pi)
        self.ax_w.plot(sim.get_time_vec(), sim.get_var_history('wy') * 180/np.pi)
        self.ax_w.plot(sim.get_time_vec(), sim.get_var_history('wz') * 180/np.pi)
        self.ax_w.legend(['wx', 'wy', 'wz'])
        self.ax_w.set_ylabel('deg/s')
        self.ax_w.set_title('Angular Velocity')

        plt.draw()
        plt.pause(0.000000000001)

    def block(self):
        plt.show(block=True)

    def on_close(self, a):
        self.window_closed = True

    def window_was_closed(self):
        return self.window_closed