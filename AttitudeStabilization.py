from BlimpController import BlimpController
import numpy as np
import control
from parameters import *

class AttitudeStabilization(BlimpController):

    def __init__(self, dT):
        super().__init__(dT)
        
        A_att = np.array([
            [-0.0249, 0, 0, 0],
            [0, -0.0168, 0, -0.1535],
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ])

        B_att_x = np.array([
            [2.1668],
            [0.0738],
            [0],
            [0]
        ])

        B_att_y = np.array([
            [2.1668],
            [-0.0738],
            [0],
            [0]
        ])

        Q = np.diag([1, 1, 1, 250])
        R = 1

        self.K_x = control.lqr(A_att, B_att_x, Q, R)[0]
        self.K_y = control.lqr(A_att, B_att_y, Q, R)[0]

    def init_sim(self, sim):
        sim.set_var('theta', 5*np.pi/180)
        sim.set_var('phi', 5*np.pi/180)
        #sim.set_var('psi', np.pi/2)

    def get_ctrl_action(self, sim):

        sim.start_timer()

        n = sim.get_current_timestep()

        # Extract state variables
        x = sim.get_var('x')
        y = sim.get_var('y')
        z = sim.get_var('z')

        phi = sim.get_var('phi')
        theta = sim.get_var('theta')
        psi = sim.get_var('psi')
        
        v_x__b = sim.get_var('vx')
        v_y__b = sim.get_var('vy')
        v_z__b = sim.get_var('vz')

        w_x__b = sim.get_var('wx')
        w_y__b = sim.get_var('wy')
        w_z__b = sim.get_var('wz')

        x_dot = sim.get_var_dot('x')
        y_dot = sim.get_var_dot('y')
        z_dot = sim.get_var_dot('z')

        phi_dot = sim.get_var_dot('psi')
        theta_dot = sim.get_var_dot('theta')
        psi_dot = sim.get_var_dot('psi')
        
        # pitch_state = np.array([
        #     [v_x__b],
        #     [w_y__b],
        #     [x - 1],
        #     [theta]
        # ])

        # roll_state = np.array([
        #     [v_y__b],
        #     [w_x__b],
        #     [y - 1],
        #     [phi]
        # ])

        # f_x = -self.K_x @ pitch_state
        # f_y = -self.K_y @ roll_state

        # u_swing = np.array([f_x.item(),
        #                     f_y.item(),
        #                     0,
        #                     0]).reshape((4, 1))
        
        # u = u_swing

        v_b = np.array([
            [v_x__b],
            [v_y__b],
            [v_z__b]
        ])

        v_n = R_b__n(0, 0, psi) @ v_b

        pitch_state = np.array([
            [v_n[0].item()],
            [theta_dot],
            [x - 1],
            [theta]
        ])

        roll_state = np.array([
            [v_n[1].item()],
            [phi_dot],
            [y - 1],
            [phi]
        ])

        f_x = -self.K_x @ pitch_state
        f_y = -self.K_y @ roll_state

        u_swing = np.array([f_x.item(),
                            f_y.item(),
                            0,
                            0]).reshape((4, 1))
        
        # A matrix is generated assuming psi = 0
        # need to perform some rotations to account for this

        u_rot = R_b__n_inv(0, 0, psi) @ np.array([u_swing[0].item(),
                                                    u_swing[1].item(),
                                                    u_swing[2].item()]).T

        u = np.array([
            [u_rot[0].item()],
            [u_rot[1].item()],
            [u_rot[2].item()],
            [u_swing[3].item()]
        ])

        sim.end_timer()

        return u
    
    def get_error(self, sim):
        n = sim.get_current_timestep() + 1
        return np.array([
                sim.get_var_history('x'),
                sim.get_var_history('y'),
                sim.get_var_history('z'),
                sim.get_var_history('psi')
        ]).T