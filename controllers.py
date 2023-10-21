import numpy as np
from parameters import *

class SwingReducingController:
    # Nested controller from paper:
    # Swing-Reducing Flight Control System for an Underactuated Indoor
    # Miniature Autonomous Blimp

    # Proportional control operating on position error used to generate
    # velocity setpoint
    # PI loop to to operating on velocity error used to generate
    # attitude setpoint
    # State feedback on attitude error and angular velocity error to
    # generate control input
    # Controls for x and y oscillations are independent

    def __init__(self, dT):

        self.dT = dT

        # Position proportional gains
        self.kp_x = 0.1
        self.kp_y = 0.1
        self.kp_z = 0.1

        # Velocity proportional and integral gains
        self.kp_vx = 0.1
        self.kp_vy = 0.1
        self.ki_vx = 0.1
        self.ki_vy = 0.1

        # Angle proportional gains
        self.kp_th = 1.0
        self.kp_wy = 1.0
        self.kp_ph = 1.0
        self.kp_wx = 1.0

        # Error integrals
        self.v_x_error_int = 0
        self.v_y_error_int = 0
        self.theta_error_int = 0
        self.phi_error_int = 0

    def get_ctrl_input(self, state, state_dot, reference):
        # x control
        x = state[0]
        x_target = reference[0]
        x_error = x - x_target

        v_x_sp = -self.kp_x * x_error
        v_x = state[6]
        v_x_error = v_x - v_x_sp
        self.v_x_error_int += v_x_error * self.dT

        theta_sp = np.arcsin(D_vxy__CB * r_z_tg__b / (r_z_gb__b * m_RB * g_acc) * v_x_sp) \
                    - self.kp_vx * v_x_error - self.ki_vx * self.v_x_error_int
        theta = state[4]
        theta_error = theta - theta_sp
        self.theta_error_int += theta_error * self.dT

        w_y_sp = 0
        w_y = state[10]
        w_y_error = w_y - w_y_sp

        f_x = r_z_gb__b / r_z_tg__b * m_RB * g_acc * np.sin(theta_sp) \
                - self.kp_th * theta_error \
                - self.kp_wy * w_y_error

        # y control
        y = state[1]
        y_target = reference[1]
        y_error = y - y_target

        v_y_sp = -self.kp_y * y_error
        v_y = state[7]
        v_y_error = v_y - v_y_sp
        self.v_y_error_int += v_y_error * self.dT

        phi_sp = np.arcsin(D_vxy__CB * r_z_tg__b / (r_z_gb__b * m_RB * g_acc) * v_y_sp) \
                    - self.kp_vy * v_y_error - self.ki_vy * self.v_y_error_int
        phi = state[3]
        phi_error = phi - phi_sp
        self.phi_error_int += phi_error * self.dT

        w_x_sp = 0
        w_x = state[9]
        w_x_error = w_x - w_x_sp

        f_y = r_z_gb__b / r_z_tg__b * m_RB * g_acc * np.sin(phi_sp) \
                - self.kp_ph * phi_error \
                - self.kp_wx * w_x_error
        
        # z control
        z = state[2]
        z_target = reference[2]
        z_error = z - z_target
        f_z = -self.kp_z * z_error
        
        return np.array([f_x.item(), f_y.item(), f_z.item(), 0]).reshape((4, 1))

class LinearizedDynamicsLQR:
    pass

class NestedPID:
    pass

class StateFeedback:
    pass

class LinearizedAttitudeLQR:
    pass