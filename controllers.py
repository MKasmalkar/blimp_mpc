import numpy as np
from parameters import *
import control

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

    def get_ctrl_input(self, state, reference):
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
    # LQR that operates on full state of blimp using linearized model
    # from ACC paper code

    def __init__(self, dT):
        self.dT = dT

        # Using Bryson's Rule

        max_acceptable_theta = 5 * np.pi/180
        max_acceptable_phi = 5 * np.pi/180
        max_acceptable_wy = 0.1
        max_acceptable_wx = 0.1

        self.Q = np.array([[0.000001, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                           [0, 0.000001, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                           [0, 0, 0.000001, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                           [0, 0, 0, 1/max_acceptable_phi**2, 0, 0, 0, 0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 1/max_acceptable_theta**2, 0, 0, 0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0, 0.000001, 0, 0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0, 0, 0.000001, 0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0, 0, 0, 0.000001, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0, 0, 0, 0, 0.000001, 0, 0, 0],
                           [0, 0, 0, 0, 0, 0, 0, 0, 0, 1/max_acceptable_wx**2, 0, 0],
                           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1/max_acceptable_wy**2, 0],
                           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.000001]])
    

        max_acceptable_fx = 1
        max_acceptable_fy = 1
        max_acceptable_fz = 1

        self.R = np.array([[1/max_acceptable_fx**2, 0, 0 ,0],
                           [0, 1/max_acceptable_fy**2, 0, 0],
                           [0, 0, 1/max_acceptable_fz**2, 0],
                           [0, 0, 0, 1]])
        
    def get_ctrl_input(self, A_lin, B_lin, state):
        K = control.lqr(A_lin, B_lin, self.Q, self.R)[0]
        return -(K @ state.reshape((12,1))).reshape((4,1))

class NestedPID:
    # Nested PID controller, based off of this paper:
    # Modeling and Control of Swing Oscillation of Underactuated Indoor
    # Miniature Autonomous Blimps

    # Note: slightly different from original implementation in paper
    # Instead of regulating angular velocities to zero, we use a nested
    # PID loop to regulate the attitude to zero:
    # Outer PID loop that generates angular velocity setpoint from
    # angle error
    # Inner PID loop that generates control input from angular velocity
    # error

    def __init__(self, dT):
        self.dT = dT
        
        self.wy_error_int = 0
        self.wx_error_int = 0
        self.th_error_int = 0
        self.ph_error_int = 0

        self.wy_error_prev = 0
        self.wx_error_prev = 0
        self.th_error_prev = 0
        self.ph_error_prev = 0

        # Used to generate angular velocity setpoint
        self.kp_th = 0.16
        self.ki_th = 0
        self.kd_th = 0

        self.kp_ph = 0.16
        self.ki_ph = 0
        self.kd_ph = 0
        
        # Used to generate control input (force)
        self.kp_wy = 10
        self.ki_wy = 100
        self.kd_wy = 1
        
        self.kp_wx = 10
        self.ki_wx = 100
        self.kd_wx = 1

        self.th_sp = 0
        self.ph_sp = 0

    def get_ctrl_input(self, state):
        theta = state[4]
        
        th_error = theta - self.th_sp
        self.th_error_int += th_error * self.dT
        th_error_dot = (th_error - self.th_error_prev) / self.dT
        self.th_error_prev = th_error
        
        wy_target = - self.kp_th*th_error \
                    - self.ki_th*self.th_error_int \
                    - self.kd_th*th_error_dot
        
        wy = state[10]
        wy_error = wy - wy_target
        
        self.wy_error_int += wy_error*self.dT
        wy_error_dot = (wy_error - self.wy_error_prev) / self.dT
        self.wy_error_prev = wy_error

        f_x = - self.kp_wy*wy_error \
              - self.ki_wy*self.wy_error_int \
              - self.kd_wy*wy_error_dot

        phi = state[3]
        
        ph_error = phi - self.ph_sp
        self.ph_error_int += ph_error * self.dT
        ph_error_dot = (ph_error - self.ph_error_prev) / self.dT
        self.ph_error_prev = ph_error

        wx_target = - self.kp_ph*ph_error \
                    - self.ki_ph*self.ph_error_int \
                    - self.kd_ph*ph_error_dot
        wx = state[9]
        wx_error = wx - wx_target
        self.wx_error_int += wx_error*self.dT
        wx_error_dot = (wx_error - self.wx_error_prev) / self.dT
        self.wx_error_prev = wx_error

        f_y = - self.kp_wx*wx_error \
              - self.ki_wx*self.wx_error_int \
              - self.kd_wx*wx_error_dot
        
        return np.array([f_x.item(), f_y.item(), 0, 0]).reshape((4,1))
        
class StateFeedback:
    pass

class LinearizedAttitudeLQR:
    def __init__(self, dT):
        self.dT = dT
        
        max_allowable_theta = 0.05
        max_allowable_phi = 0.05
        
        max_allowable_wy = 0.02
        max_allowable_wx = 0.02

        max_allowable_vx = 0.5
        max_allowable_vy = 0.5

        max_allowable_vz = 0.5

        self.Q = np.array([
            [1/max_allowable_theta**2, 0, 0, 0, 0, 0, 0],
            [0, 1/max_allowable_wy**2, 0, 0, 0, 0, 0],
            [0, 0, 1/max_allowable_phi**2, 0, 0, 0, 0],
            [0, 0, 0, 1/max_allowable_wx**2, 0, 0, 0],
            [0, 0, 0, 0, 1/max_allowable_vx**2, 0, 0],
            [0, 0, 0, 0, 0, 1/max_allowable_vy**2, 0],
            [0, 0, 0, 0, 0, 0, 1/max_allowable_vz**2]
        ])

        self.R = np.eye(3)

    def get_ctrl_input(self, state):
        phi = state[3]
        theta = state[4]
        
        v_x__b = state[6]
        v_y__b = state[7]
        v_z__b = state[8]

        w_x__b = state[9]
        w_y__b = state[10]
        
        A_lin = np.array([
            [0, np.cos(phi), -1*w_y__b*np.sin(phi), 0, 0, 0, 0],
            [-0.154*np.cos(theta), 0.00979*v_z__b-0.0168, 0, 0, 0.495*v_z__b+3.9e-4, 0, 0.495*v_x__b+0.00979*w_y__b],
            [-(w_y__b*np.sin(phi))/(np.sin(theta)**2-1), np.sin(phi)*np.tan(theta), w_y__b*np.cos(phi)*np.tan(theta), 1, 0, 0, 0],
            [0.154*np.sin(phi)*np.sin(theta), 0, -0.154*np.cos(phi)*np.cos(theta), 0.00979*v_z__b-0.0168, 0, -0.495*v_z__b-3.9e-4, 0.00979*w_x__b-0.495*v_y__b],
            [0, -1.62*v_z__b, 0, 0,-0.0249, 0, -1.62*w_y__b],
            [0, 0, 0, 1.62*v_z__b, 0, -0.0249, 1.62*w_x__b],
            [0, 0.615*v_x__b+0.0244*w_y__b, 0, 0.0244*w_x__b-0.615*v_y__b, 0.615*w_y__b, -0.615*w_x__b, -0.064]
        ])
        
        B_lin = np.array([
            [0, 0, 0],
            [0.0398, 0, 0],
            [0, 0, 0],
            [0, -0.0398, 0],
            [2.17, 0, 0],
            [0, 2.17, 0],
            [0, 0, 1.33]
        ])

        K = control.lqr(A_lin, B_lin, self.Q, self.R)[0]
        f_out = -K @ np.array([theta, w_y__b, phi, w_x__b, v_x__b, v_y__b, v_z__b]).reshape((7, 1))
        u_swing = np.array([f_out[0].item(),
                            f_out[1].item(),
                            f_out[2].item(), 0]).reshape((4, 1))
        
        return u_swing