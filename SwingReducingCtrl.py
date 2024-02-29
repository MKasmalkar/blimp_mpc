from BlimpController import BlimpController
import numpy as np
import control
from parameters import *
from PID import PID
import math
import sys

class SwingReducingCtrl(BlimpController):

    def __init__(self, dT, skip_derivatives=True):
        super().__init__(dT)

        self.x_ref = 1
        self.y_ref = 0.5
        self.z_ref = -1

        # Position proportional gains
        self.kp_x = 0.02
        self.kp_y = 0.02
        self.kp_z = 0.001

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

    def init_sim(self, sim):
        pass

    def get_ctrl_action(self, sim):

        sim.start_timer()

        # Extract state variables
        x = sim.get_var('x')
        y = sim.get_var('y')
        z = sim.get_var('z')

        phi = sim.get_var('phi')
        theta = sim.get_var('theta')
        
        v_x = sim.get_var('vx')
        v_y = sim.get_var('vy')
        
        w_x = sim.get_var('wx')
        w_y = sim.get_var('wy')
        
        # x control
        x_error = x - self.x_ref

        v_x_sp = -self.kp_x * x_error
        v_x_error = v_x - v_x_sp
        self.v_x_error_int += v_x_error * self.dT

        theta_sp = np.arcsin(D_vxy__CB * r_z_tg__b / (r_z_gb__b * m_RB * g_acc) * v_x_sp) \
                    - self.kp_vx * v_x_error - self.ki_vx * self.v_x_error_int
        theta_error = theta - theta_sp
        self.theta_error_int += theta_error * self.dT

        w_y_sp = 0
        w_y_error = w_y - w_y_sp

        f_x = r_z_gb__b / r_z_tg__b * m_RB * g_acc * np.sin(theta_sp) \
                - self.kp_th * theta_error \
                - self.kp_wy * w_y_error

        # y control
        y_error = y - self.y_ref

        v_y_sp = -self.kp_y * y_error
        v_y_error = v_y - v_y_sp
        self.v_y_error_int += v_y_error * self.dT

        phi_sp = np.arcsin(D_vxy__CB * r_z_tg__b / (r_z_gb__b * m_RB * g_acc) * v_y_sp) \
                    - self.kp_vy * v_y_error - self.ki_vy * self.v_y_error_int
        phi_error = phi - phi_sp
        self.phi_error_int += phi_error * self.dT

        w_x_sp = 0
        w_x_error = w_x - w_x_sp

        f_y = r_z_gb__b / r_z_tg__b * m_RB * g_acc * np.sin(phi_sp) \
                - self.kp_ph * phi_error \
                - self.kp_wx * w_x_error

        # z control
        z_error = z - self.z_ref
        f_z = -self.kp_z * z_error
    
        u = np.array([f_x, f_y, f_z, 0]).reshape((4, 1))

        if (np.any(np.isnan(u))):
            print("NaN encountered in input")
            sys.exit(0)

        sim.end_timer()

        return u
        
    def get_error(self, sim):
        n = sim.get_current_timestep() + 1

        return np.array([
            sim.get_var_history('x') - np.zeros(n),
            sim.get_var_history('y') - np.zeros(n),
            sim.get_var_history('z') - np.zeros(n),
            sim.get_var_history('psi') - np.zeros(n)
        ]).T
