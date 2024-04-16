import gurobipy as gp
from gurobipy import GRB
import numpy as np
import time
from BlimpController import BlimpController
from parameters import *
import sys

class CBF(BlimpController):

    def __init__(self, dT):
        super().__init__(dT)

        self.order = 12
        self.num_inputs = 4
        self.num_outputs = 6
        
        # Trajectory definition
        TRACKING_TIME = 100

        time_vec = np.arange(0, TRACKING_TIME + 1/self.dT, self.dT)
        
        x0 = 0
        y0 = 0
        z0 = 0
        psi0 = 0
        
        self.z0 = z0

        traj_z_bounds = 1
        self.safe_z_bounds = 0.5
        
        traj_z_max = z0 + traj_z_bounds / 2
        traj_z_min = z0 - traj_z_bounds / 2
        
        safe_z_max = z0 + self.safe_z_bounds / 2
        safe_z_min = z0 - self.safe_z_bounds / 2
        
        x_speed = 0.05
        
        self.traj_x = x0 + x_speed*time_vec
        self.traj_y = y0 * np.ones(len(time_vec))
        self.traj_psi = psi0 * np.ones(len(time_vec))
        
        self.traj_x_dot = x_speed * np.ones(len(time_vec))
        self.traj_y_dot = np.zeros(len(time_vec))
        self.traj_psi_dot = np.zeros(len(time_vec))
        
        self.traj_z = np.empty(len(time_vec))
        self.traj_z_dot = np.empty(len(time_vec))
        
        z_speed = 0.02
        z_ctr = z0
        direction = -1
        for i in range(len(time_vec)):
            if direction == 1 and z_ctr > traj_z_max:
                direction = -1
            elif direction == -1 and z_ctr < traj_z_min:
                direction = 1
            
            self.traj_z[i]     = z_ctr    
            self.traj_z_dot[i] = direction * z_speed
            
            z_ctr = z_ctr + direction * z_speed * self.dT
            
        self.traj_x_ddot = np.zeros(len(time_vec))
        self.traj_y_ddot = np.zeros(len(time_vec))
        self.traj_z_ddot = np.zeros(len(time_vec))
        self.traj_psi_ddot = np.zeros(len(time_vec))
        
        self.ran_before = False
        
    def get_ctrl_action(self, sim):
        
        if not self.ran_before:
            # Trajectory definition
            TRACKING_TIME = 100

            time_vec = np.arange(0, TRACKING_TIME + 1/self.dT, self.dT)
            
            x0 = sim.get_var('x')
            y0 = sim.get_var('y')
            z0 = sim.get_var('z')
            psi0 = sim.get_var('psi')
            
            self.z0 = z0

            traj_z_bounds = 1
            self.safe_z_bounds = 0.5
            
            traj_z_max = z0 + traj_z_bounds / 2
            traj_z_min = z0 - traj_z_bounds / 2
            
            safe_z_max = z0 + self.safe_z_bounds / 2
            safe_z_min = z0 - self.safe_z_bounds / 2
            
            x_speed = 0.05
            
            self.traj_x = x0 + x_speed*time_vec
            self.traj_y = y0 * np.ones(len(time_vec))
            self.traj_psi = psi0 * np.ones(len(time_vec))
            
            self.traj_x_dot = x_speed * np.ones(len(time_vec))
            self.traj_y_dot = np.zeros(len(time_vec))
            self.traj_psi_dot = np.zeros(len(time_vec))
            
            self.traj_z = np.empty(len(time_vec))
            self.traj_z_dot = np.empty(len(time_vec))
            
            z_speed = 0.02
            z_ctr = z0
            direction = -1
            for i in range(len(time_vec)):
                if direction == 1 and z_ctr > traj_z_max:
                    direction = -1
                elif direction == -1 and z_ctr < traj_z_min:
                    direction = 1
                
                self.traj_z[i]     = z_ctr    
                self.traj_z_dot[i] = direction * z_speed
                
                z_ctr = z_ctr + direction * z_speed * self.dT
                
            self.traj_x_ddot = np.zeros(len(time_vec))
            self.traj_y_ddot = np.zeros(len(time_vec))
            self.traj_z_ddot = np.zeros(len(time_vec))
            self.traj_psi_ddot = np.zeros(len(time_vec))
            
            self.env = gp.Env(empty=True)
            self.env.setParam('OutputFlag', 0)
            self.env.setParam('LogToConsole', 0)
            self.env.start()

            self.m = gp.Model(env=self.env)

            self.mu = self.m.addMVar(shape=(self.num_inputs, 1),
                                     lb=-GRB.INFINITY, ub=GRB.INFINITY)

            self.cbf_constraint = self.m.addConstr(0 == 0)

            self.gamma = 1
            
            self.ran_before = True

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

        ## Feedback-linearized tracking controller

        A = np.array([[v_y__b*(- np.cos(phi)*np.cos(psi)*psi_dot + np.sin(phi)*np.sin(psi)*phi_dot + ((np.cos(phi)*np.sin(psi) - np.cos(psi)*np.sin(phi)*np.sin(theta))*(D_vxy__CB*I_x + m_RB*m_z*r_z_gb__b*v_z__b))/(I_x*m_y - m_RB**2*r_z_gb__b**2) + np.cos(phi)*np.cos(psi)*np.sin(theta)*phi_dot + np.cos(psi)*np.cos(theta)*np.sin(phi)*theta_dot - np.sin(phi)*np.sin(psi)*np.sin(theta)*psi_dot) + w_z__b*(((I_x*(m_x*v_x__b + m_RB*r_z_gb__b*w_y__b))/(I_x*m_y - m_RB**2*r_z_gb__b**2) - (m_RB*r_z_gb__b*(I_y*w_y__b + m_RB*r_z_gb__b*v_x__b))/(I_x*m_y - m_RB**2*r_z_gb__b**2))*(np.cos(phi)*np.sin(psi) - np.cos(psi)*np.sin(phi)*np.sin(theta)) + np.cos(psi)*np.cos(theta)*((I_y*(m_y*v_y__b - m_RB*r_z_gb__b*w_x__b))/(I_y*m_x - m_RB**2*r_z_gb__b**2) + (m_RB*r_z_gb__b*(I_x*w_x__b - m_RB*r_z_gb__b*v_y__b))/(I_y*m_x - m_RB**2*r_z_gb__b**2))) - v_z__b*((D_vz__CB*(np.sin(phi)*np.sin(psi) + np.cos(phi)*np.cos(psi)*np.sin(theta)))/m_z - np.cos(phi)*np.sin(psi)*phi_dot - np.cos(psi)*np.sin(phi)*psi_dot + np.cos(psi)*np.sin(phi)*np.sin(theta)*phi_dot + np.cos(phi)*np.sin(psi)*np.sin(theta)*psi_dot - np.cos(phi)*np.cos(psi)*np.cos(theta)*theta_dot + (m_RB*r_z_gb__b*(m_y*v_y__b - m_RB*r_z_gb__b*w_x__b)*(np.cos(phi)*np.sin(psi) - np.cos(psi)*np.sin(phi)*np.sin(theta)))/(I_x*m_y - m_RB**2*r_z_gb__b**2) - (m_RB*r_z_gb__b*np.cos(psi)*np.cos(theta)*(m_x*v_x__b + m_RB*r_z_gb__b*w_y__b))/(I_y*m_x - m_RB**2*r_z_gb__b**2)) - w_x__b*(((m_y*v_y__b - m_RB*r_z_gb__b*w_x__b)*(np.sin(phi)*np.sin(psi) + np.cos(phi)*np.cos(psi)*np.sin(theta)))/m_z + ((np.cos(phi)*np.sin(psi) - np.cos(psi)*np.sin(phi)*np.sin(theta))*(I_x*m_z*v_z__b - D_omega_xy__CB*m_RB*r_z_gb__b))/(I_x*m_y - m_RB**2*r_z_gb__b**2) + (I_z*m_RB*r_z_gb__b*np.cos(psi)*np.cos(theta)*w_z__b)/(I_y*m_x - m_RB**2*r_z_gb__b**2)) + w_y__b*(((m_x*v_x__b + m_RB*r_z_gb__b*w_y__b)*(np.sin(phi)*np.sin(psi) + np.cos(phi)*np.cos(psi)*np.sin(theta)))/m_z - (np.cos(psi)*np.cos(theta)*(I_y*m_z*v_z__b - D_omega_xy__CB*m_RB*r_z_gb__b))/(I_y*m_x - m_RB**2*r_z_gb__b**2) + (I_z*m_RB*r_z_gb__b*w_z__b*(np.cos(phi)*np.sin(psi) - np.cos(psi)*np.sin(phi)*np.sin(theta)))/(I_x*m_y - m_RB**2*r_z_gb__b**2)) - v_x__b*(np.cos(theta)*np.sin(psi)*psi_dot + np.cos(psi)*np.sin(theta)*theta_dot + (np.cos(psi)*np.cos(theta)*(D_vxy__CB*I_y + m_RB*m_z*r_z_gb__b*v_z__b))/(I_y*m_x - m_RB**2*r_z_gb__b**2)) + (f_g*m_RB*r_z_gb__b**2*np.cos(theta)*np.sin(phi)*(np.cos(phi)*np.sin(psi) - np.cos(psi)*np.sin(phi)*np.sin(theta)))/(I_x*m_y - m_RB**2*r_z_gb__b**2) + (f_g*m_RB*r_z_gb__b**2*np.cos(psi)*np.cos(theta)*np.sin(theta))/(I_y*m_x - m_RB**2*r_z_gb__b**2)],
                    [v_z__b*((D_vz__CB*(np.cos(psi)*np.sin(phi) - np.cos(phi)*np.sin(psi)*np.sin(theta)))/m_z - np.cos(phi)*np.cos(psi)*phi_dot + np.sin(phi)*np.sin(psi)*psi_dot + np.cos(phi)*np.cos(psi)*np.sin(theta)*psi_dot + np.cos(phi)*np.cos(theta)*np.sin(psi)*theta_dot - np.sin(phi)*np.sin(psi)*np.sin(theta)*phi_dot + (m_RB*r_z_gb__b*(m_y*v_y__b - m_RB*r_z_gb__b*w_x__b)*(np.cos(phi)*np.cos(psi) + np.sin(phi)*np.sin(psi)*np.sin(theta)))/(I_x*m_y - m_RB**2*r_z_gb__b**2) + (m_RB*r_z_gb__b*np.cos(theta)*np.sin(psi)*(m_x*v_x__b + m_RB*r_z_gb__b*w_y__b))/(I_y*m_x - m_RB**2*r_z_gb__b**2)) - w_z__b*(((I_x*(m_x*v_x__b + m_RB*r_z_gb__b*w_y__b))/(I_x*m_y - m_RB**2*r_z_gb__b**2) - (m_RB*r_z_gb__b*(I_y*w_y__b + m_RB*r_z_gb__b*v_x__b))/(I_x*m_y - m_RB**2*r_z_gb__b**2))*(np.cos(phi)*np.cos(psi) + np.sin(phi)*np.sin(psi)*np.sin(theta)) - np.cos(theta)*np.sin(psi)*((I_y*(m_y*v_y__b - m_RB*r_z_gb__b*w_x__b))/(I_y*m_x - m_RB**2*r_z_gb__b**2) + (m_RB*r_z_gb__b*(I_x*w_x__b - m_RB*r_z_gb__b*v_y__b))/(I_y*m_x - m_RB**2*r_z_gb__b**2))) - v_y__b*(np.cos(psi)*np.sin(phi)*phi_dot + np.cos(phi)*np.sin(psi)*psi_dot + ((np.cos(phi)*np.cos(psi) + np.sin(phi)*np.sin(psi)*np.sin(theta))*(D_vxy__CB*I_x + m_RB*m_z*r_z_gb__b*v_z__b))/(I_x*m_y - m_RB**2*r_z_gb__b**2) - np.cos(phi)*np.sin(psi)*np.sin(theta)*phi_dot - np.cos(psi)*np.sin(phi)*np.sin(theta)*psi_dot - np.cos(theta)*np.sin(phi)*np.sin(psi)*theta_dot) + w_x__b*(((m_y*v_y__b - m_RB*r_z_gb__b*w_x__b)*(np.cos(psi)*np.sin(phi) - np.cos(phi)*np.sin(psi)*np.sin(theta)))/m_z + ((np.cos(phi)*np.cos(psi) + np.sin(phi)*np.sin(psi)*np.sin(theta))*(I_x*m_z*v_z__b - D_omega_xy__CB*m_RB*r_z_gb__b))/(I_x*m_y - m_RB**2*r_z_gb__b**2) - (I_z*m_RB*r_z_gb__b*np.cos(theta)*np.sin(psi)*w_z__b)/(I_y*m_x - m_RB**2*r_z_gb__b**2)) - w_y__b*(((m_x*v_x__b + m_RB*r_z_gb__b*w_y__b)*(np.cos(psi)*np.sin(phi) - np.cos(phi)*np.sin(psi)*np.sin(theta)))/m_z + (np.cos(theta)*np.sin(psi)*(I_y*m_z*v_z__b - D_omega_xy__CB*m_RB*r_z_gb__b))/(I_y*m_x - m_RB**2*r_z_gb__b**2) + (I_z*m_RB*r_z_gb__b*w_z__b*(np.cos(phi)*np.cos(psi) + np.sin(phi)*np.sin(psi)*np.sin(theta)))/(I_x*m_y - m_RB**2*r_z_gb__b**2)) - v_x__b*(- np.cos(psi)*np.cos(theta)*psi_dot + np.sin(psi)*np.sin(theta)*theta_dot + (np.cos(theta)*np.sin(psi)*(D_vxy__CB*I_y + m_RB*m_z*r_z_gb__b*v_z__b))/(I_y*m_x - m_RB**2*r_z_gb__b**2)) - (f_g*m_RB*r_z_gb__b**2*np.cos(theta)*np.sin(phi)*(np.cos(phi)*np.cos(psi) + np.sin(phi)*np.sin(psi)*np.sin(theta)))/(I_x*m_y - m_RB**2*r_z_gb__b**2) + (f_g*m_RB*r_z_gb__b**2*np.cos(theta)*np.sin(psi)*np.sin(theta))/(I_y*m_x - m_RB**2*r_z_gb__b**2)],
                    [w_x__b*((np.cos(theta)*np.sin(phi)*(I_x*m_z*v_z__b - D_omega_xy__CB*m_RB*r_z_gb__b))/(I_x*m_y - m_RB**2*r_z_gb__b**2) - (np.cos(phi)*np.cos(theta)*(m_y*v_y__b - m_RB*r_z_gb__b*w_x__b))/m_z + (I_z*m_RB*r_z_gb__b*np.sin(theta)*w_z__b)/(I_y*m_x - m_RB**2*r_z_gb__b**2)) + w_y__b*((np.sin(theta)*(I_y*m_z*v_z__b - D_omega_xy__CB*m_RB*r_z_gb__b))/(I_y*m_x - m_RB**2*r_z_gb__b**2) + (np.cos(phi)*np.cos(theta)*(m_x*v_x__b + m_RB*r_z_gb__b*w_y__b))/m_z - (I_z*m_RB*r_z_gb__b*np.cos(theta)*np.sin(phi)*w_z__b)/(I_x*m_y - m_RB**2*r_z_gb__b**2)) - v_x__b*(np.cos(theta)*theta_dot - (np.sin(theta)*(D_vxy__CB*I_y + m_RB*m_z*r_z_gb__b*v_z__b))/(I_y*m_x - m_RB**2*r_z_gb__b**2)) - w_z__b*(np.cos(theta)*np.sin(phi)*((I_x*(m_x*v_x__b + m_RB*r_z_gb__b*w_y__b))/(I_x*m_y - m_RB**2*r_z_gb__b**2) - (m_RB*r_z_gb__b*(I_y*w_y__b + m_RB*r_z_gb__b*v_x__b))/(I_x*m_y - m_RB**2*r_z_gb__b**2)) + (np.sin(theta)*(I_y*m_y*v_y__b - m_RB**2*r_z_gb__b**2*v_y__b + I_x*m_RB*r_z_gb__b*w_x__b - I_y*m_RB*r_z_gb__b*w_x__b))/(I_y*m_x - m_RB**2*r_z_gb__b**2)) - v_z__b*(np.cos(theta)*np.sin(phi)*phi_dot + np.cos(phi)*np.sin(theta)*theta_dot + (D_vz__CB*np.cos(phi)*np.cos(theta))/m_z + (m_RB*r_z_gb__b*np.sin(theta)*(m_x*v_x__b + m_RB*r_z_gb__b*w_y__b))/(I_y*m_x - m_RB**2*r_z_gb__b**2) - (m_RB*r_z_gb__b*np.cos(theta)*np.sin(phi)*(m_y*v_y__b - m_RB*r_z_gb__b*w_x__b))/(I_x*m_y - m_RB**2*r_z_gb__b**2)) - v_y__b*(- np.cos(phi)*np.cos(theta)*phi_dot + np.sin(phi)*np.sin(theta)*theta_dot + (np.cos(theta)*np.sin(phi)*(D_vxy__CB*I_x + m_RB*m_z*r_z_gb__b*v_z__b))/(I_x*m_y - m_RB**2*r_z_gb__b**2)) - (f_g*m_RB*r_z_gb__b**2*np.sin(theta)**2)/(I_y*m_x - m_RB**2*r_z_gb__b**2) + (f_g*m_RB*r_z_gb__b**2*np.sin(phi)**2*(np.sin(theta)**2 - 1))/(I_x*m_y - m_RB**2*r_z_gb__b**2)],
                    [w_y__b*((np.cos(phi)*phi_dot)/np.cos(theta) + (np.cos(phi)*(I_x*w_x__b - m_RB*r_z_gb__b*v_y__b))/(I_z*np.cos(theta)) - (np.sin(phi)*(D_omega_xy__CB*m_x - m_RB*m_z*r_z_gb__b*v_z__b))/(np.cos(theta)*(I_y*m_x - m_RB**2*r_z_gb__b**2)) - (np.sin(phi)*np.sin(theta)*theta_dot)/(np.sin(theta)**2 - 1)) - w_z__b*((np.sin(phi)*phi_dot)/np.cos(theta) + (D_omega_z__CB*np.cos(phi))/(I_z*np.cos(theta)) - (np.cos(phi)*np.sin(theta)*theta_dot)/np.cos(theta)**2 + (np.sin(phi)*(I_x*m_x*w_x__b - m_RB**2*r_z_gb__b**2*w_x__b - m_RB*m_x*r_z_gb__b*v_y__b + m_RB*m_y*r_z_gb__b*v_y__b))/(np.cos(theta)*(I_y*m_x - m_RB**2*r_z_gb__b**2))) - v_x__b*((np.cos(phi)*(m_y*v_y__b - m_RB*r_z_gb__b*w_x__b))/(I_z*np.cos(theta)) - (np.sin(phi)*(m_x*m_z*v_z__b + D_vxy__CB*m_RB*r_z_gb__b))/(np.cos(theta)*(I_y*m_x - m_RB**2*r_z_gb__b**2))) - w_x__b*((np.cos(phi)*(I_y*w_y__b + m_RB*r_z_gb__b*v_x__b))/(I_z*np.cos(theta)) - (I_z*m_x*np.sin(phi)*w_z__b)/(np.cos(theta)*(I_y*m_x - m_RB**2*r_z_gb__b**2))) + (np.cos(phi)*v_y__b*(m_x*v_x__b + m_RB*r_z_gb__b*w_y__b))/(I_z*np.cos(theta)) - (m_x*np.sin(phi)*v_z__b*(m_x*v_x__b + m_RB*r_z_gb__b*w_y__b))/(np.cos(theta)*(I_y*m_x - m_RB**2*r_z_gb__b**2)) - (f_g*m_x*r_z_gb__b*np.sin(phi)*np.sin(theta))/(np.cos(theta)*(I_y*m_x - m_RB**2*r_z_gb__b**2))]
                    ])
        
        Binv = np.array([[(np.cos(psi)*np.cos(theta)*(I_y*m_x - m_RB**2*r_z_gb__b**2))/(I_y - m_RB*r_z_gb__b*r_z_tg__b), (np.cos(theta)*np.sin(psi)*(I_y*m_x - m_RB**2*r_z_gb__b**2))/(I_y - m_RB*r_z_gb__b*r_z_tg__b), -(np.sin(theta)*(I_y*m_x - m_RB**2*r_z_gb__b**2))/(I_y - m_RB*r_z_gb__b*r_z_tg__b), 0],
                        [-((I_x*m_y - m_RB**2*r_z_gb__b**2)*(np.cos(phi)*np.sin(psi) - np.cos(psi)*np.sin(phi)*np.sin(theta)))/(I_x - m_RB*r_z_gb__b*r_z_tg__b), ((I_x*m_y - m_RB**2*r_z_gb__b**2)*(np.cos(phi)*np.cos(psi) + np.sin(phi)*np.sin(psi)*np.sin(theta)))/(I_x - m_RB*r_z_gb__b*r_z_tg__b), (np.cos(theta)*np.sin(phi)*(I_x*m_y - m_RB**2*r_z_gb__b**2))/(I_x - m_RB*r_z_gb__b*r_z_tg__b), 0],
                        [m_z*np.sin(phi)*np.sin(psi) + m_z*np.cos(phi)*np.cos(psi)*np.sin(theta), m_z*np.cos(phi)*np.sin(psi)*np.sin(theta) - m_z*np.cos(psi)*np.sin(phi), m_z*np.cos(phi)*np.cos(theta), 0],
                        [(I_z*np.cos(psi)*np.cos(theta)*np.sin(phi)*(m_RB*r_z_gb__b - m_x*r_z_tg__b))/(np.cos(phi)*(I_y - m_RB*r_z_gb__b*r_z_tg__b)), (I_z*np.cos(theta)*np.sin(phi)*np.sin(psi)*(m_RB*r_z_gb__b - m_x*r_z_tg__b))/(np.cos(phi)*(I_y - m_RB*r_z_gb__b*r_z_tg__b)), -(I_z*np.sin(phi)*np.sin(theta)*(m_RB*r_z_gb__b - m_x*r_z_tg__b))/(np.cos(phi)*(I_y - m_RB*r_z_gb__b*r_z_tg__b)), (I_z*np.cos(theta))/np.cos(phi)]
                        ])
        
        zeta1 = np.array([[x],
                          [y],
                          [z],
                          [psi]])
        zeta2 = np.array([[x_dot],
                          [y_dot],
                          [z_dot],
                          [psi_dot]])

        yd = np.array([[self.traj_x[n]],
                       [self.traj_y[n]],
                       [self.traj_z[n]],
                       [self.traj_psi[n]]])
        yd_dot = np.array([[self.traj_x_dot[n]],
                           [self.traj_y_dot[n]],
                           [self.traj_z_dot[n]],
                           [self.traj_psi_dot[n]]])
        yd_ddot = np.array([[self.traj_x_ddot[n]],
                            [self.traj_y_ddot[n]],
                            [self.traj_z_ddot[n]],
                            [self.traj_psi_ddot[n]]])
        
        e1 = zeta1 - yd
        e2 = zeta2 - yd_dot
        
        k1 = np.array([1, 1, 10, 1]).reshape((4,1))
        k2 = np.array([1, 1, 10, 1]).reshape((4,1))

        q = -k1 * e1.reshape((4,1)) - k2 * e2.reshape((4,1)) + yd_ddot
        
        k_x = Binv @ (q - A)

        ## Control barrier function
        
        dZ = z - self.z0

        h = 1/2 * (-dZ**4 + (self.safe_z_bounds/2)**4)
        psi1 = 2*((-dZ))**3*(np.cos(phi)*np.cos(theta)*v_z__b - np.sin(theta)*v_x__b + np.cos(theta)*np.sin(phi)*v_y__b) + self.gamma*h
        
        lfpsi1 = (2*self.gamma*((-dZ))**3 - 6*((-dZ))**2*(np.cos(phi)*np.cos(theta)*v_z__b - np.sin(theta)*v_x__b + np.cos(theta)*np.sin(phi)*v_y__b))*(np.cos(phi)*np.cos(theta)*v_z__b - np.sin(theta)*v_x__b + np.cos(theta)*np.sin(phi)*v_y__b) + 2*((-dZ))**3*(np.cos(phi)*np.cos(theta)*v_y__b - np.cos(theta)*np.sin(phi)*v_z__b)*(w_x__b + np.cos(phi)*np.tan(theta)*w_z__b + np.sin(phi)*np.tan(theta)*w_y__b) - 2*((-dZ))**3*(np.cos(phi)*w_y__b - np.sin(phi)*w_z__b)*(np.cos(theta)*v_x__b + np.cos(phi)*np.sin(theta)*v_z__b + np.sin(phi)*np.sin(theta)*v_y__b) - 2*np.sin(theta)*((-dZ))**3*(w_z__b*((I_y*(m_y*v_y__b - m_RB*r_z_gb__b*w_x__b))/(I_y*m_x - m_RB**2*r_z_gb__b**2) + (m_RB*r_z_gb__b*(I_x*w_x__b - m_RB*r_z_gb__b*v_y__b))/(I_y*m_x - m_RB**2*r_z_gb__b**2)) - v_x__b*((D_vxy__CB*I_y)/(I_y*m_x - m_RB**2*r_z_gb__b**2) + (m_RB*m_z*r_z_gb__b*v_z__b)/(I_y*m_x - m_RB**2*r_z_gb__b**2)) + w_y__b*((D_omega_xy__CB*m_RB*r_z_gb__b)/(I_y*m_x - m_RB**2*r_z_gb__b**2) - (I_y*m_z*v_z__b)/(I_y*m_x - m_RB**2*r_z_gb__b**2)) + (m_RB*r_z_gb__b*v_z__b*(m_x*v_x__b + m_RB*r_z_gb__b*w_y__b))/(I_y*m_x - m_RB**2*r_z_gb__b**2) - (I_z*m_RB*r_z_gb__b*w_x__b*w_z__b)/(I_y*m_x - m_RB**2*r_z_gb__b**2) + (f_g*m_RB*r_z_gb__b**2*np.sin(theta))/((I_y*m_x - m_RB**2*r_z_gb__b**2)*(np.cos(theta)**2 + np.sin(theta)**2))) - 2*np.cos(theta)*np.sin(phi)*((-dZ))**3*(w_z__b*((I_x*(m_x*v_x__b + m_RB*r_z_gb__b*w_y__b))/(I_x*m_y - m_RB**2*r_z_gb__b**2) - (m_RB*r_z_gb__b*(I_y*w_y__b + m_RB*r_z_gb__b*v_x__b))/(I_x*m_y - m_RB**2*r_z_gb__b**2)) + v_y__b*((D_vxy__CB*I_x)/(I_x*m_y - m_RB**2*r_z_gb__b**2) + (m_RB*m_z*r_z_gb__b*v_z__b)/(I_x*m_y - m_RB**2*r_z_gb__b**2)) + w_x__b*((D_omega_xy__CB*m_RB*r_z_gb__b)/(I_x*m_y - m_RB**2*r_z_gb__b**2) - (I_x*m_z*v_z__b)/(I_x*m_y - m_RB**2*r_z_gb__b**2)) - (m_RB*r_z_gb__b*v_z__b*(m_y*v_y__b - m_RB*r_z_gb__b*w_x__b))/(I_x*m_y - m_RB**2*r_z_gb__b**2) + (I_z*m_RB*r_z_gb__b*w_y__b*w_z__b)/(I_x*m_y - m_RB**2*r_z_gb__b**2) + (f_g*m_RB*r_z_gb__b**2*np.cos(theta)*np.sin(phi))/((I_x*m_y - m_RB**2*r_z_gb__b**2)*(np.cos(phi)**2*np.cos(theta)**2 + np.cos(phi)**2*np.sin(theta)**2 + np.cos(theta)**2*np.sin(phi)**2 + np.sin(phi)**2*np.sin(theta)**2))) - 2*np.cos(phi)*np.cos(theta)*((-dZ))**3*((w_x__b*(m_y*v_y__b - m_RB*r_z_gb__b*w_x__b))/m_z - (w_y__b*(m_x*v_x__b + m_RB*r_z_gb__b*w_y__b))/m_z + (D_vz__CB*v_z__b)/m_z)
        lgpsi1 = np.array(
            [-2*np.sin(theta)*((-dZ))**3*(I_y/(I_y*m_x - m_RB**2*r_z_gb__b**2) - (m_RB*r_z_gb__b*r_z_tg__b)/(I_y*m_x - m_RB**2*r_z_gb__b**2)), 2*np.cos(theta)*np.sin(phi)*((-dZ))**3*(I_x/(I_x*m_y - m_RB**2*r_z_gb__b**2) - (m_RB*r_z_gb__b*r_z_tg__b)/(I_x*m_y - m_RB**2*r_z_gb__b**2)), (2*np.cos(phi)*np.cos(theta)*((-dZ))**3)/m_z, 0]
        ).reshape((1,4))

        self.m.remove(self.cbf_constraint)
        self.cbf_constraint = self.m.addConstr(lfpsi1 + lgpsi1 @ self.mu >= -self.gamma*psi1, "cbf")

        obj = (self.mu.T - k_x.T) @ (self.mu - k_x)
        self.m.setObjective(obj)

        self.m.optimize()

        if self.m.Status == 4:
            self.m.computeIIS()

            print("\nModel is infeasible")

            # Print out the IIS constraints and variables
            print('The following constraints and variables are in the IIS:')

            print("Constraints:")
            for c in self.m.getConstrs():
                if c.IISConstr: print(f'\t{c.constrname}: {self.m.getRow(c)} {c.Sense} {c.RHS}')

            print("Variables:")
            for v in self.m.getVars():
                if v.IISLB: print(f'\t{v.varname} >= {v.LB}')
                if v.IISUB: print(f'\t{v.varname} <= {v.UB}')
                sys.exit(1)
            print()

        u = self.mu.X
        
        psi1_dot = lfpsi1 + lgpsi1 @ u.reshape((4,1))

        sim.end_timer()

        return u.reshape((4,1))
        
    def get_trajectory(self):
        return (self.traj_x,
                self.traj_y,
                self.traj_z)
    
    def get_error(self, sim):
        n = sim.get_current_timestep() + 1
        return np.array([
            sim.get_var_history('x') - self.traj_x[0:n],
            sim.get_var_history('y') - self.traj_y[0:n],
            sim.get_var_history('z') - self.traj_z[0:n],
            sim.get_var_history('psi') - self.traj_psi[0:n]
        ]).T
