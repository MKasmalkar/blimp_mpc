import gurobipy as gp
from gurobipy import GRB
import numpy as np
import time
from BlimpController import BlimpController
from parameters import *
import sys

class MPCNonlinearHelix(BlimpController):

    def __init__(self, dT):
        super().__init__(dT)

        self.order = 12
        self.num_inputs = 4
        self.num_outputs = 6
        
        # Time
        TRACKING_TIME = 20
        SETTLE_TIME = 100

        tracking_time = np.arange(0, TRACKING_TIME, dT)
        settle_time = np.arange(TRACKING_TIME, TRACKING_TIME + SETTLE_TIME + 1, dT)

        time_vec = np.concatenate((tracking_time, settle_time))

        # Trajectory definition
        f = 0.05
        self.At = 1

        self.x0 = self.At
        y0 = 0
        z0 = 0

        phi0 = 0
        theta0 = 0
        self.psi0 = np.pi/2

        v_x0 = 0.0
        v_y0 = 0.0
        v_z0 = 0.0

        w_x0 = 0
        w_y0 = 0
        w_z0 = 0

        z_slope = -1/10

        self.traj_x = np.concatenate((self.At * np.cos(2*np.pi*f*tracking_time), self.At*np.ones(len(settle_time))))
        self.traj_y = np.concatenate((self.At * np.sin(2*np.pi*f*tracking_time), np.zeros(len(settle_time))))
        self.traj_z = np.concatenate((tracking_time * z_slope, TRACKING_TIME * z_slope * np.ones(len(settle_time))))
        self.traj_psi = np.concatenate((self.psi0 + 2*np.pi*f*tracking_time, (self.psi0 + 2*np.pi) * np.ones(len(settle_time))))
        
        self.target_phi = np.zeros(self.traj_x.shape)
        self.target_theta = np.zeros(self.traj_x.shape)
    
    def init_sim(self, sim):
        # Get A matrix corresponding to zero state vector equilibrium position
        A_dis = sim.get_A_dis()
        
        sim.set_var('x', self.x0)
        sim.set_var('psi', self.psi0)

        self.B = sim.get_B_dis()

        self.C = np.matrix([[0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]])
        self.D = np.zeros((self.num_outputs, self.num_inputs))

        self.P = np.identity(self.num_outputs)
        self.Q = np.identity(self.num_outputs)
        self.R = np.identity(self.num_inputs)

        xmin = np.matrix([[-np.inf],
                        [-np.inf],
                        [-np.inf],
                        [-np.inf],
                        [-np.inf],
                        [-np.inf],
                        [-np.inf],   # x
                        [-np.inf],   # y
                        [-np.inf],   # z
                        [-np.inf],
                        [-np.inf],
                        [-np.inf]
                        ])

        xmax = np.matrix([[np.inf],
                        [np.inf],
                        [np.inf],
                        [np.inf],
                        [np.inf],
                        [np.inf],
                        [np.inf],   # x
                        [np.inf],   # y
                        [np.inf],   # z
                        [np.inf],
                        [np.inf],
                        [np.inf]
                        ])

        umin = np.matrix([[-np.inf],
                        [-np.inf],
                        [-np.inf],
                        [-np.inf]])

        umax = np.matrix([[np.inf],
                        [np.inf],
                        [np.inf],
                        [np.inf]])
        
        self.N = 10

        self.env = gp.Env(empty=True)
        self.env.setParam('OutputFlag', 1)
        self.env.setParam('LogToConsole', 1)
        self.env.start()

        self.m = gp.Model(env=self.env)
        self.m.setParam("NonConvex", 2)

        self.x = self.m.addMVar(shape=(self.N+1, self.order),
                        lb=xmin.T, ub=xmax.T, name='x')
        self.y = self.m.addMVar(shape=(self.N+1, self.num_outputs),
                        lb=-GRB.INFINITY, ub=GRB.INFINITY, name='y')
        self.z = self.m.addMVar(shape=(self.N+1, self.num_outputs),
                        lb=-GRB.INFINITY, ub=GRB.INFINITY, name='z')
        self.u = self.m.addMVar(shape=(self.N, self.num_inputs),
                        lb=umin.T, ub=umax.T, name='u')
        
        self.ic_constraint = self.m.addConstr(self.x[0, :] == np.zeros((1, 12)).flatten(), name='ic')

        for k in range(self.N):
            self.m.addConstr(self.y[k, :] == self.C @ self.x[k, :])

            # Get state variables

            v_x_k = self.x[k, 0].item()
            v_y_k = self.x[k, 1].item()
            v_z_k = self.x[k, 2].item()
            w_x_k = self.x[k, 3].item()
            w_y_k = self.x[k, 4].item()
            w_z_k = self.x[k, 5].item()

            phi_k = self.x[k, 9].item()
            the_k = self.x[k, 10].item()
            
            nu_bn_b = np.array([[v_x_k],
                                [v_y_k],
                                [v_z_k],
                                [w_x_k],
                                [w_y_k],
                                [w_z_k]])
            
            # Define trig constraints
            cos_phi = self.m.addVar(vtype=GRB.CONTINUOUS, lb=-1.0, ub=1.0)
            sin_phi = self.m.addVar(vtype=GRB.CONTINUOUS, lb=-1.0, ub=1.0)
            cos_the = self.m.addVar(vtype=GRB.CONTINUOUS, lb=-1.0, ub=1.0)
            sin_the = self.m.addVar(vtype=GRB.CONTINUOUS, lb=-1.0, ub=1.0)
            tan_the = self.m.addVar(vtype=GRB.CONTINUOUS, lb=-GRB.INFINITY, ub=GRB.INFINITY)
            one_over_cos_the = self.m.addVar(vtype=GRB.CONTINUOUS, lb=-GRB.INFINITY, ub=GRB.INFINITY)

            self.m.addGenConstrCos(phi_k, cos_phi)
            self.m.addGenConstrSin(phi_k, sin_phi)
            self.m.addGenConstrCos(the_k, cos_the)
            self.m.addGenConstrSin(the_k, sin_the)
            self.m.addGenConstrTan(the_k, tan_the)
            self.m.addConstr(one_over_cos_the * cos_the == 1)

            # Define intermediate variables because Gurobi can only handle
            # bilinear constraints
            cos_phi_cos_the = self.m.addVar(vtype=GRB.CONTINUOUS, lb=-1.0, ub=1.0)
            self.m.addConstr(cos_phi_cos_the == cos_phi * cos_the)

            cos_phi_sin_the = self.m.addVar(vtype=GRB.CONTINUOUS, lb=-1.0, ub=1.0)
            self.m.addConstr(cos_phi_sin_the == cos_phi * sin_the)

            sin_phi_sin_the = self.m.addVar(vtype=GRB.CONTINUOUS, lb=-1.0, ub=1.0)
            self.m.addConstr(sin_phi_sin_the == sin_phi * sin_the)

            cos_the_sin_phi = self.m.addVar(vtype=GRB.CONTINUOUS, lb=-1.0, ub=1.0)
            self.m.addConstr(cos_the_sin_phi == cos_the * sin_phi)
            
            cos_phi_tan_the = self.m.addVar(vtype=GRB.CONTINUOUS, lb=-GRB.INFINITY, ub=GRB.INFINITY)
            self.m.addConstr(cos_phi_tan_the == cos_phi * tan_the)

            sin_phi_tan_the = self.m.addVar(vtype=GRB.CONTINUOUS, lb=-GRB.INFINITY, ub=GRB.INFINITY)
            self.m.addConstr(sin_phi_tan_the == sin_phi * tan_the)

            cos_phi_over_cos_the = self.m.addVar(vtype=GRB.CONTINUOUS, lb=-GRB.INFINITY, ub=GRB.INFINITY)
            self.m.addConstr(cos_phi_over_cos_the == cos_phi * one_over_cos_the)

            sin_phi_over_cos_the = self.m.addVar(vtype=GRB.CONTINUOUS, lb=-GRB.INFINITY, ub=GRB.INFINITY)
            self.m.addConstr(sin_phi_over_cos_the == sin_phi * one_over_cos_the)

            # Implement dynamics constraints

            fg_B = R_b__n_inv_grb(cos_phi, sin_phi, cos_the, sin_the, 1, 0) @ fg_n
            g_CB = -np.block([[np.zeros((3, 1))],
                        [np.reshape(np.cross(r_gb__b, fg_B), (3, 1))]])
            
            tau_B = np.array([[self.u[k, 0]],
                              [self.u[k, 1]],
                              [self.u[k, 2]],
                              [-r_z_tg__b*self.u[k, 1]],
                              [r_z_tg__b*self.u[k, 1]],
                              [self.u[k, 3]]])
            
            x_dot = self.m.addVar(vtype=GRB.CONTINUOUS, lb=-GRB.INFINITY, ub=GRB.INFINITY)
            self.m.addConstr(x_dot == v_x_k*cos_the + v_z_k*cos_phi_sin_the + v_y_k*sin_phi_sin_the)
            
            y_dot = self.m.addVar(vtype=GRB.CONTINUOUS, lb=-GRB.INFINITY, ub=GRB.INFINITY)
            self.m.addConstr(y_dot == v_y_k*cos_phi - v_z_k*sin_phi)
            
            z_dot = self.m.addVar(vtype=GRB.CONTINUOUS, lb=-GRB.INFINITY, ub=GRB.INFINITY)
            self.m.addConstr(z_dot == v_z_k*cos_phi_cos_the - v_x_k*sin_the + v_y_k*cos_the_sin_phi)
            
            phi_dot = self.m.addVar(vtype=GRB.CONTINUOUS, lb=-GRB.INFINITY, ub=GRB.INFINITY)
            self.m.addConstr(phi_dot == w_x_k + w_z_k*cos_phi_tan_the + w_y_k*sin_phi_tan_the)
            
            the_dot = self.m.addVar(vtype=GRB.CONTINUOUS, lb=-GRB.INFINITY, ub=GRB.INFINITY)
            self.m.addConstr(the_dot == w_y_k*cos_phi - w_z_k*sin_phi)
            
            psi_dot = self.m.addVar(vtype=GRB.CONTINUOUS, lb=-GRB.INFINITY, ub=GRB.INFINITY)
            self.m.addConstr(psi_dot == w_z_k*cos_phi_over_cos_the + w_y_k*sin_phi_over_cos_the)
            
            nu_bn_b_dot = np.reshape(-M_CB_inv @ (C(M_CB, nu_bn_b) @ nu_bn_b + \
                                D_CB @ nu_bn_b + g_CB - tau_B), (6, 1))
            
            v_x_dot = self.m.addVar(vtype=GRB.CONTINUOUS, lb=-GRB.INFINITY, ub=GRB.INFINITY)
            self.m.addConstr(v_x_dot == nu_bn_b_dot[0].item())

            v_y_dot = self.m.addVar(vtype=GRB.CONTINUOUS, lb=-GRB.INFINITY, ub=GRB.INFINITY)
            self.m.addConstr(v_y_dot == nu_bn_b_dot[1].item())

            v_z_dot = self.m.addVar(vtype=GRB.CONTINUOUS, lb=-GRB.INFINITY, ub=GRB.INFINITY)
            self.m.addConstr(v_z_dot == nu_bn_b_dot[2].item())
            
            w_x_dot = self.m.addVar(vtype=GRB.CONTINUOUS, lb=-GRB.INFINITY, ub=GRB.INFINITY)
            self.m.addConstr(w_x_dot == nu_bn_b_dot[3].item())

            w_y_dot = self.m.addVar(vtype=GRB.CONTINUOUS, lb=-GRB.INFINITY, ub=GRB.INFINITY)
            self.m.addConstr(w_y_dot == nu_bn_b_dot[4].item())

            w_z_dot = self.m.addVar(vtype=GRB.CONTINUOUS, lb=-GRB.INFINITY, ub=GRB.INFINITY)
            self.m.addConstr(w_z_dot == nu_bn_b_dot[5].item())

            self.m.addConstr(self.x[k+1, 0]  == self.x[k, 0]  + sim.dT * v_x_dot)
            self.m.addConstr(self.x[k+1, 1]  == self.x[k, 1]  + sim.dT * v_y_dot)
            self.m.addConstr(self.x[k+1, 2]  == self.x[k, 2]  + sim.dT * v_z_dot)
            self.m.addConstr(self.x[k+1, 3]  == self.x[k, 3]  + sim.dT * w_x_dot)
            self.m.addConstr(self.x[k+1, 4]  == self.x[k, 4]  + sim.dT * w_y_dot)
            self.m.addConstr(self.x[k+1, 5]  == self.x[k, 5]  + sim.dT * w_z_dot)
            self.m.addConstr(self.x[k+1, 6]  == self.x[k, 6]  + sim.dT * x_dot)
            self.m.addConstr(self.x[k+1, 7]  == self.x[k, 7]  + sim.dT * y_dot)
            self.m.addConstr(self.x[k+1, 8]  == self.x[k, 8]  + sim.dT * z_dot)
            self.m.addConstr(self.x[k+1, 9]  == self.x[k, 9]  + sim.dT * phi_dot)
            self.m.addConstr(self.x[k+1, 10] == self.x[k, 10] + sim.dT * the_dot)
            self.m.addConstr(self.x[k+1, 11] == self.x[k, 11] + sim.dT * psi_dot)
            
        self.error_constraints = []
        for k in range(self.N):
            self.error_constraints.append(
                # z = error
                # z = y - reference
                # => y - z = reference
                self.m.addConstr(self.y[k, :] - self.z[k, :] == np.zeros((1, self.num_outputs)).flatten(),
                                 name='error' + str(k)))

        # terminal cost
        obj1 = self.z[self.N, :] @ self.P @ self.z[self.N, :]
        
        # running state/error cost
        obj2 = sum(self.z[k, :] @ self.Q @ self.z[k, :] for k in range(self.N))
        
        # running input cost
        obj3 = sum(self.u[k, :] @ self.R @ self.u[k, :] for k in range(self.N))

        self.m.setObjective(obj1 + obj2 + obj3)

        self.m.update()

        # Add attitude oscillation damping costs
        max_allowable_phi = 0.05
        max_allowable_theta = 0.05
        max_allowable_psi_deviation = 0.05

        max_allowable_x_deviation = 0.4
        max_allowable_y_deviation = 0.4
        max_allowable_z_deviation = 0.4

        self.Q[0, 0] = 1/max_allowable_x_deviation**2
        self.Q[1, 1] = 1/max_allowable_y_deviation**2
        self.Q[2, 2] = 1/max_allowable_z_deviation**2

        self.P[0, 0] = 1/max_allowable_x_deviation**2
        self.P[1, 1] = 1/max_allowable_y_deviation**2
        self.P[2, 2] = 1/max_allowable_z_deviation**2

        self.Q[3, 3] = 1/max_allowable_phi**2
        self.Q[4, 4] = 1/max_allowable_theta**2
        self.Q[5, 5] = 1/max_allowable_psi_deviation**2

        self.P[3, 3] = 1/max_allowable_phi**2
        self.P[4, 4] = 1/max_allowable_theta**2
        self.P[5, 5] = 1/max_allowable_psi_deviation**2

        # terminal cost
        obj1 = self.z[self.N, :] @ self.P @ self.z[self.N, :]
        
        # running state/error cost
        obj2 = sum(self.z[k, :] @ self.Q @ self.z[k, :] for k in range(self.N))
        
        # running input cost
        obj3 = sum(self.u[k, :] @ self.R @ self.u[k, :] for k in range(self.N))

        self.att_damp_obj = obj1 + obj2 + obj3

    def get_ctrl_action(self, sim):

        sim.start_timer()

        n = sim.get_current_timestep()

        if n == int(20/self.dT):
            self.m.setObjective(self.att_damp_obj)

        reference = np.array([
            self.traj_x[n:min(n+self.N, len(self.traj_x))],
            self.traj_y[n:min(n+self.N, len(self.traj_y))],
            self.traj_z[n:min(n+self.N, len(self.traj_z))],
            np.zeros(min(self.N, len(self.traj_x) - n)),
            np.zeros(min(self.N, len(self.traj_x) - n)),
            self.traj_psi[n:min(n+self.N, len(self.traj_psi))]
        ])

        sim_state = sim.get_state()

        # A matrix is generated assuming psi = 0
        # need to perform some rotations to account for this

        psi = sim_state[11].item()

        v_b = np.array([
            [sim_state[0].item()],
            [sim_state[1].item()],
            [sim_state[2].item()]
        ])

        v_n = R_b__n(0, 0, psi) @ v_b

        state = np.array([
            v_n[0].item(),
            v_n[1].item(),
            v_n[2].item(),
            sim_state[3].item(),
            sim_state[4].item(),
            sim_state[5].item(),
            sim_state[6].item(),
            sim_state[7].item(),
            sim_state[8].item(),
            sim_state[9].item(),
            sim_state[10].item(),
            psi
        ])

        self.ic_constraint.rhs = state

        for k in range(self.N):
            if k < reference.shape[1]:
                self.error_constraints[k].rhs = reference[:, k]
            else:
                self.m.remove(self.error_constraints[k])
            
        self.m.optimize()

        u_orig = self.u.X[0].T

        u_rot = R_b__n_inv(0, 0, psi) @ np.array([u_orig[0], u_orig[1], u_orig[2]]).T

        u = np.array([
            [u_rot[0].item()],
            [u_rot[1].item()],
            [u_rot[2].item()],
            [u_orig[3].item()]
        ])

        #print(self.m.status)
        #print(np.round(self.u.X[0].T, 3))
        
        sim.end_timer()

        print(u.reshape(4))
        print(self.m.objVal)
        return u
        
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
