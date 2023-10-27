# Luke Baird
# (c) Georgia Institute of Technology, 2023
# Uses math from Tao et. al.: Swing-Reducing Flight Control System for an Underactuated Indoor
# Miniature Autonomous Blimp.
#
# Changelog
# 07/31 - Set feedback gain K to zero.

from rta.model import Model
import numpy as np
from scipy.linalg import expm
from scipy.signal import place_poles
from matplotlib import pyplot as plt

class InvalidR3VectorException(Exception):
    def __init__(self, *args: object) -> None:
        super().__init__(*args)
        self.msg = "Invalid vector!"

class Blimp(Model):
    def __init__(self):
        super()
        self.linear = False
        self.control_affine = True
        self.shape_((12,12)) # six degrees of freedom.

        self.load_constants()
        self.jacobian_(self.jacobian)
        self.setup_B()
        self.discretized = False

    def discretize(self, dT):
        # Change the Jacobian and B matrix appropriately.
        self.dT = dT
        self.jacobian_(lambda x: expm(dT * self.jacobian(x)))
        self.B_(self.B * dT)
        self.discretized = True
    
    # Precondition: self.discretize(dT) has NOT been called.
    def setup_discrete_lti_model(self, dT):
        A = self.jacobian_np(np.zeros((12,1)))
        w,v = np.linalg.eig(A)
        p = w
        for j in range(w.size):
            if np.imag(p[j]) > 1:
                p[j] -= 1 + 2.5j
            elif np.imag(p[j]) < -1:
                p[j] += -1 + 2.5j
        K = place_poles(A, self.B, p)

        # A = A - self.B@ np.round(K.gain_matrix, 4)

        self.A_discrete = scipy.linalg.expm(A * dT)
        B_int = np.zeros((12,12))
        for i in range(10000):
            dTau = dT / 10000
            tau = i * dTau
            B_int += scipy.linalg.expm(A * tau) * dTau
        self.B_discrete = B_int @ self.B
    
    def get_discrete_model(self):
        return (self.A_discrete, self.B_discrete)

    def setup_B(self):
        # This sets up the input matrix.
        to_4 = np.array([[1, 0, 0, 0, self.r_b_z_tb, 0],
                         [0, 1, 0, -self.r_b_z_tb, 0, 0],
                         [0, 0, 1, 0, 0, 0],
                         [0, 0, 0, 0, 0, 1]]).T
        zeros_padding = np.zeros((6,4))
        M_inv_blocked = np.block([[self.M_inv, np.zeros((6,6))],
                                  [np.zeros((6, 12))]])
        self.B_(M_inv_blocked @ np.block([[to_4],[zeros_padding]])) # -self.M_inv @ 
        # self.B_(to_4)
        # print(self.B)
        return
        self.B_(self.mass_matrix@to_4)

    def f(self, x, u):
        # Compute \dot{x}.
        if self.discretized:
            return self.jacobian_np(x) @ x + self.B @ u / self.dT
        else:
            return self.jacobian_np(x) @ x + self.B @ u

    # Function: jacobian_np(x)
    # Purpose: compute the jacobian as a ndarray(12,12) about a ndarray(12,1) point x.
    def jacobian_np(self, x):
        # x is the state in R12.
        # v, w, p, theta
        v = x[0:3].squeeze(1)
        w = x[3:6].squeeze(1)
        p = x[6:9].squeeze(1)
        theta = x[9:12].squeeze(1)
        
        # inertial_gradient_matrix = np.diag([self.I[2]-self.I[1], self.I[0]-self.I[2], self.I[1]-self.I[0]])

        # grad_Iw is in R 3x3
        grad_v_M = self.mass_matrix @ self.skew(np.array([w]).T)
        grad_w_M = -self.mass_matrix @ self.skew(np.array([v]).T)
        grad_v_I = np.zeros((3,3))
        grad_w_I = np.zeros((3,3))# inertial_gradient_matrix @ np.abs(self.skew(np.array([w]).T))

        temp_grad = self.HDH + np.block([[grad_v_M, grad_w_M],
                                         [grad_v_I, grad_w_I]]);

        free_jacobian = -self.M_inv @ temp_grad

        angle_jacobian = np.array([[-np.cos(theta[0])*np.cos(theta[1]), \
                                    np.sin(theta[0])*np.sin(theta[1]), 0],
                                   [0, -np.cos(theta[1]), 0],
                                   [0, 0, 0]]) * self.r_b_z_gb * self.m_RB * self.g
        
        # Integrators portion
        integrator_jacobian = self.M_inv @ np.block([[np.zeros((3,6))],
                                [np.zeros((3,3)), angle_jacobian]])

        return np.block([[free_jacobian, integrator_jacobian],
                         [np.eye(6), np.zeros((6,6))]])
    
    # Function: jacobian
    # Purpose: compute the jacobian as an ndarray(12,12) about a ndarray(12,) point x, including a small angle approximation.
    def jacobian(self, x):
        # x is the state in R12.
        # v, w, p, theta
        v = x[0:3]
        w = x[3:6]
        p = x[6:9]
        theta = x[9:12]
        
        # inertial_gradient_matrix = np.diag([self.I[2]-self.I[1], self.I[0]-self.I[2], self.I[1]-self.I[0]])

        # grad_Iw is in R 3x3
        grad_v_M = self.mass_matrix @ self.skew2(w)
        grad_w_M = -self.mass_matrix @ self.skew2(v)
        grad_v_I = np.zeros((3,3))
        grad_w_I = np.zeros((3,3))# inertial_gradient_matrix @ np.abs(self.skew(np.array([w]).T))

        temp_grad = self.HDH + np.block([[grad_v_M, grad_w_M],
                                         [grad_v_I, grad_w_I]]);

        free_jacobian = -self.M_inv @ temp_grad

        # angle_jacobian = np.array([[-np.cos(theta[0])*np.cos(theta[1]), \
        #                             np.sin(theta[0])*np.sin(theta[1]), 0],
        #                            [0, -np.cos(theta[1]), 0],
        #                            [0, 0, 0]]) * self.r_b_z_gb * self.m_RB * self.g
        angle_jacobian = np.array([[-1, 0, 0],
                                   [0, -1, 0],
                                   [0, 0, 0]]) * self.r_b_z_gb * self.m_RB * self.g
        
        # Integrators portion
        integrator_jacobian = self.M_inv @ np.block([[np.zeros((3,6))],
                                [np.zeros((3,3)), angle_jacobian]])

        return np.block([[free_jacobian, integrator_jacobian],
                         [np.eye(6), np.zeros((6,6))]])
    
    # Function: compute_stable_physics_model_feedback(poles)
    # Purpose: compute the K matrix for a stabilized linearized physics model of the blimp
    # Precondition: self.discretize(dT) has been externally called.
    def compute_stable_physics_model_feedback(self, poles=None):
        # Setup a vector representing the desired pole locations.
        J = self.jacobian_np(np.zeros((12,1))) # Linearize about the origin in R12.
        A = np.eye(12) + self.dT * J # Get the discretized system dynamics
        w, v = np.linalg.eig(A) # w = array of eigenvalues
        for i in range(w.size):
            if np.abs(w[i]) > 1:
                # scale it down to a magnitude of 0.99.
                w[i] /= (1.01 * np.abs(w[i]))
        if poles is None:
            p_discrete = w # np.array([1, 1, 0.6542 + 0.7562j, 0.6542 - 0.7562j, 0.9818, 0.6542 + 0.7562j, 0.6542 - 0.7562j, 0.9818, 1, 0.9331, 1, 0.9579])
        else:
            p_discrete = poles # permit user to override computations completed in this routine.
        self.K = place_poles(A, self.B, p_discrete)

    # Function: stable_physics_model
    # Purpose: create a simple linearized stable LTI model of the blimp about nu===0.
    # Precondition: self.dT is externally set.
    #               self.K is set b compute_stable_physics_model_feedback
    def stable_physics_model(self):
        A = scipy.linalg.expm(self.dT * self.jacobian_np(np.zeros((12,1))))
        B = self.B # already discretized and converted to R12x4.
        # K = self.K.gain_matrix
        self.K = np.array([[0.0222, 0.0013, 0.0003, 0.0003, 0.1281, -0.0001, 0.0000, -0.0000, -0.0000, 0.0018, -0.4361, 0.0000, ],
                      [0.0013, -0.0044, 0.0009, 0.1264, 0.0003, -0.0000, 0.0000, 0.0000, 0.0000, -0.4449, 0.0018, 0.0000, ],
                      [0.0007, 0.0019, -0.0180, -0.0023, -0.0004, -0.0000, -0.0000, -0.0000, 0.0000, 0.0007, 0.0023, 0.0000, ],
                      [-0.0000, -0.0000, -0.0000, 0.0000, 0.0000, -0.0006, 0.0000, 0.0000, -0.0000, -0.0000, -0.0000, -0.0000, ]]) # dT = 0.25
        self.K = np.zeros((4,12))
        # print(self.K.gain_matrix)
        return (A - B@self.K, B) # K is computed from pole_placement.py. The unstable poles are moved into the unit circle.
    
    # Function: skew2(v)
    # Purpose: Take a ndarray(3,) and turn it into a skew ndarray(3,3)
    def skew2(self, v): # valid for vectors in R3.
        if v.size == 3:
            return np.array([[0, -v[2], v[1]],
                            [v[2], 0, -v[0]],
                            [-v[1], v[0], 0]])
        else:
            raise InvalidR3VectorException

    # Function: skew(v)
    # Purpose: Take a ndarray(3,1) and turn it into a skew ndarray(3,3)
    # Note: this function works as well for Gurobi variables.
    def skew(self, v): # valid for vectors in R3.
        if v.size == 3:
            return np.array([[0, -v[2,0], v[1,0]],
                            [v[2,0], 0, -v[0,0]],
                            [-v[1,0], v[0,0], 0]])
        else:
            raise InvalidR3VectorException
    
    # Function: H(v)
    # Purpose: compute an H-transform of a vector (see Fossen, 2012)
    def H(self, v):
        return np.block([[np.eye(3), self.skew(v).T],
                         [np.zeros((3,3)), np.eye(3)]])

    # Function: gravity(latitude)
    # Purpose: WSG2 model of gravity as a ellipsoid.
    # Computes the local gravity down vector as a function of latitude.
    def gravity(self, latitude):
        # Calculate the gravity of Atlanta
        g_e = 9.780327
        beta_1 = 5.30244e-3
        beta_2 = -5.8e-6
        return g_e * (1 + beta_1 * np.sin(latitude)**2 + beta_2 * np.sin(2*latitude)**2)

    # Function: load_constants()
    # Purpose: initializes blimp parameters as class variables
    def load_constants(self):
        m_RB = 0.1249 # kg
        m_Ax = 0.0466 # kg
        m_Az = 0.0545 # kg
        r_b_z_gb = 0.09705 # m
        self.r_b_z_tb = 0.14 - r_b_z_gb # m, TODO MEASURE: 11in = 0.28m.
        # self.r_b_z_tb = 0.3048 - r_b_z_gb # m, perhaps this is way too small above?
        D_CB_vx = 0.0125 # unitless
        D_CB_vz = 0.0480 # unitless
        D_CB_wy = 0.000980 # N * m * s / rad
        I_CG_y = 0.005821 # kg * m^2

        # Updated constants.
        m_RB = 0.1049 # kg # 2
        m_Ax = 0.3566 # kg
        m_Az = 0.645 # kg
        r_b_z_gb = 0.08705 # m
        self.r_b_z_tb = 0.13 - r_b_z_gb # m, TODO MEASURE.
        D_CB_vx = 0.0115 # unitless
        D_CB_vz = 0.0480 # unitless
        D_CB_wy = 0.00980 # N * m * s / rad
        I_CG_y = 0.5821 # kg * m^2

        atl_latitude = 33.74888889 * np.pi / 180
        g = self.gravity(atl_latitude) # WSG2 model. Standard g=9.80665 m/s^2

        I_b__gA = np.zeros((3,3))
        I_b__gRB = np.diag([I_CG_y, I_CG_y, I_CG_y])
        m_A = np.diag([m_Ax, m_Ax, m_Az])
        r_b__bg = np.array([[0],
                            [0],
                            [r_b_z_gb]])
        D_CB = np.diag([D_CB_vx, D_CB_vx, D_CB_vz, D_CB_wy, D_CB_wy, D_CB_wy])

        M1 = np.block([[m_RB * np.eye(3), np.zeros((3,3))],
                       [np.zeros((3,3)), I_b__gRB]])
        M2 = np.block([[m_A, np.zeros((3,3))],
                       [np.zeros((3,3)), I_b__gA]])

        self.M_inv = np.linalg.inv(M1 + M2)
        self.HDH = self.H(-r_b__bg).T @ D_CB @ self.H(-r_b__bg)
        self.I = np.diag(I_b__gA + I_b__gRB)
        self.mass_matrix = m_RB * np.eye(3) + m_A

        # export whatever we need
        self.r_b_z_gb = r_b_z_gb
        self.m_RB = m_RB
        self.g = g

        # Here, build coefficients for the simplified model.
        self.alpha_1 = - (g * (m_RB + m_Ax) * m_RB * r_b_z_gb) / (- (m_Ax * r_b_z_gb)**2 + I_CG_y * (m_RB + m_Ax))
        self.alpha_2 = - (D_CB_wy * (m_RB + m_Ax) + D_CB_vx * m_RB * r_b_z_gb ** 2) / (- (m_Ax * r_b_z_gb)**2 + I_CG_y * (m_RB + m_Ax))
        self.alpha_3 = (m_Ax * r_b_z_gb + (m_RB + m_Ax) * self.r_b_z_tb) / (- (m_Ax * r_b_z_gb) ** 2 + I_CG_y * (m_RB + m_Ax))

        self.beta_1 = - (D_CB_vx * r_b_z_gb + D_CB_vx * self.r_b_z_tb) / (self.r_b_z_tb * (m_RB + m_Ax) + r_b_z_gb * m_Ax)
        self.beta_2 = (g * m_RB * r_b_z_gb) / (self.r_b_z_tb * (m_RB + m_Ax) + r_b_z_gb * m_Ax)

        # Out of laziness, pull the coefficient for the yaw dynamics from matrix multiplication
        self.gamma_1 = (self.M_inv @ self.HDH)[5,5]
        self.gamma_2 = 0

        self.delta_1 = -D_CB_vz / (m_RB + m_Az)
        self.delta_2 = 1 / (m_RB + m_Az)
        
        # export extra values for the swing reducing flight controller.
        self.theta_setpoint_coefficient = (D_CB_vx * self.r_b_z_tb) / (r_b_z_gb * m_RB * g) # exported.
        self.inner_loop_coefficient = (r_b_z_gb / self.r_b_z_tb) * m_RB * g # exported.

        #### MORE COEFFICIENTS FOR SIMPLIFIED MODEL
        v_x_demoniator = (-m_Ax * m_Ax * r_b_z_gb * r_b_z_gb + I_CG_y * (m_RB + m_Ax))
        self.v_x_0 = (D_CB_vx * m_Ax * r_b_z_gb * r_b_z_gb - D_CB_vx * I_CG_y) / v_x_demoniator
        self.v_x_1 = -(g * m_Ax * m_RB * r_b_z_gb * r_b_z_gb) / v_x_demoniator
        self.v_x_2 = (D_CB_vx * I_CG_y * r_b_z_gb - D_CB_vx * m_Ax * r_b_z_gb * r_b_z_gb * r_b_z_gb - D_CB_wy * m_Ax * r_b_z_gb) / v_x_demoniator
        self.v_x_3 = (I_CG_y * m_Ax * r_b_z_gb * self.r_b_z_tb) / v_x_demoniator
        # the same coefficients are used for v_y equations of motion.
        self.w_y_0 = (D_CB_vx * m_RB * r_b_z_gb) / v_x_demoniator
        self.w_y_1 = -(g * (m_RB + m_Ax) * m_RB * r_b_z_gb) / v_x_demoniator
        self.w_y_2 = -(D_CB_wy * (m_RB + m_Ax) + D_CB_vx * m_RB * r_b_z_gb * r_b_z_gb) / v_x_demoniator
        self.w_y_3 = (m_Ax * r_b_z_gb + (m_RB + m_Ax)) / v_x_demoniator
        # the same coefficients are used for w_x equations of motion.
        self.v_z_0 = -D_CB_vz / (m_RB + m_Az)
        self.v_z_1 = 1 / (m_RB + m_Az)
        self.psi_0 = -0.16835595
        self.psi_1 = 171.79178835
    
    # Function: Compute LTI
    # Purpose: returns a linearized model of the blimp about x0
    # The jacobian used is the primary `jacobian` function.
    def compute_LTI(self, x0):
        J = self.jacobian_hidden(x0)
        return (J, self.B)

    # Function: compute_LTV(x0, steps)
    # Purpose: computes a linear-time-varying model using an euler discretization of the blimp.
    # This function accepts three inputs:
    # x0 \in R12
    # u \in R4 x b
    # dT \in R+
    # The state is projected out based on x0 and u for b steps, using an euler discretization
    # of the dynamics.
    # `jacobian` is the dynamics model used.
    def compute_LTV(self, x0, steps):
        # This function does not work properly yet!
        A_list = [None] * steps
        for k in range(steps): # u is in R4xn
            A_list[k] = self.jacobian_hidden(np.zeros((12,1)))#x0[:, k:k+1]) # np.eye(12) + dT * 
            print(f'Norm of A[{k}] = {np.linalg.norm(A_list[k])}')
        return (A_list, self.B)# * dT)
    
    # Function: ddt_simplified_model
    # Purpose: develop the swing-reducing flight control simplified model naively, implementing code from the blimp simulator
    # Note that this model is unstable for self.dT > 0.001.
    def ddt_simplified_model(self, x, u): # This model needs to be re-written as an LTV model.
        self.setpoint_inputs(u)
        # Return a simplified state dynamics model.
        # Recall: x \in R12.
        v = x[0:3]
        w = x[3:6]
        p = x[6:9]
        theta = x[9:12]

        self.k_theta = 0.27
        self.k_omega = 0.25

        # Load in a reference theta.
        dTheta_x = theta[0] - self.theta_setpoint_x
        dTheta_y = theta[1] - self.theta_setpoint_y

        dV_x = v[0] - self.v_setpoint_x
        dV_y = v[1] - self.v_setpoint_y

        dw_x = np.array([[self.alpha_1 - self.k_theta, self.alpha_2 - self.k_omega * self.alpha_3]]) @ np.array([[dTheta_x], [w[0]]])
        dw_y = np.array([[self.alpha_1 - self.k_theta, self.alpha_2 - self.k_omega * self.alpha_3]]) @ np.array([[dTheta_y], [w[1]]])
        dv_x = self.beta_1 * dV_x + self.beta_2 * dTheta_y
        dv_y = self.beta_1 * dV_y + self.beta_2 * dTheta_x

        dp_x = w[0]
        dp_y = w[1]
        dp_z = w[2]

        dtheta_x = w[0]
        dtheta_y = w[1]
        dtheta_z = w[2]

        # for the z in velocity and acceleration, use as inputs the positions, and then we havel a PD controller.
        # Recall the vertical motion primitive from the paper.
        # I need to plug this in and do it on the whiteboard.
        k_psi = 0.5
        k_v = 0.5
        k_p = 0.27
        dw_z = self.gamma_1 * theta[2] - k_psi * theta[2] + self.gamma_2 * w[2] - k_psi * w[2] + k_psi * self.psi_setpoint
        dv_z = self.delta_1 * v[2] - k_v * self.delta_2 * v[2] - k_p * self.delta_2 * (p[2] - self.z_setpoint)

        return np.diag([dv_x, dv_y, dv_z, dw_x, dw_y, dw_z, dp_x, dp_y, dp_z, dtheta_x, dtheta_y, dtheta_z])
    
        # I need to factor out the input matrices, if possible, to form...
        # xdot = Ax + Bu.

        # What I really need to do is LaTeX this.
    
    # Function: simplified_model_with_feedback(x0, steps)
    # Purpose: develop the swing-reducing flight controller simplified model, but with feedback to stabilize the system
    # after discretization.
    def simplified_model_with_feedback(self):
        # This calculates the A-matrix, LTI assumption.
        A = np.zeros((12,12))
        B = np.zeros((12, 6))
        to_4 = np.array([[1, 0, 0, 0, self.r_b_z_tb, 0],
                         [0, 1, 0, self.r_b_z_tb, 0, 0],
                         [0, 0, 1, 0, 0, 0],
                         [0, 0, 0, 0, 0, 1]]).T
        
        # v_x
        A[0, 0] = self.v_x_0 # v_x
        A[0, 10] = self.v_x_1 # theta
        A[0, 4] = self.v_x_2 # w_y
        B[0, 0] = self.v_x_3 # f_x
        # v_y
        A[1, 1] = self.v_x_0 # v_y
        A[1, 9] = self.v_x_1 # phi
        A[1, 3] = self.v_x_2 # w_x
        B[1, 1] = self.v_x_3 # f_y

        # w_x
        A[3, 1] = self.w_y_0 # v_y
        A[3, 9] = self.w_y_1 # phi, roll
        A[3, 3] = self.w_y_2 # w_x
        B[3, 3] = self.w_y_3 # tau_x
        # w_y
        A[4, 0] = self.w_y_0 # v_x
        A[4, 10] = self.w_y_1 # theta, pitch
        A[4, 4] = self.w_y_2 # w_y
        B[4, 4] = self.w_y_3 # tau_y

        # v_z
        A[2, 2] = self.v_z_0
        B[2, 2] = self.v_z_1
        # w_z
        A[5, 5] = self.psi_0
        B[5, 5] = self.psi_1

        # and now the rest...
        for j in range(6,12):
            A[j,j - 6] = 1

#         K = np.array([[-0.9358, -0.0000, -0.0000, -0.0000, 0.3077, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.3790, 0.0000, ],
# [-0.0000, -0.9358, 0.0000, 0.3077, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.3790, 0.0000, 0.0000, ],
# [-0.0000, 0.0000, 0.0417, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, -0.0000, 0.0000, 0.0000, ],
# [-0.0000, -0.0000, -0.0000, 0.0000, -0.0000, 0.0019, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, ],]) # dT = 0.1
        K = np.array([[-0.0376, -0.0000, -0.0000, 0.0000, 0.1159, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, -0.7545, 0.0000, ],
[-0.0000, -0.0376, -0.0000, 0.1159, 0.0000, -0.0000, 0.0000, 0.0000, 0.0000, -0.7545, 0.0000, 0.0000, ],
[-0.0000, -0.0000, -0.0121, -0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, -0.0000, 0.0000, ],
[-0.0000, 0.0000, -0.0000, -0.0000, 0.0000, 0.0002, 0.0000, 0.0000, 0.0000, -0.0000, 0.0000, 0.0000, ],]) # dT = 0.25
        # Note: K should not be chosen such that the poles are fundamentally different.
        # This model assumes linearization above the equilibrium nu === 0.
        A_cl = (np.eye(12) + self.dT *A) - self.dT * (B@to_4) @ K # B @ to_4 @ K
                       
        return (A_cl, B @ to_4 * self.dT)
        # This still has poles very near to the jw axis.)
        # return A, B @ to_4
    
    # Function: simplified_model()
    # Purpose: create a model of the blimp based on the swing-reducing flight controller simplified model
    # The output is discretized for a globally specified dT
    # Precondition: self.dT is defined.
    def simplified_model(self): # This model needs to be re-written as an LTV model.
        # self.setpoint_inputs(u)
        # Return a simplified state dynamics model.
        # Recall: x \in R12.
        # v = x[0:3]
        # w = x[3:6]
        # p = x[6:9]
        # theta = x[9:12]

        self.k_theta = 0.27
        self.k_omega = 0.25

        # Load in a reference theta.
        # dTheta_x = theta[0] - self.theta_setpoint_x
        # dTheta_y = theta[1] - self.theta_setpoint_y

        # dV_x = v[0] - self.v_setpoint_x
        # dV_y = v[1] - self.v_setpoint_y
        A = np.zeros((12,12))
        B = np.zeros((12,6))

        # dw_x_A = (self.alpha_1 - self.k_theta) * theta[0] + (self.alpha_2 - self.k_omega * self.alpha_3) * w[0]
        # dw_x_B = (self.alpha_1 - self.k_theta) * -self.theta_setpoint_x
        A[3,9] = self.alpha_1 - self.k_theta
        A[3,4] = self.alpha_2 - self.k_omega * self.alpha_3
        B[3,2] = -(self.alpha_1 - self.k_theta)

        A[4,10] = self.alpha_1 - self.k_theta
        A[4,5] = self.alpha_2 - self.k_omega * self.alpha_3
        B[4,3] = -(self.alpha_1 - self.k_theta) 

        A[0,0] = self.beta_1
        A[0,9] = self.beta_2
        B[0,0] = -self.beta_1
        B[0,2] = -self.beta_2

        A[1,1] = self.beta_1
        A[1,10] = self.beta_2
        B[1,1] = -self.beta_1
        B[1,3] = -self.beta_2

        # dw_x = np.array([[self.alpha_1 - self.k_theta, self.alpha_2 - self.k_omega * self.alpha_3]]) @ np.array([[dTheta_x], [w[0]]])
        # dw_y = np.array([[self.alpha_1 - self.k_theta, self.alpha_2 - self.k_omega * self.alpha_3]]) @ np.array([[dTheta_y], [w[1]]])
        # dv_x = self.beta_1 * dV_x + self.beta_2 * dTheta_y
        # dv_y = self.beta_1 * dV_y + self.beta_2 * dTheta_x

        # dp_x = v[0]
        # dp_y = v[1]
        # dp_z = v[2]

        A[6,0] = 1
        A[7,1] = 1
        A[8,2] = 1

        # dtheta_x = w[0]
        # dtheta_y = w[1]
        # dtheta_z = w[2]

        A[9,3] = 1
        A[10,4] = 1
        A[11,5] = 1

        # for the z in velocity and acceleration, use as inputs the positions, and then we havel a PD controller.
        # Recall the vertical motion primitive from the paper.
        # I need to plug this in and do it on the whiteboard.
        k_psi = 0.5
        k_v = 0.5
        k_p = 0.27
        # dw_z = self.gamma_1 * theta[2] - k_psi * theta[2] + self.gamma_2 * w[2] - k_psi * w[2] + k_psi * self.psi_setpoint
        # dv_z = self.delta_2 * v[2] - k_v * v[2] - k_p * (p[2] - self.z_setpoint) + self.delta_1

        A[5,11] = self.gamma_1 - k_psi
        A[5,5] = self.gamma_2 - k_psi
        B[5,5] = k_psi

        A[2,2] = self.delta_1 - self.delta_2 * k_v
        A[2,8] = - self.delta_2 * k_p
        B[4,4] = self.delta_2 * k_p

        #return np.diag([dv_x, dv_y, dv_z, dw_x, dw_y, dw_z, dp_x, dp_y, dp_z, dtheta_x, dtheta_y, dtheta_z])
        #return (A,B)

        A *= self.dT
        
        # Convert B from R 12 x 6 to R 12 x 4. Why? The theta setpoints are always zero.
        conversionMatrix = np.zeros((6,4))
        conversionMatrix[0,0] = 1
        conversionMatrix[1,1] = 1
        conversionMatrix[4,2] = 1
        conversionMatrix[5,3] = 1
        
        return (A, B @ conversionMatrix * self.dT)
    
        # I need to factor out the input matrices, if possible, to form...
        # xdot = Ax + Bu.

    # Function: setpoint_inputs(U)
    # Purpose: save inputs as global variables, for use with the blimp swing-reducing simplified models.
    def setpoint_inputs(self, U): # "Update the inputs"
        self.v_setpoint_x = U[0] # Set the velocity in the x direction (not the position)
        self.v_setpoint_y = U[1] # Set the velocity in the y direction
        self.theta_setpoint_x = U[2] # Set the pitch angle (start: 0)
        self.theta_setpoint_y = U[3] # Set the roll angle (start: 0)
        self.z_setpoint = U[4] # Set the position in z.
        self.psi_setpoint = U[5] # Set the yaw angle
        
    # Function: plot_results(t, nu)
    # Purpose: plots the states of the blimp in a 4x3 subplot, for a total of twelve plots.
    # nu: state, t: time vector
    def plot_results(self, t, nu, show=True):
        assert(t.size == nu.shape[1])
        # Plot each state in a grid of subplots, 12 by 12.
        _, axs = plt.subplots(4, 3, sharex=True)
        for i in range(12):
            axs[int(i/3), i % 3].plot(t, nu[i, :])
            axs[int(i/3), i % 3].set_ylabel(self.get_state(i))
        # _.set_xlabel('t')

        if show:
            plt.show()

    # Function: get_state(i)
    # Purpose: based on an index i, select the appropriate state label in a LaTeX wrapper.
    def get_state(self, i):
        if i == 0:
            return '$v_x$'
        elif i == 1:
            return '$v_y$'
        elif i == 2:
            return '$v_z$'
        elif i == 3:
            return '$\omega_{\\theta}$'
        elif i == 4:
            return '$\omega_{\phi}$'
        elif i == 5:
            return '$\omega_{\psi}$'
        elif i == 6:
            return '$x$'
        elif i == 7:
            return '$y$'
        elif i == 8:
            return '$z$'
        elif i == 9:
            return '$\\theta$'
        elif i == 10:
            return '$\phi$'
        elif i == 11:
            return '$\psi$'

# Luke Baird
# (c) Georgia Institute of Technology, 2023
# Uses math from Tao et. al.: Swing-Reducing Flight Control System for an Underactuated Indoor
# Miniature Autonomous Blimp.
import scipy.integrate

class Continuous:
    # This class uses scipy.integrate.solve_ivp to propagate the dynamics.

    def __init__(self):
        # Create a target point as a global variable.
        self.x_target = np.array([[2],[2],[1],[0]])
        self.blimp = Blimp()

        # Create an initial point
        x0 = np.zeros((4,1)) # match waypoint maneuver states

        t_span = [0, 30]
        self.t_last = t_span[0] # start time for integration.

        nu = np.zeros((12)) # state
        nu[6:9] = x0[0:3, 0]
        nu[11] = x0[3,0]

        self.v_last = np.array([nu[0], nu[1]]) # initial xy velocity of the blimp for swing-reducing flight controller.
        
        simulation = scipy.integrate.solve_ivp(self.ddt, t_span, nu)
        # message, nfev, njev, nlu, sol, status, success, t, y, t_events, y_events
        self.plot_results(simulation.t, simulation.y)
    
    def swing_reducing_controller(self, nu, v_setpoint, t):
        # 1. Outer loop: select theta_setpoint for the inner loop.
        # 2. Inner loop: select f_x, f_y for theta_setpoint.

        # The relevant functions should primarily occur in blimp.py, as they use the blimp's model parameters.
        # Or I can be lazy and do them here...

        k_p = np.array([0.27, 0.27])
        k_i = np.array([0.02, 0.02])
        k_theta = np.array([0.27, 0.27])
        k_omega = np.array([0.25, 0.25])

        # k_r:  [0.27, 0., -1.6]
        # k_p:  [0.27, 0., -1.6]
        # k_yw: [0.002, 0.001, 0.07]
        # k_pz: [0.29, 0.001, 0.045]

        # Step 1: theta_setpoint = a PI controller.
        theta_setpoint = np.arcsin(self.blimp.theta_setpoint_coefficient * v_setpoint) - k_p * (nu[0:2, 0] - v_setpoint)

        # Perform the accumulation separately, using the same t as the outer integration function.
        self.v_last += (nu[0:2, 0] - v_setpoint) * (t - self.t_last)
        theta_setpoint -= k_i * self.v_last # integral accumulation

        theta_setpoint = np.flip(theta_setpoint)
        
        # Step 2: calculate the applied force.
        f_xy = self.blimp.inner_loop_coefficient * np.sin(theta_setpoint) - k_theta * (nu[9:11, 0] - theta_setpoint) - k_omega * nu[3:5, 0]
        return np.flip(f_xy.reshape((2,1)))

    def pd_half_controller(self, nu, x_target):
        # PD waypoint controller
        p = np.array([[nu[8, 0]], [nu[11, 0]]])
        d = np.array([[nu[2, 0]], [nu[5, 0]]])

        Kp = np.diag([0.5, 0.5])
        Kd = np.diag([0.3, 0.2])

        return -Kp @ (p - x_target[2:, :]) - Kd @ d

    def setpoint_velocity(self, nu, x_target):
        # get setpoint velocities for x and y
        k_w = np.array([0.2, 0.2])
        return -np.array([nu[6,0] - x_target[0, 0], nu[7,0] - x_target[1, 0]]) * k_w
        # deviation in the desired distance = setpoint velocity

    def ddt(self, t, x):
        # convert x to a 2D array for numpy manipulations
        nu = np.expand_dims(x, 0).T

        # Get setpoint velocity
        v_setpoint = self.setpoint_velocity(nu, self.x_target)
        # print(v_setpoint)

        # Get the x, y forces from this.
        u_xy = self.swing_reducing_controller(nu, v_setpoint, t) # in R2x1

        # Use a PD controller for torque about z, and f_z.
        u_zpsi = self.pd_half_controller(nu, self.x_target) # in R2x1
        
        u = np.block([[u_xy], [u_zpsi]])

        self.t_last = t # update the previous time step.
        return self.blimp.ddt(nu, u)[:, 0]
    