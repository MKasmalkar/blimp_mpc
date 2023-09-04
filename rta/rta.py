#!/usr/bin/env python3

# File: rta.py
# Purpose: Implements the algorithm in "Guaranteeing Signal Temporal Logic Safety Specifications with Runtime Assurance"
#          This version works in a ROS2 Node.
# 
# This demonstrates optimization for the blimp with a nontrivial safety specification and the nonlinear
# blimp model. The "recursively feasible set" is now a bit more complicated and requires more future time
# steps to compute it (b > 1). 
# 
# Author: Luke Baird
# (c) Georgia Institute of Technology, 2023

import numpy as np
import time
import signal # for interrupts
import scipy
import scipy.io # For matlab logging.
from scipy.signal import lfilter, butter, lfilter_zi, filtfilt
from stlpy.systems import LinearSystem
from stlpy.STL import LinearPredicate
from stlpy.solvers import GurobiMICPWarmStartSolver
# from matplotlib import pyplot as plt # Python plotting
# from matplotlib import rc
from . polytopes import Polytope
from . blimp import Blimp
from . utils import * #quat2euler, euler2quat
from quickzonoreach.zono import get_zonotope_reachset

# Ros2 imports
import rclpy
from rclpy import executors
from rclpy.node import Node
from geometry_msgs.msg import Quaternion, Vector3
from nav_msgs.msg import Odometry
from mocap_msgs.msg import RigidBody

# Class: RTAPublisher
# Purpose: Publishes input messages to the Ros2 message bus for the blimp node.
class RTANode(Node):
    
    def __init__(self, blimp_id: int) -> None:
        super().__init__(f'rta_routine_{blimp_id}')
        self.publisher_ = self.create_publisher(
            Quaternion,
            f"/agents/blimp{blimp_id}/motion_command",
            10 # Queue size.
        )

        # Noise notes: there is noise in velocity and angular velocity. This should be filtered.

        # Create some class variables
        self.x = np.zeros((12,1)) # Current state to subscriber to information.
        self.x[6] = 1.5 # Initial states for STL computation purposes.
        self.x[7] = 1.5
        # NOTE: x is in the nav frame except for states 3,4,5, which are in the body frame.

        # Backup input sequence initialization
        self.u_index = 0
        self.u_backup = np.zeros((4,1))

        self.mocap_window_width = 12 # More!
        self.mocap_filter_coefficient = 0.8 # 450 Hz cutoff frequency
        self.msgs = [None] * self.mocap_window_width # save the last 10 messages for filtering.
        self.msg_idx = 0
        self.gyro_filter_coefficient = 0.9 # save the last 5 gyro messages for filtering.
        self.gyro_idx = 0

        self.x_hist = None # Historical states

        self.reference_zero = np.zeros(6) # For calibration.
        self.reference_set = False

        self.dT = 0.25
        self.square_dim = 1 + .38 + 0.03 # m, Dimensions of the square in the center of the room.
        # square is 2x2, half is 1.
        # 0.38 = radius of the blimp
        # 0.03 is the width of the tape on the floor.
        self.gyro_updated = False
        self.mocap_updated = False
        # Ready or not, publish an input message every dT
        self.timer = self.create_timer(self.dT, self.publish_input)

        self.save_data_timer = self.create_timer(3.0, self.cleanup) # every three seconds, save the data if possible.

        self.current_target = 1 # Which of the four targets in the room to start at

        # Variables for logging to MatLab
        self.x_reported = []
        self.x_reported_projected = []
        self.u_reported = []
        self.rho_reported = []
        self.t_reported = []

        # Transformation matrices
        # self.R_mocap_to_nav = rotation_matrix(np.pi, 0, 0)
        # self.T_mocap_to_nav = transformation_matrix(np.pi, 0, 0)
        # self.R_nav_to_cob = np.eye(3) # No rotation until computed properly.
        self.R_mocap_to_body = rotation_matrix(np.pi,0,0) # initialization.
        self.T_mocap_to_body = transformation_matrix(np.pi,0,0) # initialization.

        # Setup subscriptions
        self.update_mocap_subscription = self.create_subscription(
            RigidBody,
            f'/rigid_bodies',
            self.update_mocap,
            2 # Queue size.
        )

        self.update_gyros_subscription = self.create_subscription(
            Vector3,
            f'agents/blimp{blimp_id}/gyros',
            self.update_gyros,
            2 # Queue size.
        )

        # Initialize the solver
        self.setup_optimization_program()

        self.t0 = time.time()

    def cleanup(self):
        scipy.io.savemat('blimp-rta-output.mat', mdict={
            'x': np.block(self.x_reported),
            'x_projected': np.array(self.x_reported_projected),
            'u': np.block(self.u_reported),
            'rho': np.array(self.rho_reported), # likely not very useful.
            't': np.array(self.t_reported)
        })
    
    def print_x_deg(self):
        x_deg = self.x.copy()
        x_deg[3:6] *= 180 / np.pi
        x_deg[9:] *= 180 / np.pi
        self.get_logger().info(f'{x_deg}')

    # Subscriber functions.

    # Function: update_mocap
    # Purpose: get the next state reading for p, v, and theta.
    def update_mocap(self, msg):
        self.mocap_updated = True
        self.msg_idx = (self.msg_idx + 1) % self.mocap_window_width
        self.msgs[self.msg_idx] = msg

        # Do processing for velocity here.
        # compute velocity update
        # v = 0.9 * v_new + 0.1 * v
    
    # Function: extract_mocap
    # Purpose: extract the latest pose data from the last message received, to save computation time.
    def extract_mocap(self):
        msg = self.msgs[self.msg_idx] # is MOCAP frame.
        self.x[6:9, 0] = np.array([msg.pose.position.x, msg.pose.position.y, msg.pose.position.z])
        self.x[9:, 0] = quat2euler([msg.pose.orientation.x, msg.pose.orientation.y,
                                          msg.pose.orientation.z, msg.pose.orientation.w])
        # self.x[11, 0] += 0.25 # TODO calibration
        # self.x[9:, 0] = quat2euler(np.array([msg.pose.orientation.w, msg.pose.orientation.x,
        #                                   msg.pose.orientation.y, msg.pose.orientation.z]))
        # self.x[6:, 0] -= self.reference_zero # TODO re-add zeroing.
        
        # TODO: later, define zero yaw using this!
        if not self.reference_set:
            # self.reference_zero[0:3] = np.array([msg.pose.position.x, msg.pose.position.y, msg.pose.position.z])
            self.reference_zero[3:6] = quat2euler(np.array([msg.pose.orientation.x, msg.pose.orientation.y,
                                          msg.pose.orientation.z, msg.pose.orientation.w]))
            self.reference_set = True
        
        # Transform state data to NAV frame.
        # self.x[6:9, 0:1] = self.R_mocap_to_nav @ np.array([[msg.pose.position.x],
        #                                                    [msg.pose.position.y],
        #                                                    [msg.pose.position.z]])
        # self.x[9:, 0:1] = self.R_mocap_to_nav @ self.x[9:, 0:1]

    # Function: update_gyros
    # Purpose: get the next state reading for w,
    def update_gyros(self, msg):
        self.gyro_updated = True

        # self.x[3:6, 0] = (1 - self.gyro_filter_coefficient) * self.x[3:6,0] + \
        #     self.gyro_filter_coefficient * np.array([msg.x, msg.y, msg.z])
        self.x[3:6, 0] = np.array([msg.x, msg.y, msg.z])
        # Updating self.x might be too slow, or conflict with RTA computations that use x. (Race condition)
        # Instead, we can do a similar method to update_mocap and extract_mocap.

    
    # Publisher functions.

    # Function: publish_input
    # Purpose:  performs ros2 routine to publish data to an input.
    # Calls compute_input to get the input to send.
    def publish_input(self):
        # Update x
        now = time.time()
        if self.mocap_updated:
            self.extract_mocap()
            self.R_mocap_to_body = rotation_matrix(np.pi,0,0) @ rotation_matrix(self.x[9,0], self.x[10,0], self.x[11,0]).T
            self.T_mocap_to_body = transformation_matrix(np.pi,0,0) @ rotation_matrix(self.x[9,0],self.x[10,0],self.x[11,0]).T
            self.compute_velocity_from_position() # Get the current velocity.
        x_transformed = self.x.copy()
        # Transform the state data for the compute_input function to use.
        # Rotate the velocity into the body frame.
        x_transformed[0:3, :] = self.R_mocap_to_body @ x_transformed[0:3, :]
        x_transformed[6:9, :] = self.R_mocap_to_body @ x_transformed[6:9, :]
        x_transformed[9:,:] = self.T_mocap_to_body @ x_transformed[9:,:]
        self.get_logger().info(f'{x_transformed}')
        
        self.x_hist = np.block([self.x_hist[:, 1:], x_transformed])
        self.x_reported.append(x_transformed.copy()) # before running any input computations.
        if self.mocap_updated:
            self.mocap_updated = False
        msg = Quaternion() # Hacky, but works. We should define a custom message type.


        data = self.compute_input().squeeze(1)
        # This is now in the body frame! Therefore...
        msg.x = data[0]
        msg.y = data[1]
        msg.z = data[2]
        msg.w = data[3]

        # Rotate the planar force so it matches x, y in the mocap frame.
        # Rotate about z by positive psi, as we are going from NAV to COB.

        # msg.x = data[0] * np.cos(self.x[11,0]) + data[1] * np.sin(self.x[11, 0])
        # msg.y = data[0] * np.sin(self.x[11,0]) - data[1] * np.cos(self.x[11, 0])

        # # I think this all comes from rotation / transformation matrices.

        # msg.z = -data[2]
        # msg.w = -data[3]

        self.publisher_.publish(msg)
        self.get_logger().info(f'Input publisher time is {time.time() - now}')
    
    # Function: compute_velocity_from_position
    # Purpose: compute an estimate of velocity from the current position
    def compute_velocity_from_position(self):
        # use self.msgs to compute this
        try:
            # Another idea from Akash's professor
            # self.mocap_filter_coefficient is a cutoff frequency of ~450Hz.
            v_update = np.zeros((3,))#self.x[0:3, 0].copy() # NOT np.zeros! Even this is not really correct...
            for j in range(0, self.mocap_window_width - 3, 2): # The indices are chosen s.t. we start with the oldest msg, and work towards the newest msg.
                idx_now = (self.msg_idx + j + 3) % self.mocap_window_width # current index.
                idx_last = (self.msg_idx + j + 1) % self.mocap_window_width # previous index.
                msg_now = self.msgs[idx_now]
                msg_last = self.msgs[idx_last]
                if msg_now is None or msg_last is None: # don't do anything.
                    self.get_logger().warn("not enough messages to get an accurate v yet.")
                    continue
                x_now = np.array([msg_now.pose.position.x, msg_now.pose.position.y, msg_now.pose.position.z])
                x_last = np.array([msg_last.pose.position.x, msg_last.pose.position.y, msg_last.pose.position.z])

                t_now = float( '{:d}.{:09d}'.format(msg_now.header.stamp.sec, msg_now.header.stamp.nanosec) )
                t_last = float( '{:d}.{:09d}'.format(msg_last.header.stamp.sec, msg_last.header.stamp.nanosec) )

                if j >= 1:
                    v_update = self.mocap_filter_coefficient * ((x_now - x_last) / (t_now - t_last)) + \
                        (1 - self.mocap_filter_coefficient) * v_update
                else:
                    v_update = ((x_now - x_last) / (t_now - t_last)) # initialization point.

            # The following update is in the MOCAP frame.
            self.x[0:3, 0] = self.mocap_filter_coefficient * v_update + (1-self.mocap_filter_coefficient) * self.x[0:3, 0]
            # TODO self.R_mocap_to_nav @ v_update # MOCAP to NAV to COB frame
        except Exception as e:
            self.get_logger().warn(f'{e}')
            self.get_logger().debug("not enough messages to use the filter to get velocity yet.")
            # self.x[0:3, 0] = np.zeros(3) # if there's not enough messages to get an accurate estimate.
            # Don't update self.x[0:3, 0].


    # Function: nominal_controller
    # Purpose: Given a waypoint, compute a nominal PD controller
    #          Nominal z-controller and yaw-controller are zero.
    def nominal_controller(self):
        nominal = np.zeros((4, 1))
        x = self.x.copy()
        # self.x is known.# Nominal controller parameters
        
        targets = np.array([[0, self.square_dim + 0.1], [self.square_dim + 0.1, 0],
                            [0, -(self.square_dim + 0.1)], [-(self.square_dim + 0.1), 0]])
        close_enough = 0.1 # within 5 cm, switch to the next waypoint for the blimp

        # Setup (4) waypoints, and use a PD controller to approach each of them.
        currentXY = x[6:8, 0] # self.x_hist[6:8, -1]
        if np.all(np.abs(currentXY - targets[self.current_target, :]) <= close_enough):
            self.get_logger().debug('updating the current target')
            self.current_target = (self.current_target + 1) % 4

        # self.x is known. # z nominal = 2.0 m
        target = np.array([[targets[self.current_target, 0]], [targets[self.current_target, 1]], [2.0], [0]]) # (0,0,1) in space
        
        dX = x[6, 0] - target[0]
        dY = x[7, 0] - target[1]
        dZ = x[8, 0] - target[2]
        dPsi = (x[11, 0] - target[3])# % (2*np.pi)

        dVX = x[0, 0]
        dVY = x[1, 0]
        dVZ = x[2, 0]
        dWPsi = x[5, 0]

        nominal[0, 0] = -0.05 * dX - 0.03 * dVX
        nominal[1, 0] = -0.05 * dY - 0.03 * dVY
        nominal[2, 0] = -0.1 * dZ - 0.04 * dVZ
        nominal[3, 0] = -0.001 * dPsi - 0.07 * dWPsi # this should not rotate quickly ever!!!

        if nominal[2, 0] < 0: # applying a downwards force
            nominal[2, 0] *= 0.35 # weaken it.

        max_xy = 0.03
        max_torque = 0.0005
        # Apply saturation values.
        if np.abs(nominal[0, 0]) > max_xy:
            nominal[0, 0] = max_xy * np.sign(nominal[0, 0])
        if np.abs(nominal[1, 0]) > max_xy:
            nominal[1, 0] = max_xy * np.sign(nominal[1, 0])
        if nominal[2, 0] > 0.0908:
            nominal[2, 0] = 0.0908
        elif nominal[2, 0] < -0.0366:
            nominal[2, 0] = -0.0366
        if np.abs(nominal[3, 0]) > max_torque:
            nominal[3, 0] = max_torque * np.sign(nominal[3, 0])

        return nominal

    def nominal_pd_controller(self):
        nominal = np.zeros((4, 1))
        # self.x is known.# Nominal controller parameters
        x = self.x.copy()
        
        targets = np.array([[1, 1], [-1, -1],
                            [1, 1], [-1, -1]])
        close_enough = 0.1 # within 5 cm, switch to the next waypoint for the blimp

        # Setup (4) waypoints, and use a PD controller to approach each of them.
        currentXY = x[6:8, 0] # self.x_hist[6:8, -1]
        if np.all(np.abs(currentXY - targets[self.current_target, :]) <= close_enough):
            self.get_logger().debug('updating the current target')
            self.current_target = (self.current_target + 1) % 4

        # self.x is known. # z nominal = 2.0 m
        target = np.array([[targets[self.current_target, 0]], [targets[self.current_target, 1]], [2.0], [0]]) # (0,0,1) in space
        
        dX = x[6, 0] - target[0]
        dY = x[7, 0] - target[1]
        dZ = x[8, 0] - target[2]
        dPsi = (x[11, 0] - target[3])# % (2*np.pi)

        dVX = x[0, 0]
        dVY = x[1, 0]
        dVZ = x[2, 0]
        dWPsi = x[5, 0]

        nominal[0, 0] = -0.05 * dX - 0.04 * dVX
        nominal[1, 0] = -0.05 * dY - 0.04 * dVY
        nominal[2, 0] = -0.1 * dZ - 0.04 * dVZ
        nominal[3, 0] = -0.001 * dPsi -0.04 * dWPsi # this should not rotate quickly ever!!! # -0.0005 * dPsi + 

        if nominal[2, 0] < 0: # applying a downwards force
            nominal[2, 0] *= 0.35 # weaken it.

        max_xy = 0.03
        max_torque = 0.0005
        # Apply saturation values.
        if np.abs(nominal[0, 0]) > max_xy:
            nominal[0, 0] = max_xy * np.sign(nominal[0, 0])
        if np.abs(nominal[1, 0]) > max_xy:
            nominal[1, 0] = max_xy * np.sign(nominal[1, 0])
        if nominal[2, 0] > 0.0908:
            nominal[2, 0] = 0.0908
        elif nominal[2, 0] < -0.0366:
            nominal[2, 0] = -0.0366
        if np.abs(nominal[3, 0]) > max_torque:
            nominal[3, 0] = max_torque * np.sign(nominal[3, 0])

        return nominal
    
    def sys_id_controller(self):
        nominal = np.zeros((4, 1))
        t = (time.time() - self.t0) % 32

        if t <= 2:
            pass
        elif t <= 7:
            nominal[0,0] = 1
        elif t <= 12:
            nominal[0,0] = -1
        elif t <= 17:
            nominal[1,0] = 1
        elif t <= 22:
            nominal[1,0] = -1
        elif t <= 25:
            nominal[2,0] = 1
        elif t <= 27:
            nominal[2,0] = -1
        # else: do nothing for five seconds.
        
        dPsi = self.x[11, 0] # negative to take yaw from NAV to COB frame.
        dWPsi = self.x[5, 0]

        nominal[3, 0] = -0.001 * dPsi - 0.041 * dWPsi # this should not rotate quickly ever.

        max_xy = 0.04752
        max_torque = 0.00126
        # Apply saturation values.
        if np.abs(nominal[0, 0]) > max_xy:
            nominal[0, 0] = max_xy * np.sign(nominal[0, 0])
        if np.abs(nominal[1, 0]) > max_xy:
            nominal[1, 0] = max_xy * np.sign(nominal[1, 0])
        if nominal[2, 0] > 0.0908:
            nominal[2, 0] = 0.0908
        elif nominal[2, 0] < -0.0366:
            nominal[2, 0] = -0.0366
        if np.abs(nominal[3, 0]) > max_torque:
            nominal[3, 0] = max_torque * np.sign(nominal[3, 0])

        return nominal

    # Function: compute_input
    # Purpose: compute a provably safe control action with respect to the node's STL formula.
    def compute_input(self):
        # self.get_logger().info(f'{np.round(self.x[9:,:] * 180 / np.pi, 2)}')
        # self.print_x_deg()

        # The last historical 'x' is saved by the function that calls compute_input().

        # Positional states are in the NAV frame.
        # Dynamical states (velocity, rotation) are in the BODY frame.

        self.u_index += 1 # in case the solver fails.

        u_nominal = self.nominal_controller()
        # u_nominal = self.sys_id_controller()
        # u_nominal = self.nominal_pd_controller()
        # self.t_reported.append(time.time()) # TODO comment.
        # return u_nominal
        # Coordinate transform u_nominal.
        u_nominal[:3, :] = self.R_mocap_to_body @ u_nominal[:3, :]
        u_nominal[3, :] = u_nominal[3, :] * -1 # cheating here.
        # self.u_reported.append(u_nominal)
        # return u_nominal # to start.

        self.program.updateModel(sys=self.program.sys, x0=self.x_hist, i=self.horizon+1, u_hat=u_nominal)
        now = time.time()
        solution = self.program.Solve()
        self.get_logger().info(f'{time.time() - now} for compute_input Solve')
        if solution[5]:
            try:
                # pass
                # It is important to log the following information:
                # 1. Computed input.
                # 2. self.x
                # 3. the solved future input.
                # It returns (x,u,rho,self.model.Runtime,objective,success)
                self.x_reported_projected.append(solution[0][:, self.horizon:self.horizon+self.N])

                # Reset the backup input sequence with the model solution.
                self.u_index = 0
                self.u_backup = solution[1][:, self.horizon-1:] # Backup input sequence
                self.u_reported.append(self.u_backup[:, self.u_index:self.u_index+1])

                self.rho_reported.append(solution[2]) # 0 = safety spec # [0, 0]
                # self.get_logger().info(f'{np.min(solution[2][0, :, :])}')
                # self.get_logger().info(f'{solution[2]}')
                # 0:1 = historical robustness (only know one, rest are projected).

                self.t_reported.append(time.time())
                # self.get_logger().info(f'{self.blimp.K @ self.x}')
                return self.u_backup[:, self.u_index:self.u_index+1]# - self.blimp.K @ self.x
            
                # A coordinate transform of some sort is needed...
            except:
                self.get_logger().error(f'AN ERROR HAS OCCURRED!')
        else:
            self.get_logger().warn(f'An error has occurred!')
        
        # Logging regardless.
        self.x_reported_projected.append(np.zeros((12, self.N))) # This should be "NaN" or something.
        # self.rho_reported.append(np.zeros((12,self.N))) # For now... don't append anything. Obviously, bad!
        self.t_reported.append(time.time())

        if self.u_index < self.u_backup.shape[1]:
            self.get_logger().warn(f'Applying a backup solution, u_index={self.u_index}')
            self.u_reported.append(self.u_backup[:, self.u_index:self.u_index+1])
            return self.u_backup[:, self.u_index:self.u_index+1]
        else: # Je me rends!
            self.u_reported.append(u_nominal)
            return u_nominal


    # Function: setup_optimization_program
    # Purpose: construct the persistent functions and optimization routines for the program
    #          e.g., STL formulae, persistently feasible set, 1-norm to LP, etc.
    def setup_optimization_program(self):
        # u_constraints = np.array([[-0.04752, 0.04752], # Computed from "diamond" constraint shape.
        #                           [-0.04752, 0.04752], # Computed from "diamond" input shape
        #                           [-0.0366, 0.0908], # Directly read off
        #                           [-0.00126, 0.00126]]) # Directly read off
        # u_constraints = np.array([[-0.04752, 0.04752], # 0.0672
        #                           [-0.04752, 0.04752], # 0.0472
        #                           [-0.0266, 0.0808], # -0.0366, 0.0908
        #                           [-0.00126, 0.00126]]) # From blimp parameters # 0.0013
        
        u_constraints = np.array([[-0.04752, 0.04752], # 0.0672
                                  [-0.04752, 0.04752], # 0.0472
                                  [-0.0908, 0.0366], # -0.0366, 0.0908
                                  [-0.00005, 0.00005]]) # From blimp parameters # 0.00126
        
        # Define the disturbance set.
        # Passive
        # w = np.array([[0.005, 0.005, 0.001, 0.0065, 0.0065, 0.0065, 0.00015, 0.00015, 0.00015, 0.01, 0.01, 0.001]])
        # Aggressive
        # w = np.array([[0.02, 0.02, 0.005, 0.009, 0.009, 0.009, 0.001, 0.001, 0.001, 0.01, 0.01, 0.001]])
        # Hyper-aggressive.
        w = np.array([[0.04, 0.04, 0.005, 0.01, 0.01, 0.01, 0.008, 0.008, 0.008, 0.008, 0.008, 0.001]])

        w = np.block([[-w],[w]]) # This is the "input box" for the zonotopes.
        
         # Create a new linearized blimp model
        my_blimp = Blimp()
        my_blimp.setup_discrete_lti_model(self.dT)
        my_blimp.discretize(dT=self.dT)
        my_blimp.compute_stable_physics_model_feedback()
    
        # Extract matrices for their size.
        # A,B = my_blimp.stable_physics_model()
        # Instead!
        A = my_blimp.A_discrete
        B = my_blimp.B_discrete

        # Create and set the outputs of the blimp in R6.
        C = np.block([np.zeros((6,6)), np.eye(6)]) # outputs = positions and angles.
        my_blimp.C_(C)

        # Current blimp model:
        # x[k+1] = Ax[k] + Bu[k]
        # y[k] = Cx[k]

        # Build the STL Formulae. "Positive Normal Form" (can only negate predicates.)
        A_height = np.zeros((6,1))
        A_height[2, 0] = 1

        # Depending on the appropriate coordinate transformations / frames.
        lb_h = LinearPredicate(A_height, 1.0) # above z=+1.0
        ub_h = LinearPredicate(-A_height, -2.5) # below z=2.5


        # lb_h = LinearPredicate(-A_height, 0.5) # -z >= 0.5, z <= -0.5
        # ub_h = LinearPredicate(A_height, -2.5) # z >= -2.5

        # Lower and upper bounds on the z-axis
        height_spec = lb_h & ub_h

        A_pitch = np.zeros((1,6))
        A_roll = np.zeros((1,6))
        A_pitch[0,4] = 1#0.5
        A_roll[0,3] = 1
        lb_theta = LinearPredicate(A_pitch, -0.2) # above -0.2
        ub_theta = LinearPredicate(-A_pitch, -0.2) # below 0.2
        lb_phi = LinearPredicate(A_roll, -0.2)
        ub_phi = LinearPredicate(-A_roll, -0.2)
        angle_spec = (lb_theta & ub_theta & lb_phi & ub_phi)

        # Create the square.
        A_x = np.zeros((1,6))
        A_y = np.zeros((1,6))
        A_x[0, 0] = 1
        A_y[0, 1] = 1
        lb_x = LinearPredicate(-A_x, self.square_dim) # Safe: x <= -square_dim
        ub_x = LinearPredicate(A_x, self.square_dim) # Safe: x >= square_dim
        lb_y = LinearPredicate(-A_y, self.square_dim) # Safe: y <= -square_dim
        ub_y = LinearPredicate(A_y, self.square_dim) # Safe: y >= square_dim
        # Recall: implication a=>b is ~a | b. # (lb_x | ub_x | lb_y | ub_y) |
        x_between_0_and_20_spec = (lb_x | ub_x | lb_y | ub_y) | (lb_x | ub_x | lb_y | ub_y).always(0,
                                 round(1/self.dT) - 1).eventually(0, round(3/self.dT) - 1) # 3 / self.dT
        easy_spec = (lb_x | ub_x | lb_y | ub_y) | (lb_x | ub_x | lb_y | ub_y).eventually(0, round(5/self.dT) - 1)
        # pi = (x_between_0_and_20_spec, angle_spec, height_spec) # This is the specification that we will enforce.
        pi = x_between_0_and_20_spec # another attempt - remove other specifications for higher DoF.

        # Setup model parameters.
        self.horizon = round(4 / self.dT) - 1
        fts = self.horizon*2 # future time steps to project the system out. (N=horizon, b=1, applying Proposition 1 in the paper).
        self.N = self.horizon-1 #int(horizon / 2) # Where to place the terminal set constraint along the prediction horizon.

        # Create an array to hold a buffer of horizon states in the past
        self.x_hist = np.zeros((12, self.horizon+1))
        self.x_hist[6:8, :] = self.square_dim * 2 # that way, robustness is always positive for dummy past states.
        self.x_hist[8, :] = 2 # 2m, initialize height spec with a positive robustness value
        
        # Create a dummy system with the same dimensions as the blimp to avoid errors.
        D = np.zeros((6,4))
        sys = LinearSystem(A, B, C, D)

        # Logging: system dimensions.
        self.get_logger().info('System dimensions.')
        self.get_logger().info(f'p: {sys.p}, n: {sys.n}, m: {sys.m}')

        # Create zonotopic disturbance sets for tubeMPC
        a_mat = ((A - np.eye(12)) / self.dT) # Continuous-time dynamics
        b_mat = np.eye(12) / self.dT # Continuous-time dynamics
        init_box = w.T
        a_mat_list = []; b_mat_list = []; input_box_list = []; dt_list = []
        for _ in range(self.N):
            a_mat_list.append(a_mat)
            b_mat_list.append(b_mat) # A^j @ W
            input_box_list.append(w.T)
            dt_list.append(self.dT)
        # Perhaps try quick=False?
        zonotopes = get_zonotope_reachset(init_box, a_mat_list, b_mat_list, input_box_list, dt_list, quick=True)

        # Use overbounding zonotopes for the y_bounds
        y_bounds = [] # Noting y \in R^6
        for i, z in enumerate(zonotopes):
            # i = index, z = zonotope
            box = z.box_bounds()
            temp_bounds = box[0:6, 1:]
            temp_bounds[3] = min(temp_bounds[3], 0.099) # avoid set of measure zero in the pitch/roll angle specification
            temp_bounds[4] = min(temp_bounds[4], 0.099)
            y_bounds.append(temp_bounds) # The columns of the A-matrix are standard basis vectors.
        y_bounds.pop() # Remove extraneous last element.

        Qp = self.persistently_feasible_set()
        self.program = GurobiMICPWarmStartSolver(spec=pi, sys=sys, model=my_blimp, x0=self.x_hist[:, 0:1], T=self.horizon+fts,
             robustness_cost=True, hard_constraint=False, horizon=self.horizon, verbose=False, dT=self.dT, N=self.N, rho_min=-np.inf,
             tube_mpc_enabled=True, tube_mpc_buffer=y_bounds)
        self.program.AddControlBounds(u_constraints[:, 0], u_constraints[:, 1])
        self.program.AddRecursiveFeasibilityConstraint(Qp)
        self.program.AddLPCost()

        # 1 norm: worst-case is 0.1765, with current code
        # 2 norm: 0.1048
        # self.program.AddInputOneNormConstraint(0.09) 

        self.blimp = my_blimp


    # Function: persistently_feasible_set
    # Purpose: compute a tuple of polytopes representing a nonconvex terminal set constraint for MPC
    def persistently_feasible_set(self):
        H_x = np.zeros((17,12))
        h_x = np.zeros((17,1))
        H_x[0, 2] = 1; h_x[0, 0] = 0.1#0.5 # v_z <= 0.1
        H_x[1, 2] = -1; h_x[1, 0] = 0.1#0.5 # -v_z <= 0.1, v_z >= -0.1
        H_x[2, 8] = 1; h_x[2, 0] = 0.5 # z <= 0.5
        H_x[3, 8] = -1; h_x[3, 0] = 2.5 # -z <= 2.5, z >= -2.5

        H_x[4, 9] = 1; h_x[4, 0] = 0.2 # pitch <= 0.2 rad
        H_x[5, 9] = -1; h_x[5, 0] = 0.2 # pitch >= -0.2 rad
        H_x[6, 10] = 1; h_x[6, 0] = 0.2 # roll <= 0.2 rad
        H_x[7, 10] = -1; h_x[7, 0] = 0.2 # roll >= -0.2 rad

        H_x[8,4] = 1; h_x[8,0] = 0.3 # w_y <= 0.4
        H_x[9,4] = -1; h_x[9,0] = 0.3 # w_y >= -0.4
        H_x[10,3] = 1; h_x[10,0] = 0.2 # w_x <= 0.2
        H_x[11,3] = -1; h_x[11,0] = 0.2 # w_x >= -0.2

        H_x[12,5] = 1; h_x[12,0] = 0.05 # w_z <= 0.05
        H_x[13,5] = -1; h_x[13,0] = 0.05 # w_z >= -0.05
        H_x[14,11] = 1; h_x[14,0] = 10; # yaw <= 10 rad
        H_x[15,11] = -1; h_x[15,0] = 10; # yaw >= -10 rad

        Hx = Polytope(H_x, h_x)
        
        # I need to create five persistently feasible sets as follows:
        # Set (1): convex constraints.
        # Set (j), non-convex constraints OR'd.
        H_x_left = np.zeros((2, 12))
        h_x_left = np.zeros((2, 1))
        H_x_left[0, 6] = 1; h_x_left[0, 0] = -self.square_dim # x <= -square_dim
        H_x_left[1, 0] = 1; h_x_left[1, 0] = 0.01 # v_x <= 0.01

        H_x_right = np.zeros((2, 12))
        h_x_right = np.zeros((2,1))
        H_x_right[0, 6] = -1; h_x_right[0, 0] = -self.square_dim # -x <= -square_dim, x >= square_dim
        H_x_right[1, 0] = -1; h_x_right[1, 0] = 0.01; # -v_x <= .01, v_x >= -0.01

        H_y_top = np.zeros((2, 12))
        h_y_top = np.zeros((2, 1))
        H_y_top[0, 7] = -1; h_y_top[0, 0] = -self.square_dim # y >= square_dim
        H_y_top[1, 1] = -1; h_y_top[1, 0] = .01 # v_y <= 0.01

        H_y_bottom = np.zeros((2, 12))
        h_y_bottom = np.zeros((2, 1))
        H_y_bottom[0, 7] = 1; h_y_bottom[0, 0] = -self.square_dim
        H_y_bottom[1, 0] = 1; h_y_bottom[1, 0] = .01 # v_y >= 0.01

        # For now:
        return (Hx, Polytope(H_x_left, h_x_left), Polytope(H_x_right, h_x_right),
                Polytope(H_y_top, h_y_top), Polytope(H_y_bottom, h_y_bottom))
        
def main(args=None):
    rclpy.init(args=args)
    rta_routine = RTANode(0) # 0 is the blimp id for now.

    # bob = executors.MultiThreadedExecutor(num_threads=4)
    # bob.add_node(rta_routine)
    # bob.spin()

    def exit_handle(signum, frame):
        rta_routine.get_logger().info('command to cleanup and destroy rta node received.')
        rta_routine.destroy_node()
        rclpy.shutdown()

    signal.signal(signal.SIGINT, exit_handle) # Gracefully shutdown when user presses Ctrl+C.

    # Main execution loop spin
    rclpy.spin(rta_routine)

    # Shutdown sequence
    rta_routine.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()