#!/usr/bin/env python3
from __future__ import print_function
import roslib
import sys
import rospy
import numpy as np
import datetime
import time
from geometry_msgs.msg import Twist
from dse_msgs.msg import PoseMarkers
from std_msgs.msg import Float64MultiArray
from std_msgs.msg import MultiArrayLayout
from std_msgs.msg import MultiArrayDimension
from dse_msgs.msg import InfFilterPartials
from dse_msgs.msg import InfFilterResults
from scipy.spatial.transform import Rotation as R

import dse_lib
import dse_constants
roslib.load_manifest('dse_simulation')


class information_filter:

    # Set up initial variables
    # Pass in the ID of this agent and the state dimension (6 or 12)
    def __init__(self):

        # Get parameters from launch file
        self.ros_prefix = rospy.get_param('~prefix')
        if len(self.ros_prefix) != 0 and self.ros_prefix[0] != '/':
            self.ros_prefix = '/' + self.ros_prefix
        self.this_agent_id = rospy.get_param('~id')
        self.dim_state = rospy.get_param('~dim_state')
        self.init_ids = rospy.get_param('~initial_ids', [])
        self.init_est = rospy.get_param('~initial_estimates', [])
        self.pub_errors = rospy.get_param('~pub_errors', 0)

        # self.ros_prefix = '/tb3_0'
        # self.this_agent_id = 5
        # self.dim_state = 6
        # self.init_ids = [5, 0, 1, 2]
        # self.init_est = [-1.0, 0.0, 0.0, 0, 0, 0,
        #                   1.0,-0.5, 0.0, 0, 0, 0,
        #                   1.0, 0.0, 0.0, 0, 0, 0,
        #                   1.0, 0.5, 0.0, 0, 0, 0]
        # self.init_ids = []
        # self.init_est = []

        # Define publishers and subscribers
        # Subscribes to control signals
        self.control_sub = rospy.Subscriber(self.ros_prefix + '/cmd_vel', Twist, self.control_callback)
        # Subscribe to the final information filter results, from the direct estimator or later the consensus
        self.results_sub = rospy.Subscriber(self.ros_prefix + "/dse/inf/results", InfFilterResults, self.results_callback)
        # Subscribe to the pose output from the camera
        self.pose_sub = rospy.Subscriber(self.ros_prefix + "/dse/pose_simulated", PoseMarkers, self.measurement_callback)
        # Publish the information priors (inf_Y = Y_01) and the measurements (inf_I = delta_I)
        self.inf_pub = rospy.Publisher(self.ros_prefix + "/dse/inf/partial", InfFilterPartials, queue_size=10)

        # if self.pub_errors:
        #     self.camera_error_pub = rospy.Publisher(self.ros_prefix + "/diag/camera_error", PoseMarkers, queue_size=1)
        #     self.motion_error_pub = rospy.Publisher(self.ros_prefix + "/diag/motion_error", PoseMarkers, queue_size=1)
        #     self.imu_error_pub = rospy.Publisher(self.ros_prefix + "/diag/imu_error", PoseMarkers, queue_size=1)
        #     self.control_error_pub = rospy.Publisher(self.ros_prefix + "/diag/control_error", PoseMarkers, queue_size=1)

        # Grab the state dimension and make sure it is either 6 or 12, as only those two sizes are currently implemented.
        if self.dim_state == 6:
            self.dim_obs = 3
        elif self.dim_state == 12:
            self.dim_obs = 6
        else:
            rospy.signal_shutdown('invalid state dimension passed in')

        # Define static variables
        self.dt = 0.1
        self.t_last = rospy.get_time()
        self.euler_order = dse_constants.EULER_ORDER

        # Define information variables
        self.inf_P = []
        self.inf_x = []
        self.inf_I = []
        self.inf_i = []
        self.inf_id_obs = []
        self.inf_id_comm = []

        print(self.init_ids)
        print(self.init_est)

        # Initialize information variables
        if len(self.init_ids) > 0:
            self.inf_id_list = np.array(self.init_ids)
            n_agents = len(self.init_ids)
            self.inf_Y = dse_constants.INF_MATRIX_INITIAL * np.eye(self.dim_state*n_agents, dtype=np.float64)
            self.inf_y = self.inf_Y.dot(self.init_est)[:, None]
            print(np.shape(self.inf_y))
        else:
            self.inf_id_list = np.array([self.this_agent_id])
            self.inf_Y = dse_constants.INF_MATRIX_INITIAL * np.eye(self.dim_state, dtype=np.float64)
            self.inf_y = dse_constants.INF_VECTOR_INITIAL * \
                         np.transpose(1 * np.arange(1, self.dim_state + 1, dtype=np.float64))[:, None]

        # store the control data
        self.ctrl_twist = Twist()

    # When control signals are sent, store them. More logic to come later for storing more than just one agent.
    def control_callback(self, data):
        self.ctrl_twist = data

    # When the direct estimator or consensus returns the combined information variables
    def results_callback(self, data):
        self.inf_id_list = np.array(data.ids)
        self.inf_Y = dse_lib.multi_array_2d_output(data.inf_matrix)
        self.inf_y = dse_lib.multi_array_2d_output(data.inf_vector)

    # When the camera sends a measurement
    def measurement_callback(self, data):
        # Compute the actual dt
        self.dt = rospy.get_time() - self.t_last
        self.t_last = rospy.get_time()

        # Grab the tag poses from the camera
        observed_poses = data.pose_array.poses
        observed_ids = data.ids
        n = 1 + len(observed_ids)

        # update local values from the last time step
        Y_11 = self.inf_Y                       # Information matrix - Covariance
        y_11 = self.inf_y                       # Information vector - States
        x_11 = np.linalg.inv(Y_11).dot(y_11)    # Kalman State
        P_11 = np.linalg.inv(Y_11)              # Kalman Covariance
        id_list = self.inf_id_list              # list of all known IDs

        # print('old: ')
        # print(y_11)
        # If we find an ID that isn't currently known, add it
        id_list, Y_11, y_11, P_11, x_11 = dse_lib.extend_arrays(observed_ids, id_list, Y_11, y_11, self.dim_state)

        # Fill in R, H, z, F, and Q
        # R - Measurement Covariance
        # H - Measurement Jacobian
        # z - The measurement itself
        # This function is defined in src/dse_lib.py
        R_0, H_0, z_0 = dse_lib.fill_RHz_gazebo(id_list, self.this_agent_id, observed_ids, observed_poses, x_11,
                                         self.euler_order, self.dim_state, self.dim_obs)

        # F - Motion Jacobian
        # Q - Motion Covariance
        F_0, Q_0 = dse_lib.fill_FQ(id_list, self.dt, x_11, self.dim_state, self.dim_obs)

        # B - Control matrix
        # u - Control signals
        # This function is not ready yet.
        # B_0, u_0 = dse_lib.fill_Bu(id_list, self.this_agent_id, self.ctrl_twist, self.dim_state, self.dim_obs)

        # y = z_0 - H_0.dot(x_11)
        # for i in range(len(z_0) // 3):
        #     while np.abs(y[2 + 3*i]) > 2*np.pi:
        #         if y[2 + 3*i] > 0:
        #             z_0[2 + 3*i] = z_0[2 + 3*i] - 2 * np.pi
        #         else:
        #             z_0[2 + 3*i] = z_0[2 + 3*i] + 2 * np.pi
        #         y = z_0 - H_0.dot(x_11)

        y = z_0 - H_0.dot(x_11)
        for i in range(len(z_0) // 3):
            while y[2 + 3*i] > np.pi or y[2 + 3*i] < -np.pi:
                if y[2 + 3*i] > np.pi:
                    z_0[2 + 3*i] = z_0[2 + 3*i] - 2 * np.pi
                if y[2 + 3*i] < -np.pi:
                    z_0[2 + 3*i] = z_0[2 + 3*i] + 2 * np.pi
                y = z_0 - H_0.dot(x_11)

        P_11 = np.linalg.inv(Y_11)
        x_11 = np.linalg.inv(Y_11).dot(y_11)

        # Compute the information filter steps
        M_0 = np.transpose(np.linalg.inv(F_0)).dot(Y_11.dot(np.linalg.inv(F_0)))
        C_0 = M_0.dot(np.linalg.inv(M_0 + np.linalg.inv(Q_0)))
        L_0 = np.eye(np.shape(C_0)[0]) - C_0
        Y_01 = L_0.dot(M_0.dot(np.transpose(L_0))) + C_0.dot(np.linalg.inv(Q_0).dot(np.transpose(C_0)))
        y_01 = L_0.dot(np.transpose(np.linalg.inv(F_0)).dot(y_11))# + Y_01.dot(B_0.dot(u_0))
        Y_00 = Y_01 + np.transpose(H_0).dot(np.linalg.inv(R_0).dot(H_0))
        y_00 = y_01 + np.transpose(H_0).dot(np.linalg.inv(R_0).dot(z_0))
        # Don't use z, loop up extended information filter

        # # Compute the Kalman filter steps (For comparison and math checking)
        # x_01 = F_0.dot(x_11) #+ B_0.dot(u_0)
        # P_01 = F_0.dot(P_11.dot(np.transpose(F_0))) + Q_0
        # y = z_0 - H_0.dot(x_01)
        # S = H_0.dot(P_01.dot(np.transpose(H_0))) + R_0
        # K = P_01.dot(np.transpose(H_0).dot(np.linalg.inv(S)))
        # x_00 = x_01 + K.dot(y)
        # P_00 = (np.eye(np.shape(K)[0]) - K.dot(H_0).dot(P_01))
        #
        # # Compare information filter and kalman filter outputs
        # x_inf = np.linalg.inv(Y_00).dot(y_00)
        # #print('measurement: ' + str(z_0))
        # #print('state: ' + str(x_inf))
        # P_inf = np.linalg.inv(Y_00)
        # P_kal = F_0.dot(np.linalg.inv(Y_11).dot((np.transpose(F_0)))) + Q_0
        # P_inf = np.linalg.inv(Y_01)

        # Store the consensus variables
        # inf_Y = np.linalg.inv(P_01)
        # inf_y = np.linalg.inv(P_01).dot(x_01)
        inf_Y = Y_01
        inf_y = y_01
        inf_I = np.transpose(H_0).dot(np.linalg.inv(R_0).dot(H_0))
        inf_i = np.transpose(H_0).dot(np.linalg.inv(R_0).dot(z_0))
        inf_id_list = id_list
        inf_id_obs = observed_ids

        # Write the consensus variables to the publisher
        inf_partial = InfFilterPartials()
        inf_partial.sender_id = self.this_agent_id
        inf_partial.ids = inf_id_list
        inf_partial.obs_ids = inf_id_obs
        inf_partial.inf_matrix_prior = dse_lib.multi_array_2d_input(inf_Y, inf_partial.inf_matrix_prior)
        inf_partial.inf_vector_prior = dse_lib.multi_array_2d_input(inf_y, inf_partial.inf_vector_prior)
        inf_partial.obs_matrix = dse_lib.multi_array_2d_input(inf_I, inf_partial.obs_matrix)
        inf_partial.obs_vector = dse_lib.multi_array_2d_input(inf_i, inf_partial.obs_vector)
        self.inf_pub.publish(inf_partial)
        # print('new: ')
        # print(inf_y)
        # print()

        if self.pub_errors:
            print('errors')


def main(args):
    rospy.init_node('information_filter_node', anonymous=True)
    il = information_filter()   # This agent's ID is 1, and the state dimension is 6 (x, y, w, x_dot, y_dot, w_dot)
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")


if __name__ == '__main__':
    main(sys.argv)
