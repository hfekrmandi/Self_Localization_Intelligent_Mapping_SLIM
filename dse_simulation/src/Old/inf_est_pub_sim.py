#!/usr/bin/env python3
from __future__ import print_function
import os
import roslib
import sys
import rospy
import numpy as np
import cv2
from cv2 import aruco
import pickle
import datetime
import time
from geometry_msgs.msg import Pose
from dse_msgs.msg import PoseMarkers
from dse_msgs.msg import InfFilterResults
from cv_bridge import CvBridge, CvBridgeError
from scipy.spatial.transform import Rotation as R

import dse_lib
import dse_constants
roslib.load_manifest('dse_simulation')


def main(args):
    # 10 calculations per second
    rate = 10

    rospy.init_node('inf_est_pub_sim_node', anonymous=True)
    # Publisher for measurements
    pose_pub = rospy.Publisher("/dse/pose_markers", PoseMarkers, queue_size=10)
    # Publisher for true values
    true_pub = rospy.Publisher("/dse/python_pose_true", PoseMarkers, queue_size=10)
    # Publisher for true values
    inf_pub = rospy.Publisher("/dse/inf/results", InfFilterResults, queue_size=10)

    # Define constants
    dt = 1.0 / rate
    robot_id = 1
    tag_id = 0
    n_agents = 2
    dim_state = 12
    dim_obs = 6
    inf_dim_state = 6
    inf_dim_obs = 3
    euler_order = dse_constants.EULER_ORDER

    # Define state as zeros
    x = np.zeros((n_agents * dim_state, 1))
    true_id_list = [1, 0]

    # For testing,
    # # = [0, 1, 2, 3,     4,     5,     6,     7,     8,     9,         10,        11       ]
    # x = [x, y, z, z_ang, y_ang, x_ang, x_dot, y_dot, z_dot, z_ang_dot, y_ang_dot, x_ang_dot]
    # and then x = [x_agent_0, x_agent_1, .....]

    x[6] = 0.05         # 5 cm/sec forward velocity
    x[9] = 0.10       # 0.05 rad/sec (~3 degrees/sec) rotation
    k = 0

    # Define static variables
    this_agent_id = 1
    dt = 0.1
    euler_order = dse_constants.EULER_ORDER

    # Define information variables
    inf_P = []
    inf_x = []
    inf_I = []
    inf_i = []
    inf_id_obs = []
    inf_id_comm = []

    # Initialize information variables
    id_list = [this_agent_id]
    inf_Y = dse_constants.INF_MATRIX_INITIAL * np.eye(dim_state, dtype=np.float64)
    inf_y = dse_constants.INF_VECTOR_INITIAL * \
                 np.transpose(1 * np.arange(1, dim_state + 1, dtype=np.float64))[:, None]

    # Loop forever
    while True:
        # Process noise (w)
        w = 0.00 * np.random.rand(n_agents * dim_state, 1)
        # Measurement noise (v)
        v = 0.00 * np.random.rand(dim_obs, 1)

        # Define motion model
        F = np.eye(n_agents * dim_state)
        F[0:dim_state, 0:dim_state] = dse_lib.f_unicycle(dt, x, 0, dim_state)

        # Upodate state
        x = F.dot(x) + w

        # Publish the true pose for each agent
        true_pose = PoseMarkers()
        true_pose.ids = true_id_list
        true_pose.pose_array = dse_lib.pose_array_from_state(true_pose.pose_array, x, dim_state, dim_obs)
        true_pose.pose_array.header.stamp = rospy.Time.now()
        true_pose.pose_array.header.frame_id = 'dse'
        true_pub.publish(true_pose)

        # Compute the measurement
        agent1_row_min = dim_state * 0
        agent1_row_max = agent1_row_min + dim_obs
        agent2_row_min = dim_state * 1
        agent2_row_max = agent2_row_min + dim_obs

        x1 = x[agent1_row_min:agent1_row_max]
        t1 = x1[0:3]
        r1 = R.from_euler(euler_order, x1[3:6, 0])
        R1 = r1.as_dcm()

        x2 = x[agent2_row_min:agent2_row_max]
        t2 = x2[0:3]
        r2 = R.from_euler(euler_order, x2[3:6, 0])
        R2 = r2.as_dcm()

        tz = (np.transpose(R1).dot(t2) - np.transpose(R1).dot(t1))[:, 0]
        Rz = np.transpose(R1).dot(R2)
        rz = R.from_dcm(Rz)
        rz = rz.as_euler(euler_order)
        z_true = np.concatenate((tz, rz))[:, None] + v

        # Publish the measurement
        marker_pose = PoseMarkers()
        marker_pose.ids = [0]
        marker_pose.pose_array = dse_lib.pose_array_from_measurement(marker_pose.pose_array, z_true, dim_obs)
        marker_pose.pose_array.header.stamp = rospy.Time.now()
        marker_pose.pose_array.header.frame_id = 'dse'
        pose_pub.publish(marker_pose)

        # Grab the tag poses from the camera
        observed_poses = marker_pose.pose_array.poses
        observed_ids = marker_pose.ids
        n = 1 + len(observed_ids)

        # update local values from the last time step
        Y_11 = inf_Y  # Information matrix - Covariance
        y_11 = inf_y  # Information vector - States
        x_11 = np.linalg.inv(Y_11).dot(y_11)  # Kalman State
        P_11 = np.linalg.inv(Y_11)  # Kalman Covariance

        # If we find an ID that isn't currently known, add it
        id_list, Y_11, y_11, P_11, x_11 = dse_lib.extend_arrays(observed_ids, id_list, Y_11, y_11, inf_dim_state)

        # Fill in R, H, z, F, and Q
        # R - Measurement Covariance
        # H - Measurement Jacobian
        # z - The measurement itself
        # This function is defined in src/dse_lib.py
        R_0, H_0, z_0 = dse_lib.fill_RHz(id_list, this_agent_id, observed_ids, observed_poses, x_11,
                                         euler_order, inf_dim_state, inf_dim_obs)

        # F - Motion Jacobian
        # Q - Motion Covariance
        F_0, Q_0 = dse_lib.fill_FQ(id_list, dt, x_11, inf_dim_state, inf_dim_obs)

        y = z_0 - H_0.dot(x_11)
        while y[2] > np.pi or y[2] < -np.pi:
            if y[2] > np.pi:
                z_0[2] = z_0[2] - 2 * np.pi
            if y[2] < -np.pi:
                z_0[2] = z_0[2] + 2 * np.pi
            y = z_0 - H_0.dot(x_11)

        # Compute the information filter steps
        M_0 = np.transpose(np.linalg.inv(F_0)).dot(Y_11.dot(np.linalg.inv(F_0)))
        C_0 = M_0.dot(np.linalg.inv(M_0 + np.linalg.inv(Q_0)))
        L_0 = np.eye(np.shape(C_0)[0]) - C_0
        Y_01 = L_0.dot(M_0.dot(np.transpose(L_0))) + C_0.dot(np.linalg.inv(Q_0).dot(np.transpose(C_0)))
        y_01 = L_0.dot(np.transpose(np.linalg.inv(F_0)).dot(y_11))  # + Y_01.dot(B_0.dot(u_0))
        Y_00 = Y_01 + np.transpose(H_0).dot(np.linalg.inv(R_0).dot(H_0))
        y_00 = y_01 + np.transpose(H_0).dot(np.linalg.inv(R_0).dot(z_0))
        # Don't use z, loop up extended information filter

        # Compute the Kalman filter steps (For comparison and math checking)
        x_01 = F_0.dot(x_11)  # + B_0.dot(u_0)
        P_01 = F_0.dot(P_11.dot(np.transpose(F_0))) + Q_0
        y = z_0 - H_0.dot(x_01)
        S = H_0.dot(P_01.dot(np.transpose(H_0))) + R_0
        K = P_01.dot(np.transpose(H_0).dot(np.linalg.inv(S)))
        x_00 = x_01 + K.dot(y)
        P_00 = (np.eye(np.shape(K)[0]) - K.dot(H_0)).dot(P_01)

        kal_x_01 = x_01
        kal_x_00 = x_00
        kal_P_01 = P_01
        kal_P_00 = P_00

        inf_x_01 = np.linalg.inv(Y_01).dot(y_01)
        inf_x_00 = np.linalg.inv(Y_00).dot(y_00)
        inf_P_01 = np.linalg.inv(Y_01)
        inf_P_00 = np.linalg.inv(Y_00)

        kal_x_01_eq_inf_x_01 = np.allclose(kal_x_01, inf_x_01)
        kal_x_00_eq_inf_x_01 = np.allclose(kal_x_00, inf_x_00)
        kal_P_01_eq_inf_P_01 = np.allclose(kal_P_01, inf_P_01)
        kal_P_00_eq_inf_P_00 = np.allclose(kal_P_00, inf_P_00)

        inf_Y = np.linalg.inv(kal_P_00)
        inf_y = inf_Y.dot(kal_x_00)

        # Write the consensus variables to the publisher
        inf_results = InfFilterResults()
        inf_results.ids = id_list
        inf_results.inf_matrix = dse_lib.multi_array_2d_input(inf_Y, inf_results.inf_matrix)
        inf_results.inf_vector = dse_lib.multi_array_2d_input(inf_y, inf_results.inf_vector)
        inf_pub.publish(inf_results)

        # Pause for dt time then run the next step
        k = k + 1
        rospy.sleep(dt)


if __name__ == '__main__':
    main(sys.argv)
