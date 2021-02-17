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
from cv_bridge import CvBridge, CvBridgeError
from scipy.spatial.transform import Rotation as R

import dse_lib
import dse_constants
roslib.load_manifest('dse_simulation')


def main(args):
    # 10 calculations per second
    rate = 10

    rospy.init_node('aruco_sim', anonymous=True)
    # Publisher for measurements
    pose_pub = rospy.Publisher("/dse/pose_markers", PoseMarkers, queue_size=10)
    # Publisher for true values
    true_pub = rospy.Publisher("/dse/python_pose_true", PoseMarkers, queue_size=10)

    # Define constants
    dt = 1.0 / rate
    robot_id = 1
    tag_id = 0
    n_agents = 2
    dim_state = 12
    dim_obs = 6
    euler_order = dse_constants.EULER_ORDER

    # Define state as zeros
    x = np.zeros((n_agents * dim_state, 1))

    # For testing,
    # # = [0, 1, 2, 3,     4,     5,     6,     7,     8,     9,         10,        11       ]
    # x = [x, y, z, z_ang, y_ang, x_ang, x_dot, y_dot, z_dot, z_ang_dot, y_ang_dot, x_ang_dot]
    # and then x = [x_agent_0, x_agent_1, .....]

    x[6] = 0.05         # 5 cm/sec forward velocity
    x[9] = 0.10       # 0.05 rad/sec (~3 degrees/sec) rotation
    k = 0

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
        true_pose.ids = [1, 0]
        true_pose.pose_array = dse_lib.pose_array_from_state(true_pose.pose_array, x, dim_state, dim_obs)
        true_pose.pose_array.header.stamp = rospy.Time.now()
        true_pose.pose_array.header.frame_id = 'dse'
        true_pub.publish(true_pose)

        # Compute the measurement
        agent1 = 1
        agent2 = 0
        agent1_row_min = dim_state * agent1
        agent1_row_max = agent1_row_min + dim_obs
        agent2_row_min = dim_state * agent2
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
        marker_pose.pose_array = dse_lib.pose_array_from_state(marker_pose.pose_array, z_true, dim_obs, dim_obs)
        marker_pose.pose_array.header.stamp = rospy.Time.now()
        marker_pose.pose_array.header.frame_id = 'dse'
        pose_pub.publish(marker_pose)

        # Pause for dt time then run the next step
        k = k + 1
        rospy.sleep(dt)


if __name__ == '__main__':
    main(sys.argv)
