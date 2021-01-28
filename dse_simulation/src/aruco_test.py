#!/usr/bin/env python
from __future__ import print_function
import roslib
import sys
import rospy
import csv
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


class aruco_test:

    # Set up initial variables
    # Pass in the ID of this agent and the state dimension (6 or 12)
    def __init__(self):

        # # Get parameters from launch file
        # self.ros_prefix = rospy.get_param('~prefix')
        # if len(self.ros_prefix) != 0 and self.ros_prefix[0] != '/':
        #     self.ros_prefix = '/' + self.ros_prefix
        # self.camera_limits = rospy.get_param('~camera_limits')
        # # [dist_min, dist_max, horiz_fov, vert_fov]
        # self.tag_limit = rospy.get_param('~tag_limit')
        # # tag rotation at which the tag no longer detects
        # self.tag_size = rospy.get_param('~tag_size')
        # # height of the tag

        self.ros_prefix = '/tb3_0'
        self.camera_limits = [0.5, 1, 1.085595, 1.085595*480/640]
        self.tag_limit = 0.785398
        self.tag_size = 0.1*1.5

        # Define publishers and subscribers
        # Subscribe to the pose output from the camera
        self.pose_sub = rospy.Subscriber(self.ros_prefix + "/dse/pose_markers", PoseMarkers, self.camera_callback)
        self.num_results = 0
        self.results = []

    # When the direct estimator or consensus returns the combined information variables
    def camera_callback(self, data):
        self.num_results += 1
        result = np.zeros((10))
        result[0:6] = dse_lib.state_from_pose_array(data.pose_array, 12, 6)[0:6, 0]
        result[6:10] = dse_lib.eul2quat(result[3:6, None])
        self.results.append(result)

    def collect_data(self):
        if self.num_results > 0:
            tmp = np.array(self.results)
            [x, y, z, y, p, r, qx, qy, qz, qw] = np.split(np.array(self.results), 10, axis=1)
            print('Success')
            self.num_results = 0
            self.results = []
        print('data')


def main(args):
    with open('data.csv', mode="w") as data_file:
        data_writer = csv.writer(data_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        rospy.init_node('aruco_test_node', anonymous=True)
        il = aruco_test()
        il.csv_writer = data_writer
        r = rospy.Rate(1)
        il.dt = 1 / 1
        try:
            while True:
                r.sleep()
                il.collect_data()
        except KeyboardInterrupt:
            print("Shutting down")


if __name__ == '__main__':
    main(sys.argv)
