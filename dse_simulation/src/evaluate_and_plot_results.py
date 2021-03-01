#!/usr/bin/env python3
from __future__ import print_function
import roslib
import sys
import rospy
import numpy as np
import datetime
import time
from geometry_msgs.msg import PoseArray
from geometry_msgs.msg import Pose
from geometry_msgs.msg import PoseWithCovariance
from nav_msgs.msg import Odometry
from dse_msgs.msg import PoseMarkers
from std_msgs.msg import Float64MultiArray
from std_msgs.msg import MultiArrayLayout
from std_msgs.msg import MultiArrayDimension
from dse_msgs.msg import InfFilterResults
from visualization_msgs.msg import Marker
from scipy.spatial.transform import Rotation as R
from gazebo_msgs.msg import LinkStates
import tf_conversions
import tf2_ros
import matplotlib.pyplot as plt

import dse_lib
import dse_constants
roslib.load_manifest('dse_simulation')


class information_filter:

    # Define initial/setup values
    def __init__(self):

        # # Get parameters from launch file
        # self.ros_prefix = rospy.get_param('~prefix')
        # if len(self.ros_prefix) != 0 and self.ros_prefix[0] != '/':
        #     self.ros_prefix = '/' + self.ros_prefix
        # self.tf_pretix = self.ros_prefix[1:]
        # self.dim_state = rospy.get_param('~dim_state')

        self.ros_prefix = '/tb3_0'
        self.tf_prefix = self.ros_prefix[1:]
        self.this_agent_id = 5
        self.dim_state = 6

        self.inf_results_sub = rospy.Subscriber(self.ros_prefix + "/dse/inf/results", InfFilterResults, self.results_callback)

        if self.dim_state == 6:
            self.dim_obs = 3
        elif self.dim_state == 12:
            self.dim_obs = 6
        else:
            rospy.signal_shutdown('invalid state dimension passed in')

        # Slowly animates the configuration and workspace plots for a device with:
        # two angles t1 and t2 in radians
        # x-y coordinates in cm
        self.time_permanent = []
        self.x_permanent = []
        self.y_permanent = []
        self.est_1_xyt_permanent = []
        self.est_1_covar_permanent = []
        self.est_2_xyt_permanent = []
        self.est_2_covar_permanent = []
        self.est_3_xyt_permanent = []
        self.est_3_covar_permanent = []

    # Create pose_array for the information results
    def results_callback(self, data):
        inf_id_list = np.array(data.ids)
        inf_Y = dse_lib.multi_array_2d_output(data.inf_matrix)
        inf_y = dse_lib.multi_array_2d_output(data.inf_vector)
        self.inf_x = np.linalg.inv(inf_Y).dot(inf_y)
        inf_P = np.linalg.inv(inf_Y)


        for id in inf_id_list:
            i = np.where(inf_id_list == id)[0][0]

            i_min = i * self.dim_state
            i_max = i_min + self.dim_state
            self.x_permanent.append(self.inf_x[i_min])
            self.y_permanent.append(self.inf_x[i_min+1])
            if len(self.x_permanent) > 1000:
                self.x_permanent = self.x_permanent[-1000:]
                self.y_permanent = self.y_permanent[-1000:]


        # self.time_permanent.append(rospy.Time.now())
        # i_min = np.where(inf_id_list == 5)[0][0] * self.dim_state
        # self.est_1_xyt_permanent.append(self.inf_x[i_min:i_min+3])
        # #self.est_1_covar_permanent.append()
        # i_min = np.where(inf_id_list == 6)[0][0] * self.dim_state
        # self.est_2_xyt_permanent.append(self.inf_x[i_min:i_min+3])
        # #self.est_2_covar_permanent.append()
        # i_min = np.where(inf_id_list == 7)[0][0] * self.dim_state
        # self.est_3_xyt_permanent.append(self.inf_x[i_min:i_min+3])
        # #self.est_3_covar_permanent.append()

        plt.clf()
        plt.plot(self.x_permanent, self.y_permanent, 'r.', lw=2)
        # plt.plot(self.est_1_xyt_permanent[:][0], self.est_1_xyt_permanent[:][1], 'r-', lw=2)
        # plt.plot(self.est_2_xyt_permanent[:][0], self.est_2_xyt_permanent[:][1], 'g-', lw=2)
        # plt.plot(self.est_3_xyt_permanent[:][0], self.est_3_xyt_permanent[:][1], 'b-', lw=2)
        plt.xlabel('x (m)')
        plt.ylabel('y (m)')
        plt.title('agent 1s trajectory estimates')
        plt.grid(True)

        plt.show(block=False)
        plt.pause(0.001)


def main(args):

    rospy.init_node('dse_plotting_node', anonymous=True)
    imf = information_filter()
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")


if __name__ == '__main__':
    main(sys.argv)
