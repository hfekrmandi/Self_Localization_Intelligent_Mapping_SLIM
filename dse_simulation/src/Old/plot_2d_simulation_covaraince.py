#!/usr/bin/env python2
from __future__ import print_function
import roslib
import sys
import rospy
import matplotlib.pyplot as plt
import numpy as np
import datetime
import time
from geometry_msgs.msg import PoseArray
from geometry_msgs.msg import PoseArray
from geometry_msgs.msg import Pose
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

import dse_lib
import dse_constants
roslib.load_manifest('dse_simulation')

# x vs. y plot
#   true x
#       tag[state[]]. extendable array for each tag
#   true y
#   x estimated by each agent
#       agent[tag[state[]]]. extendable array for each tag, for each agent
#   y estimated by each agent
#   estimated covariances

class plot_simulation_with_cov:

    # Define initial/setup values
    def __init__(self):

        # Get parameters from launch file
        self.ros_prefixes = rospy.get_param('~prefix')
        if len(self.ros_prefix) != 0 and self.ros_prefix[0] != '/':
            self.ros_prefix = '/' + self.ros_prefix
        self.this_agent_id = rospy.get_param('~id')
        self.dim_state = rospy.get_param('~dim_state')

        self.meas_vis_sub = rospy.Subscriber(self.ros_prefix + "/dse/vis/measurement", PoseArray, self.meas_vis_callback)
        self.est_vis_sub = rospy.Subscriber(self.ros_prefix + "/dse/vis/estimates", PoseArray, self.est_vis_callback)

        if self.dim_state == 6:
            self.dim_obs = 3
        elif self.dim_state == 12:
            self.dim_obs = 6
        else:
            rospy.signal_shutdown('invalid state dimension passed in')

        # Define static variables
        self.dt = 0.1
        self.t_last = rospy.get_time()
        self.gzbo_ref_obj_state = None
        self.pthn_ref_obj_state = None

        # Slowly animates the configuration and workspace plots for a device with:
        # two angles t1 and t2 in radians
        # x-y coordinates in cm
        self.time_permanent = []
        self.est_1_xyt_permanent = []
        self.est_1_covar_permanent = []
        self.est_2_xyt_permanent = []
        self.est_2_covar_permanent = []
        self.est_3_xyt_permanent = []
        self.est_3_covar_permanent = []

    # Create pose_array for measurement data
    def meas_vis_callback(self, data):
        poses = PoseArray()
        poses.poses = data.pose_array.poses
        poses.header.stamp = rospy.Time.now()
        if self.ros_prefix == '':
            poses.header.frame_id = 'base_footprint'
        else:
            poses.header.frame_id = self.ros_prefix + '/base_footprint'
        self.meas_vis_pub.publish(poses)

    # Create pose_array for the information results
    def pthn_true_callback(self, data):
        n = len(data.ids)
        for i in range(n):
            if data.ids[i] == dse_constants.GAZEBO_REFERENCE_OBJECT_ID:
                self.pthn_ref_obj_state = dse_lib.state_from_pose_3D(data.pose_array.poses[i])
                break

    # Create pose_array for the information results
    def results_callback(self, data):
        inf_id_list = np.array(data.ids)
        inf_Y = dse_lib.multi_array_2d_output(data.inf_matrix)
        inf_y = dse_lib.multi_array_2d_output(data.inf_vector)
        self.inf_x = np.linalg.inv(inf_Y).dot(inf_y)
        inf_P = np.linalg.inv(inf_Y)

        poses = PoseArray()
        poses.header.stamp = rospy.Time.now()
        if self.ros_prefix == '':
            poses.header.frame_id = 'base_footprint'
        else:
            poses.header.frame_id = self.ros_prefix + '/base_footprint'

        estimated_ids, estimated_states = dse_lib.relative_states_from_global_3D(self.this_agent_id, inf_id_list,
                                                                                 self.inf_x, self.dim_state, self.dim_obs)
        poses = dse_lib.pose_array_from_state(poses, estimated_states, self.dim_state, self.dim_obs)
        self.est_vis_pub.publish(poses)

        self.x_permanent.append(x)
        self.y_permanent.append(y)
        self.x_check_permanent.append(x_check)
        self.y_check_permanent.append(y_check)
        self.t1_permanent.append(theta_1)
        self.t2_permanent.append(theta_2)

        plt.subplot(211)
        plt.plot(t1_permanent, t2_permanent, '-', lw=2)
        plt.xlim(-2 * np.pi, 2 * np.pi)
        plt.ylim(-2 * np.pi, 2 * np.pi)
        plt.xlabel('theta 1 (rad)')
        plt.ylabel('theta 2 (rad)')
        plt.title('Configuration space')
        plt.grid(True)

        plt.show(block=False)
        plt.pause(0.001)


def main(args):

    rospy.init_node('plot_simulation_with_cov_node', anonymous=True)
    imf = plot_simulation_with_cov()
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")


if __name__ == '__main__':
    main(sys.argv)
