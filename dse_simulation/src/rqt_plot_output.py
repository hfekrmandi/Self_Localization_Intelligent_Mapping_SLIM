#!/usr/bin/env python3

"""
ROS Node for outputting Pose topics so that they can be displayed with rqt_plot
Currently only works for one agent and one tag, will have to figure out how to scale it
"""

from __future__ import print_function
import roslib
import sys
import rospy
import numpy as np
import datetime
import time
from geometry_msgs.msg import PoseArray
from geometry_msgs.msg import Pose
from dse_msgs.msg import PoseMarkers
from std_msgs.msg import Float64MultiArray
from std_msgs.msg import MultiArrayLayout
from std_msgs.msg import MultiArrayDimension
from dse_msgs.msg import InfFilterResults
from visualization_msgs.msg import Marker
from scipy.spatial.transform import Rotation as R

import dse_lib
import dse_constants

roslib.load_manifest('dse_simulation')

class information_filter:

    # Define initial/setup values
    def __init__(self):

        # Get parameters from launch file
        self.ros_prefix = rospy.get_param('~prefix', '')
        if len(self.ros_prefix) != 0 and self.ros_prefix[0] != '/':
            self.ros_prefix = '/' + self.ros_prefix
        self.this_agent_id = rospy.get_param('~id', 1)
        self.dim_state = rospy.get_param('~dim_state', 6)

        self.pose_sub = rospy.Subscriber(self.ros_prefix + "/dse/pose_markers", PoseMarkers, self.measurement_callback)
        self.true_sub = rospy.Subscriber(self.ros_prefix + "/dse/pose_true", PoseMarkers, self.true_callback)
        self.results_sub = rospy.Subscriber(self.ros_prefix + "/dse/inf/results", InfFilterResults, self.results_callback)
        self.meas_vis_pub = rospy.Publisher(self.ros_prefix + "/dse/plt/measurement", Pose, queue_size=10)
        self.true_robot_pub = rospy.Publisher(self.ros_prefix + "/dse/plt/true/robot", Pose, queue_size=10)
        self.true_tag_pub = rospy.Publisher(self.ros_prefix + "/dse/plt/true/tag", Pose, queue_size=10)
        self.est_robot_pub = rospy.Publisher(self.ros_prefix + "/dse/plt/estimates/robot", Pose, queue_size=10)
        self.est_tag_pub = rospy.Publisher(self.ros_prefix + "/dse/plt/estimates/tag", Pose, queue_size=10)


        if self.dim_state == 6:
            self.dim_obs = 3
        elif self.dim_state == 12:
            self.dim_obs = 6
        else:
            rospy.signal_shutdown('invalid state dimension passed in')

    # Publish the measurement pose
    def measurement_callback(self, data):
        pose = data.pose_array.poses[0]
        self.meas_vis_pub.publish(pose)

    # Publish the true poses
    def true_callback(self, data):
        for i in range(len(data.ids)):

            # I the ID is this agent's, publish that data under robot_pub. Otherwise, use tag_pub
            id = data.ids[i]
            if id == self.this_agent_id:
                self.true_robot_pub.publish(data.pose_array.poses[i])
            else:
                self.true_tag_pub.publish(data.pose_array.poses[i])

    # Publish the information estimation poses
    def results_callback(self, data):

        # Grab information values
        inf_id_list = np.array(data.ids)
        inf_Y = dse_lib.multi_array_2d_output(data.inf_matrix)
        inf_y = dse_lib.multi_array_2d_output(data.inf_vector)
        inf_x = np.linalg.inv(inf_Y).dot(inf_y)
        inf_P = np.linalg.inv(inf_Y)

        for i in range(len(inf_id_list)):
            pose = Pose()
            i_low = self.dim_state * i
            i_high = i_low + self.dim_obs

            # Grab position from x
            if self.dim_obs == 3:
                pose.position.x = inf_x[i_low]
                pose.position.y = inf_x[i_low + 1]
                pose.position.z = 0

                r = R.from_euler(dse_constants.EULER_ORDER_3D_OBS, inf_x[i_low + 2, 0])
                quat = r.as_quat()
            else:
                pose.position.x = inf_x[i_low]
                pose.position.y = inf_x[i_low + 1]
                pose.position.z = inf_x[i_low + 2]

                r = R.from_euler(dse_constants.EULER_ORDER, inf_x[i_low + 3:i_low + 6, 0])
                quat = r.as_quat()

            # Grab orientation quaternion
            pose.orientation.x = quat[0]
            pose.orientation.y = quat[1]
            pose.orientation.z = quat[2]
            pose.orientation.w = quat[3]

            # If the ID is this agent's, publish that data under robot_pub. Otherwise, use tag_pub
            if inf_id_list[i] == self.this_agent_id:
                self.est_robot_pub.publish(pose)
            else:
                self.est_tag_pub.publish(pose)


def main(args):
    rospy.init_node('dse_plotting_node', anonymous=True)
    il = information_filter()
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")


if __name__ == '__main__':
    main(sys.argv)
