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
from dse_msgs.msg import PoseMarkers
from std_msgs.msg import Float64MultiArray
from std_msgs.msg import MultiArrayLayout
from std_msgs.msg import MultiArrayDimension
from dse_msgs.msg import InfFilterResults
from visualization_msgs.msg import Marker
from scipy.spatial.transform import Rotation as R
import tf
from gazebo_msgs.msg import ModelStates
import tf_conversions
import tf2_ros

import dse_lib
import dse_constants
roslib.load_manifest('dse_simulation')


class visualizer:

    # Define initial/setup values
    def __init__(self):

        # Get parameters from launch file

        self.n_params = 4
        self.n_params = rospy.get_param('~n_params')
        # self.object_names = []
        # #self.object_pose_pubs = []
        # #self.object_name_pubs = []
        self.tf_broadcasters = []
        for i in range(self.n_params):
            self.tf_broadcasters.append(tf.TransformBroadcaster())
            self.object_names = rospy.get_param('~objects')
            #self.object_pose_pubs.append(rospy.Publisher("/gazebo_true/Pose/%s" % self.object_names[i], PoseArray, queue_size=10))
            #self.object_name_pubs.append(rospy.Publisher("/gazebo_true/Name/%s" % self.object_names[i], Marker, queue_size=10))
        #self.object_names = ['aruco_marker_0', 'aruco_marker_1', 'aruco_marker_2', 'aruco_marker_3']
        self.object_names = np.array(self.object_names)
        self.link_states_sub = rospy.Subscriber('/gazebo/model_states', ModelStates, self.gzbo_true_callback)

    # Create pose_array for the information results
    def gzbo_true_callback(self, data):
        n = len(data.name)
        for i in range(n):
            if data.name[i] in self.object_names:
                index = np.where(self.object_names == data.name[i])[0][0]
                position = (data.pose[i].position.x, data.pose[i].position.y, data.pose[i].position.z)
                orientation = (data.pose[i].orientation.x, data.pose[i].orientation.y, data.pose[i].orientation.z, data.pose[i].orientation.w)
                self.tf_broadcasters[index].sendTransform(position, orientation, rospy.Time.now(), data.name[i], 'world')


def main(args):

    rospy.init_node('dse_gazebo_object_visualization_node', anonymous=True)
    vis = visualizer()
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")


if __name__ == '__main__':
    main(sys.argv)
