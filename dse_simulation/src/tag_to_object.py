#!/usr/bin/env python3
from __future__ import print_function
import roslib
import sys
import rospy
import numpy as np
import datetime
import time
from geometry_msgs.msg import Twist
from geometry_msgs.msg import Pose
from dse_msgs.msg import PoseMarkers
from std_msgs.msg import Float64MultiArray
from std_msgs.msg import MultiArrayLayout
from std_msgs.msg import MultiArrayDimension
from dse_msgs.msg import InfFilterPartials
from dse_msgs.msg import InfFilterResults
from scipy.spatial.transform import Rotation as R
import tf

import dse_lib
import dse_constants
roslib.load_manifest('dse_simulation')


class measurement_error:

    # Set up initial variables
    # Pass in the ID of this agent and the state dimension (6 or 12)
    def __init__(self):

        # Get parameters from launch file
        self.ros_prefix = rospy.get_param('~prefix')
        if len(self.ros_prefix) != 0 and self.ros_prefix[0] != '/':
            self.ros_prefix = '/' + self.ros_prefix
        self.this_agent_id = rospy.get_param('~id')

        # A tag cube:
        # 6 tags, ids [cube id * 6], plus the next 5 consecutive ids
        #   ex: cube id 3 will be 18-23
        # All 0.1m tags (0.138m with border?)

        # links for each tag cube
        # tag_cube_link
        #   bottom_tag_link
        #   front_tag_link
        #   right_tag_link
        #   back_tag_link
        #   left_tag_link
        #   top_tag_link
        # where these links are a perfect match to what the ARUCO library gets as a measurement
        # ie a measurement from another agent with 0 error will perfectly match up with a TF transformation
        #   between the measuring '/camera_rgb_frame' and the measured '/front_tag_link'

        # Simulation flow
        # agents make camera measurements, send out tag ids + relative pose
        # this node takes in those measurements and converts them to relative measurement of the agent/object's base_link
        #   The id of the measured object is also changed to the simulation id (not the tag id)
        # These measurements are used in the information filter
        #   measurements of a stationary object are reversed, to be a measurement of our position in the global coordinates


        # This file will be passed:
        # A CSV file with:
        # all agents
        #   simulation ID
        #   initial pose?????
        #   tag ids to location (bottom, right...)
        # all objects
        #   simulation ID
        #   all tag cubes
        #       tag ids to location (bottom, right...)

        self.object_list = {

        }

        self.id_to_tf = {
            0:'/aruco_marker_0',
            1:'/aruco_marker_1',
            2:'/aruco_marker_2',
            3:'/aruco_marker_3',
            5:'/tb3_0',
            6:'/tb3_1',
            7:'/tb3_2',
        }

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
        # Subscribe to the pose output from the camera
        self.pose_sub = rospy.Subscriber(self.ros_prefix + "/dse/pose_markers", PoseMarkers, self.measurement_callback)

        self.camera_true_pub = rospy.Publisher(self.ros_prefix + "/diag/camera_true", PoseMarkers, queue_size=1)
        self.camera_error_pub = rospy.Publisher(self.ros_prefix + "/diag/camera_error", PoseMarkers, queue_size=1)

    # When the camera sends a measurement
    def measurement_callback(self, data):
        listener = tf.TransformListener()

        # Grab the tag poses from the camera
        observed_poses = data.pose_array.poses
        observed_ids = data.ids
        n = 1 + len(observed_ids)

        true_poses = PoseMarkers()
        true_poses.ids = observed_ids
        from_tf = self.id_to_tf[self.this_agent_id] + '/camera_rgb_frame'
        true_poses.pose_array.header.stamp = rospy.Time.now()
        true_poses.pose_array.header.frame_id = self.id_to_tf[self.this_agent_id] + '/camera_rgb_frame'
        for id in observed_ids:
            to_tf = self.id_to_tf[id] + ''
            listener.waitForTransform(from_tf, to_tf, rospy.Time(), rospy.Duration(5))
            (trans, quat) = listener.lookupTransform(from_tf, to_tf, rospy.Time(0))

            pose = Pose()
            pose.position.x = trans[0]
            pose.position.y = trans[0]
            pose.position.z = trans[0]
            pose.orientation.x = quat[0]
            pose.orientation.y = quat[1]
            pose.orientation.z = quat[2]
            pose.orientation.w = quat[3]
            true_poses.pose_array.poses += [pose]
        self.camera_true_pub.publish(true_poses)

        error_poses = PoseMarkers()
        error_poses.ids = observed_ids
        error_poses.pose_array.header.stamp = rospy.Time.now()
        error_poses.pose_array.header.frame_id = self.id_to_tf[self.this_agent_id] + '/camera_rgb_frame'
        for i in range(len(observed_ids)):
            meas_pose = data.pose_array.poses[i]
            meas_eul = dse_lib.quat_from_pose2eul(meas_pose.orientation)
            r = R.from_euler(dse_constants.EULER_ORDER, meas_eul)
            meas_rotm = r.as_matrix()
            true_pose = true_poses.pose_array.poses[i]
            true_eul = dse_lib.quat_from_pose2eul(true_pose.orientation)
            r = R.from_euler(dse_constants.EULER_ORDER, true_eul)
            true_rotm = r.as_matrix()
            error_rotm = meas_rotm.dot(true_rotm.T)
            r = R.from_matrix(error_rotm)
            error_quat = r.as_quat()

            pose = Pose()
            pose.position.x = true_pose.position.x - meas_pose.position.x
            pose.position.y = true_pose.position.y - meas_pose.position.y
            pose.position.z = true_pose.position.z - meas_pose.position.z

            pose.orientation.x = error_quat[0]
            pose.orientation.y = error_quat[1]
            pose.orientation.z = error_quat[2]
            pose.orientation.w = error_quat[3]
            error_poses.pose_array.poses += [pose]
        self.camera_true_pub.publish(error_poses)


def main(args):
    rospy.init_node('measurement_error_node', anonymous=True)
    me = measurement_error()   # This agent's ID is 1, and the state dimension is 6 (x, y, w, x_dot, y_dot, w_dot)
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")


if __name__ == '__main__':
    main(sys.argv)
