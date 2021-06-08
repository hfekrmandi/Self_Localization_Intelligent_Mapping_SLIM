#!/usr/bin/env python2
from __future__ import print_function
import os
import roslib
import sys
import rospy
import numpy as np
import cv2
import tf2_ros
import datetime
import time
from geometry_msgs.msg import PoseStamped
from geometry_msgs.msg import PoseWithCovarianceStamped
import tf2_geometry_msgs
from dse_msgs.msg import PoseMarkers
from scipy.spatial.transform import Rotation as R
import dse_lib
import gazebo_lib
import copy

roslib.load_manifest('dse_simulation')


class fake_pose_sensor:

    def __init__(self):

        # # Get parameters from launch file
        # self.ros_prefix = 'tb3_0'
        # if len(self.ros_prefix) != 0 and self.ros_prefix[0] != '/':
        #     self.ros_prefix = '/' + self.ros_prefix
        # # side length of tag in meters
        # self.markerLength = 0.1
        # self.cal_file = 'calibrationSave_gazebo.p'

        # Get parameters from launch file
        self.ros_prefix = rospy.get_param('~prefix', '')
        if len(self.ros_prefix) != 0 and self.ros_prefix[0] != '/':
            self.ros_prefix = '/' + self.ros_prefix
        self.r = rospy.Rate(rospy.get_param('~rate', 10))
        # max measurement distance in meters
        self.meas_threshold = rospy.get_param('~meas_threshold', 5)
        self.this_agent_id = rospy.get_param('~id')
        id_to_tf_arr = rospy.get_param('~id_to_tf')
        fixed_relations_arr = rospy.get_param('~fixed_relations', [])

        # # Get parameters from launch file
        # self.ros_prefix = 'tb3_0'
        # if len(self.ros_prefix) != 0 and self.ros_prefix[0] != '/':
        #     self.ros_prefix = '/' + self.ros_prefix
        # # side length of tag in meters
        # self.r = rospy.Rate(10)
        # # max measurement distance in meters
        # self.meas_threshold = 10.0
        # self.this_agent_id = 5
        # id_to_tf_arr = [[0, 'aruco_marker_0'], [1, 'aruco_marker_1'], [2, 'aruco_marker_2'], [3, 'aruco_marker_3'], [5, 'tb3_0/base_footprint'], [6, 'tb3_1/base_footprint']]
        # fixed_relations_arr = [[0, 'world', [0, 1, 2, 3]]]

        self.id_to_tf = {}
        for val in id_to_tf_arr:
            self.id_to_tf[val[0]] = val[1]

        # self.id_to_tf = {
        #     0:'aruco_marker_0',
        #     1:'aruco_marker_1',
        #     2:'aruco_marker_2',
        #     3:'aruco_marker_3',
        #     5:'tb3_0/base_footprint',
        #     6:'tb3_1/base_footprint',
        #     7:'tb3_2/base_footprint'
        # }

        self.fixed_relations = {}
        for val in fixed_relations_arr:
            for id in val[2]:
                self.fixed_relations[id] = [val[0], val[1]]

        # self.fixed_relations = {
        #     0:[0, 'world'],
        #     1:[0, 'world'],
        #     2:[0, 'world'],
        #     3:[0, 'world']
        # }

        self.tfBuffer = tf2_ros.Buffer()
        self.listener = tf2_ros.TransformListener(self.tfBuffer)
        time.sleep(5)
        self.pose_pub = rospy.Publisher(self.ros_prefix + "/dse/pose_markers", PoseMarkers, queue_size=1)

    def estimate_and_send(self):

        sim_poses = PoseMarkers()
        sim_poses.ids = []
        from_tf = self.id_to_tf[self.this_agent_id]
        # Get object poses
        # For each object within measurement criterion
        #   Add estimate to array of poses

        # robot in world frame = robot in tag * tag in world

        for id in self.id_to_tf.keys():
            # It's not useful to observe out state relative to ourselves
            if id == self.this_agent_id:
                continue

            to_tf = self.id_to_tf[id]
            transform = self.tfBuffer.lookup_transform(from_tf, to_tf, rospy.Time(0))
            true_position = transform.transform.translation
            true_xyz = np.array([true_position.x, true_position.y, true_position.z])
            dist = np.linalg.norm(true_xyz)

            # skip if not measurable
            if dist > self.meas_threshold:
                continue

            tag_tf = self.id_to_tf[id]
            robot_tf = self.id_to_tf[self.this_agent_id]

            if id in self.fixed_relations:
                robot_in_tag_xfm = self.tfBuffer.lookup_transform(
                    tag_tf, robot_tf, rospy.Time(0))

                world_tf = self.fixed_relations[id][1]

                tag_in_world_xfm = self.tfBuffer.lookup_transform(
                    world_tf, tag_tf, rospy.Time(0))

                robot_in_tag_xfm_noisy, covariance = gazebo_lib.noisy_transform(robot_in_tag_xfm)
                robot_in_tag_pose = gazebo_lib.transform_to_pose_stamped(robot_in_tag_xfm_noisy)
                robot_in_world_pose = tf2_geometry_msgs.do_transform_pose(robot_in_tag_pose, tag_in_world_xfm)

                pose_stamped = gazebo_lib.pose_stamped_to_pose_stamped_covariance(robot_in_world_pose, covariance)
                sim_poses.ids.append(self.fixed_relations[id][0])
            else:
                tag_in_robot_xfm = self.tfBuffer.lookup_transform(
                    robot_tf, tag_tf, rospy.Time(0))

                tag_in_robot_xfm_noisy, covariance = gazebo_lib.noisy_transform(tag_in_robot_xfm)
                pose_stamped = gazebo_lib.transform_to_pose_stamped_covariance(tag_in_robot_xfm, covariance)
                sim_poses.ids.append(id)

            sim_poses.pose_array.append(pose_stamped)

        self.pose_pub.publish(sim_poses)
        return


def main(args):
    rospy.init_node('dse_consensus_node', anonymous=True)
    sens = fake_pose_sensor()
    try:
        while True:
            sens.r.sleep()
            sens.estimate_and_send()
    except KeyboardInterrupt:
        print("Shutting down")


if __name__ == '__main__':
    main(sys.argv)
