#!/usr/bin/env python3
from __future__ import print_function
import roslib
import sys
import rospy
import numpy as np
import datetime
import time
from geometry_msgs.msg import Twist
from tf2_geometry_msgs import PoseStamped
from geometry_msgs.msg import Pose
from std_msgs.msg import Bool
from dse_msgs.msg import PoseMarkers
from std_msgs.msg import Float64MultiArray
from std_msgs.msg import MultiArrayLayout
from std_msgs.msg import MultiArrayDimension
from dse_msgs.msg import InfFilterPartials
from dse_msgs.msg import InfFilterResults
from scipy.spatial.transform import Rotation as R
import tf2_ros
import csv
import matplotlib.pyplot as plt

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
        self.init_ids = rospy.get_param('~initial_ids', [])

        self.id_to_tf = {
            0:'aruco_marker_0',
            1:'aruco_marker_1',
            2:'aruco_marker_2',
            3:'aruco_marker_3',
            5:'tb3_0',
            6:'tb3_1',
            7:'tb3_2',
        }

        # self.data_file = open('/home/alex/error_data.csv', mode='w')
        # self.csv_writer = csv.writer(self.data_file, delimiter=',')
        # csv_header = []
        # for i in self.init_ids:
        #     for j in ['true_dist', 'dist error', 'angle error']:
        #         csv_header.append(str(self.id_to_tf[i][1:] + ' ' + j))
        # self.csv_writer.writerow(csv_header)
        # self.csv_writer.writerow(csv_header)
        #
        # self.all_data_file = open('/home/alex/all_data.csv', mode='w')
        # self.all_csv_writer = csv.writer(self.all_data_file, delimiter=',')
        # csv_header = []
        # for k in ['est', 'true', 'error']:
        #     for i in self.init_ids:
        #         for j in ['x', 'y', 'z', 'y', 'p', 'r']:
        #             csv_header.append(str(self.id_to_tf[i][1:] + ' ' + j + '' + k))
        # self.all_csv_writer.writerow(csv_header)
        # self.all_csv_writer.writerow(csv_header)

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
        # self.store_data_sub = rospy.Subscriber('/plot_measurements', Bool, self.plot_measurements)

        self.camera_sim_pub = rospy.Publisher(self.ros_prefix + "/dse/pose_simulated", PoseMarkers, queue_size=1)
        self.camera_true_pub = rospy.Publisher(self.ros_prefix + "/diag/camera_true", PoseMarkers, queue_size=1)
        self.camera_error_pub = rospy.Publisher(self.ros_prefix + "/diag/camera_error", PoseMarkers, queue_size=1)
        self.est_pose_data = []
        self.true_pose_data = []
        self.error_pose_data = []
        self.tfBuffer = tf2_ros.Buffer()
        self.listener = tf2_ros.TransformListener(self.tfBuffer)

    def order_poses(self, these_ids, poses):
        all_poses = []
        nothing = np.array([-1, -1, -1, -1, -1, -1])
        for id in self.init_ids:
            if id in these_ids:
                index = these_ids.index(id)
                all_poses.append(poses[index])
            else:
                all_poses.append(nothing[:, None])
        return all_poses

    # def send_poses_csv(self, error_poses, true_poses):
    #     poses_np = np.array(error_poses)
    #     poses_flat = poses_np.flatten()
    #     true_poses_np = np.array(true_poses)
    #     true_poses_flat = true_poses_np.flatten()
    #     comp_errors = []
    #     for i in range(len(self.init_ids)):
    #         i_min = i*6
    #         if poses_flat[i_min] == -1:
    #             true_dist = -1
    #             dist = -1
    #             angle_sqr = -1
    #         else:
    #             true_dist = np.sqrt(true_poses_flat[i_min]**2 + true_poses_flat[i_min+1]**2 + true_poses_flat[i_min+2]**2)
    #             dist = np.sqrt(poses_flat[i_min]**2 + poses_flat[i_min+1]**2 + poses_flat[i_min+2]**2)
    #             if poses_flat[i_min+5] < 0:
    #                 poses_flat[i_min + 5] = poses_flat[i_min + 5] + np.pi
    #             else:
    #                 poses_flat[i_min + 5] = poses_flat[i_min + 5] - np.pi
    #             angle_sqr = np.sqrt(poses_flat[i_min+3]**2 + poses_flat[i_min+4]**2 + poses_flat[i_min+5]**2)
    #         comp_errors.append(true_dist)
    #         comp_errors.append(dist)
    #         comp_errors.append(angle_sqr)
    #     self.csv_writer.writerow(comp_errors)
    #
    # def send_all_poses_csv(self, error_poses, true_poses, est_poses):
    #     poses_np = np.array(error_poses)
    #     poses_flat = poses_np.flatten()
    #     true_poses_np = np.array(true_poses)
    #     true_poses_flat = true_poses_np.flatten()
    #     est_poses_np = np.array(est_poses)
    #     est_poses_flat = est_poses_np.flatten()
    #
    #     all = np.concatenate((est_poses_flat, true_poses_flat, poses_flat))
    #     self.all_csv_writer.writerow(all)

    # When the camera sends a measurement
    def measurement_callback(self, data):
        # Grab the tag poses from the camera
        observed_poses = []
        for pose_stamped in data.pose_array:
            observed_poses.append(pose_stamped.pose)

        observed_ids = data.ids
        est_xyzypr = []
        for pose in observed_poses:
            est_xyzypr.append(dse_lib.state_from_pose(pose))
        self.est_pose_data.append(self.order_poses(observed_ids, est_xyzypr))
        n = 1 + len(observed_ids)

        sim_poses = PoseMarkers()
        sim_poses.ids = observed_ids
        from_tf = self.id_to_tf[self.this_agent_id] + '/base_link'

        # true_poses = PoseMarkers()
        # true_poses.ids = observed_ids
        # from_tf = self.id_to_tf[self.this_agent_id] + '/camera_rgb_frame'
        # true_poses.pose_array.header.stamp = rospy.Time.now()
        # true_poses.pose_array.header.frame_id = self.id_to_tf[self.this_agent_id] + '/camera_rgb_frame'
        # true_xyzypr = []
        for id in observed_ids:
            # If we're observing a turtlebot, we need the aruco_tag_link frame
            if self.id_to_tf[id][0:3] == 'tb3':
                to_tf = self.id_to_tf[id] + '/aruco_tag_link'
            else:
                to_tf = self.id_to_tf[id]

            transform = self.tfBuffer.lookup_transform(from_tf, to_tf, rospy.Time(0))

            pose_stamped = PoseStamped()
            pose_stamped.header.stamp = rospy.Time.now()
            pose_stamped.header.frame_id = self.id_to_tf[self.this_agent_id] + '/base_link'
            pose = pose_stamped.pose
            pose.position = transform.transform.translation
            pose.orientation = transform.transform.rotation
            # true_poses.pose_array.append(pose_stamped)

            sim_pose_stamped = PoseStamped()
            sim_pose_stamped.header.stamp = rospy.Time.now()
            sim_pose_stamped.header.frame_id = self.id_to_tf[self.this_agent_id] + '/base_link'
            sim_pose = sim_pose_stamped.pose
            true_eul = dse_lib.quat_from_pose2eul(pose.orientation)
            trans = [pose.position.x, pose.position.y, pose.position.z]
            true_state = np.concatenate((trans, true_eul))
            noise = np.random.multivariate_normal([0, 0, 0, 0, 0, 0], dse_lib.gazebo_R_from_range(np.linalg.norm(trans)))
            sim_state = true_state + noise
            sim_pose.orientation = dse_lib.euler2quat_from_pose(sim_pose.orientation, sim_state[3:6, None])
            sim_pose.position.x = sim_state[0]
            sim_pose.position.y = sim_state[1]
            sim_pose.position.z = sim_state[2]
            sim_poses.pose_array.append(pose_stamped)

            # eul = dse_lib.quat2eul(pose.orientation)
            # true_xyzypr.append([trans[0], trans[1], trans[2], eul[0], eul[1], eul[2]])
            # true_xyzypr.append(dse_lib.state_from_pose(pose))
        #self.camera_true_pub.publish(true_poses)
        self.camera_sim_pub.publish(sim_poses)
        # self.true_pose_data.append(self.order_poses(observed_ids, true_xyzypr))

        # error_poses = PoseMarkers()
        # error_poses.ids = observed_ids
        # error_poses.pose_array.header.stamp = rospy.Time.now()
        # error_poses.pose_array.header.frame_id = self.id_to_tf[self.this_agent_id] + '/camera_rgb_frame'
        # error_xyzypr = []
        # for i in range(len(observed_ids)):
        #     meas_pose = data.pose_array.poses[i]
        #     meas_eul = dse_lib.quat_from_pose2eul(meas_pose.orientation)
        #     r = R.from_euler(dse_constants.EULER_ORDER, meas_eul)
        #     meas_rotm = r.as_matrix()
        #     true_pose = true_poses.pose_array.poses[i]
        #     true_eul = dse_lib.quat_from_pose2eul(true_pose.orientation)
        #     r = R.from_euler(dse_constants.EULER_ORDER, true_eul)
        #     true_rotm = r.as_matrix()
        #     error_rotm = meas_rotm.dot(true_rotm.T)
        #     r = R.from_matrix(error_rotm)
        #     error_quat = r.as_quat()
        #
        #     pose = Pose()
        #     pose.position.x = true_pose.position.x - meas_pose.position.x
        #     pose.position.y = true_pose.position.y - meas_pose.position.y
        #     pose.position.z = true_pose.position.z - meas_pose.position.z
        #
        #     pose.orientation.x = error_quat[0]
        #     pose.orientation.y = error_quat[1]
        #     pose.orientation.z = error_quat[2]
        #     pose.orientation.w = error_quat[3]
        #     error_poses.pose_array.poses += [pose]
        #     error_xyzypr.append(dse_lib.state_from_pose(pose))
        # #self.camera_true_pub.publish(error_poses)
        # self.error_pose_data.append(self.order_poses(observed_ids, error_xyzypr))
        # # self.send_poses_csv(self.order_poses(observed_ids, error_xyzypr),
        # #                     self.order_poses(observed_ids, true_xyzypr))
        # # self.send_all_poses_csv(self.order_poses(observed_ids, error_xyzypr),
        # #                     self.order_poses(observed_ids, true_xyzypr),
        # #                     self.order_poses(observed_ids, est_xyzypr))
    #
    # def plot_measurements(self, data):
    #
    #     num_objects = len(self.init_ids)
    #     colors = ['k', 'g', 'r', 'm', 'b', 'c', 'y']
    #     agent_index = 0
    #     start_time = 0
    #     num_datapoints = len(self.error_pose_data)
    #
    #     plt.figure()
    #     # plt.tight_layout()
    #     # plt.suptitle('agent ' + str(agent_index))
    #     # plt.subplot(211)
    #     # plt.tight_layout()
    #     plt.grid()
    #     plt.xlim(-1.5, 2.5)
    #     plt.plot(0, 0, 'k.-', lw=2, label='true')
    #     plt.plot(0, 0, 'k--', lw=1, label='estimated')
    #     for i in range(num_objects):
    #         name = object_names[i]
    #         true_data = true_poses[agent_index][start:end, i]
    #         est_data = est_poses[agent_index][start:end, i]
    #         plt.plot(true_data[:, 0], true_data[:, 1], colors[i % len(colors)] + '.-', lw=2)
    #         plt.plot(est_data[:, 0], est_data[:, 1], colors[i % len(colors)] + '--', lw=2, label=name)
    #
    #     plt.legend()
    #     plt.xlabel('x (m)')
    #     plt.ylabel('y (m)')
    #     plt.title('true vs. estimated position')
    #
    #     plt.figure()
    #     # plt.subplot(212)
    #     # plt.tight_layout()
    #     plt.grid()
    #     # plt.xlim(15, 70)
    #     for i in range(num_objects):
    #         if object_ids[i] in agent_ids:
    #             name = agent_names[agent_ids.index(object_ids[i])]
    #             if name[0] == '/':
    #                 name = name[1:]
    #         else:
    #             name = object_names[i]
    #         true_data = true_poses[agent_index][start:end, i]
    #         est_data = est_poses[agent_index][start:end, i]
    #         time_data = time[agent_index][start:end]
    #         error = true_data - est_data
    #         error_dist = np.sqrt(error[:, 0] ** 2 + error[:, 1] ** 2)
    #         plt.plot(np.array(time_data), error_dist, colors[i % len(colors)] + '--', lw=2, label=name)
    #
    #     plt.legend()
    #     plt.xlabel('time (seconds)')
    #     plt.ylabel('distance error (m)')
    #     plt.title('error vs. time')
    #     plt.show()


def main(args):
    rospy.init_node('measurement_error_node', anonymous=True)
    me = measurement_error()   # This agent's ID is 1, and the state dimension is 6 (x, y, w, x_dot, y_dot, w_dot)
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")


if __name__ == '__main__':
    main(sys.argv)
