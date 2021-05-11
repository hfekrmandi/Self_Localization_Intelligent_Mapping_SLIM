#!/usr/bin/env python2
from __future__ import print_function
import roslib
import sys
import rospy
import numpy as np
import datetime
import time
import os
import pickle
from geometry_msgs.msg import PoseArray
from geometry_msgs.msg import Pose
from geometry_msgs.msg import PoseWithCovariance
from std_msgs.msg import Bool
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

import gazebo_lib
import dse_lib
import dse_constants
roslib.load_manifest('dse_simulation')


class information_filter:

    # Define initial/setup values
    def __init__(self):

        self.time = []
        self.true_poses = []
        self.est_poses = []
        self.est_covariances = []

        # # Get parameters from launch file
        # self.ros_prefix = rospy.get_param('~prefix')
        # if len(self.ros_prefix) != 0 and self.ros_prefix[0] != '/':
        #     self.ros_prefix = '/' + self.ros_prefix
        # self.tf_pretix = self.ros_prefix[1:]
        # self.dim_state = rospy.get_param('~dim_state')
        # Get parameters from launch file
        # self.n_params = 3
        # self.dim_state = 6
        # self.object_names = ['tb3_0', 'tb3_1', 'tb3_2', 'aruco_marker_0', 'aruco_marker_1', 'aruco_marker_2', 'aruco_marker_3']
        # self.object_ids = [5, 6, 7, 0, 1, 2, 3]
        # self.agent_ids = [5, 6, 7]
        # self.dim_state = 6
        self.object_names = rospy.get_param('~objects')
        self.object_ids = rospy.get_param('~object_ids')
        self.agent_ids = rospy.get_param('~agent_ids')
        self.dim_state = rospy.get_param('~dim_state', 6)

        n_params = len(self.agent_ids)
        self.store_data_sub = rospy.Subscriber('/store_data', Bool, self.store_data)

        self.gazebo_model_object = gazebo_lib.GazeboModel(self.object_names)
        self.inf_results_subs = []
        self.agent_names = []
        for i in range(n_params):
            index = self.object_ids.index(self.agent_ids[i])
            self.agent_names.append(self.object_names[index])

            if len(self.agent_names[i]) != 0 and self.agent_names[i][0] != '/':
                self.agent_names[i] = '/' + self.agent_names[i]

            self.inf_results_subs.append(rospy.Subscriber(
                self.agent_names[i] + "/dse/inf/results", InfFilterResults, self.results_callback, i))

            self.time.append([])
            self.true_poses.append([])
            self.est_poses.append([])
            self.est_covariances.append([])

        if self.dim_state == 6:
            self.dim_obs = 3
        elif self.dim_state == 12:
            self.dim_obs = 6
        else:
            rospy.signal_shutdown('invalid state dimension passed in')

    # Create pose_array for the information results
    def results_callback(self, data, agent_index):
        inf_id_list = np.array(data.ids)
        inf_Y = dse_lib.multi_array_2d_output(data.inf_matrix)
        inf_y = dse_lib.multi_array_2d_output(data.inf_vector)
        self.inf_x = np.linalg.inv(inf_Y).dot(inf_y)
        inf_P = np.linalg.inv(inf_Y)

        est_pose = []
        true_pose = []
        est_covariance = []

        for id in inf_id_list:
            name_index = self.object_ids.index(id)
            name = self.object_names[name_index]
            i = np.where(inf_id_list == id)[0][0]

            i_min = i * self.dim_state
            i_max = i_min + self.dim_state
            this_pose = dse_lib.pose_from_state_3D(self.inf_x[i_min:i_max])
            this_xyy = dse_lib.state_to_xyzypr(dse_lib.state_from_pose_3D(this_pose))
            est_pose.append(this_xyy)
            cov = dse_lib.sub_matrix(inf_P, inf_id_list, id, self.dim_state)
            cov = dse_lib.state_cov_to_covariance_matrix(cov)
            est_covariance.append(cov)
            this_pose = self.gazebo_model_object.get_model_pose(name)
            this_xyy = dse_lib.state_to_xyzypr(dse_lib.state_from_pose_3D(this_pose))
            true_pose.append(this_xyy)

        time = rospy.Time.now().secs + rospy.Time.now().nsecs / 1000000000
        self.time[agent_index].append(time)
        self.true_poses[agent_index].append(np.array(true_pose))
        self.est_poses[agent_index].append(np.array(est_pose))
        #self.est_covariances[agent_index].append(est_covariance)

    def store_data(self, data):
        if data.data:
            time_np = []
            true_poses_np = []
            est_poses_np = []
            est_covariances_np = []
            for i in range(len(self.agent_ids)):
                time_np.append(np.array(self.time[i]))
                true_poses_np.append(np.array(self.true_poses[i]))
                est_poses_np.append(np.array(self.est_poses[i]))
                est_covariances_np.append(np.array(self.est_covariances[i]))

            header = '[header, time, object_ids, object_names, agent_names, agent_ids, true_poses, est_poses, est_covariances]'
            cal = [header, time_np, self.object_ids, self.object_names, self.agent_names, self.agent_ids,
                   true_poses_np, est_poses_np, est_covariances_np]
            dump_file = "simulation_data_" + str(rospy.Time.now()) + ".p"
            pickle.dump(cal, open(os.path.join(sys.path[0], dump_file), "wb"))


def main(args):

    rospy.init_node('dse_plotting_node', anonymous=True)
    imf = information_filter()
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")


if __name__ == '__main__':
    main(sys.argv)
