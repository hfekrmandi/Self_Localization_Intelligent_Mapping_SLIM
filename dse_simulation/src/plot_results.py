#!/usr/bin/env python3
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


def main(args):

    dump_file = "simulation_data_3_agents_left_awayLine.p"
    cal = pickle.load(open(os.path.join(sys.path[0], dump_file), "rb"))
    [header, time, object_ids, object_names, agent_names, agent_ids, true_poses, est_poses, est_covariances] = cal
    print('got data')

    for i in range(len(agent_names)):
        if agent_names[i][0] == '/':
            agent_names[i] = agent_names[i][1:]

    for i in range(len(object_names)):
        name = object_names[i]
        if name[:5] == 'aruco':
            object_names[i] = 'aruco_' + name[-1:]
        if name[:3] == 'tb3':
            object_names[i] = 'agent_' + name[-1:]
            agent_names[agent_names.index(name)] = 'agent_' + name[-1:]

    agents = [4, 5, 6]
    new_order = agents[:len(agent_ids)] + [0, 1, 2, 3]
    new_order = np.array(new_order)
    for i in range(len(agent_ids)):
        true_poses[i] = true_poses[i][:, new_order, :]
        est_poses[i] = est_poses[i][:, new_order, :]

    num_objects = len(object_ids)
    num_agents = len(agent_ids)
    colors = ['k', 'g', 'r', 'm', 'b', 'c', 'y']
    agent_index = 0
    start_time = 0
    num_datapoints = np.shape(time[agent_index][time[agent_index] > start_time])[0]
    
    start = np.shape(time[agent_index])[0] - num_datapoints 
    end = np.shape(time[agent_index])[0]

    plt.figure()
    #plt.tight_layout()
    #plt.suptitle('agent ' + str(agent_index))
    #plt.subplot(211)
    #plt.tight_layout()
    plt.grid()
    plt.xlim(-1.5, 2.5)
    plt.plot(0, 0, 'k.-', lw=2, label='true')
    plt.plot(0, 0, 'k--', lw=1, label='estimated')
    for i in range(num_objects):
        if object_ids[i] in agent_ids:
            name = agent_names[agent_ids.index(object_ids[i])]
            if name[0] == '/':
                name = name[1:]
        else:
            name = object_names[i]
        true_data = true_poses[agent_index][start:end, i]
        est_data = est_poses[agent_index][start:end, i]
        plt.plot(true_data[:, 0], true_data[:, 1], colors[i % len(colors)] + '.-', lw=2)
        plt.plot(est_data[:, 0], est_data[:, 1], colors[i % len(colors)] + '--', lw=2, label=name)

    plt.legend()
    plt.xlabel('x (m)')
    plt.ylabel('y (m)')
    plt.title('true vs. estimated position')

    plt.figure()
    #plt.subplot(212)
    #plt.tight_layout()
    plt.grid()
    #plt.xlim(15, 70)
    for i in range(num_objects):
        if object_ids[i] in agent_ids:
            name = agent_names[agent_ids.index(object_ids[i])]
            if name[0] == '/':
                name = name[1:]
        else:
            name = object_names[i]
        true_data = true_poses[agent_index][start:end, i]
        est_data = est_poses[agent_index][start:end, i]
        time_data = time[agent_index][start:end]
        error = true_data - est_data
        error_dist = np.sqrt(error[:, 0] ** 2 + error[:, 1] ** 2)
        plt.plot(np.array(time_data), error_dist, colors[i % len(colors)] + '--', lw=2, label=name)

    plt.legend()
    plt.xlabel('time (seconds)')
    plt.ylabel('distance error (m)')
    plt.title('error vs. time')
    plt.show()



if __name__ == '__main__':
    main(sys.argv)



# plt.plot(self.x_permanent, self.y_permanent, 'r.', lw=2)
# # plt.plot(self.est_1_xyt_permanent[:][0], self.est_1_xyt_permanent[:][1], 'r-', lw=2)
# # plt.plot(self.est_2_xyt_permanent[:][0], self.est_2_xyt_permanent[:][1], 'g-', lw=2)
# # plt.plot(self.est_3_xyt_permanent[:][0], self.est_3_xyt_permanent[:][1], 'b-', lw=2)
# plt.xlabel('x (m)')
# plt.ylabel('y (m)')
# plt.title('agent 1s trajectory estimates')
# plt.grid(True)
#
# plt.show(block=False)
# plt.pause(0.001)
#
#
# class information_filter:
#
#     # Define initial/setup values
#     def __init__(self):
#
#         self.time = []
#         self.true_poses = []
#         self.est_poses = []
#         self.est_covariances = []
#
#         # # Get parameters from launch file
#         # self.ros_prefix = rospy.get_param('~prefix')
#         # if len(self.ros_prefix) != 0 and self.ros_prefix[0] != '/':
#         #     self.ros_prefix = '/' + self.ros_prefix
#         # self.tf_pretix = self.ros_prefix[1:]
#         # self.dim_state = rospy.get_param('~dim_state')
#         # Get parameters from launch file
#         # self.n_params = 3
#         # self.dim_state = 6
#         # self.object_names = ['tb3_0', 'tb3_1', 'tb3_2']
#         self.object_names = rospy.get_param('~objects')
#         self.object_ids = rospy.get_param('~object_ids')
#         self.agent_ids = rospy.get_param('~agent_ids')
#
#         self.dim_state = rospy.get_param('~dim_state', 6)
#         n_params = len(self.agent_ids)
#         self.store_data_sub = rospy.Subscriber('/store_data', Bool, self.store_data)
#
#         self.gazebo_model_object = gazebo_lib.GazeboModel(self.object_names)
#         self.inf_results_subs = []
#         self.agent_names = []
#         for i in range(n_params):
#             if len(self.agent_names[i]) != 0 and self.agent_names[i][0] != '/':
#                 self.agent_names[i] = '/' + self.agent_names[i]
#
#             self.inf_results_subs.append(rospy.Subscriber(
#                 self.agent_names[i] + "/dse/inf/results", InfFilterResults, self.results_callback, i))
#
#             self.true_poses.append([])
#             self.est_poses.append([])
#             self.est_covariances.append([])
#
#             index = self.object_ids.index(self.agent_ids[i])
#             self.agent_names.append(self.object_names[index])
#
#         if self.dim_state == 6:
#             self.dim_obs = 3
#         elif self.dim_state == 12:
#             self.dim_obs = 6
#         else:
#             rospy.signal_shutdown('invalid state dimension passed in')
#
#     # Create pose_array for the information results
#     def results_callback(self, data, agent_index):
#         inf_id_list = np.array(data.ids)
#         inf_Y = dse_lib.multi_array_2d_output(data.inf_matrix)
#         inf_y = dse_lib.multi_array_2d_output(data.inf_vector)
#         self.inf_x = np.linalg.inv(inf_Y).dot(inf_y)
#         inf_P = np.linalg.inv(inf_Y)
#
#         for id in inf_id_list:
#             name_index = self.object_ids.index(id)
#             name = self.object_names[name_index]
#             i = np.where(inf_id_list == id)[0][0]
#
#             i_min = i * self.dim_state
#             i_max = i_min + self.dim_state
#             est_pose = dse_lib.pose_from_state_3D(self.inf_x[i_min:i_max])
#             cov = dse_lib.sub_matrix(inf_P, inf_id_list, id, self.dim_state)
#             cov = dse_lib.state_cov_to_covariance_matrix(cov)
#             est_covariance = list(dse_lib.covariance_to_ros_covariance(cov))
#             true_pose = self.gazebo_model_object.get_model_pose(name)
#
#             self.true_poses[agent_index].append(true_pose)
#             self.est_poses[agent_index].append(est_pose)
#             #self.est_covariances[agent_index].append(est_covariance)
#
#     def store_data(self, data):
#         if data.data:
#             header = '[header, object_ids, object_names, agent_names, agent_ids, true_poses, est_poses, est_covariances]'
#             cal = [header, self.object_ids, self.object_names, self.agent_names, self.agent_ids, self.true_poses, self.est_poses, self.est_covariances]
#             dump_file = "simulation_data_" + rospy.Time.now() + ".p"
#             pickle.dump(cal, open(os.path.join(sys.path[0], dump_file), "wb"))
#
#     def plot_callback(self, data):
#
#         plt.clf()
#         plt.plot(self.x_permanent, self.y_permanent, 'r.', lw=2)
#         # plt.plot(self.est_1_xyt_permanent[:][0], self.est_1_xyt_permanent[:][1], 'r-', lw=2)
#         # plt.plot(self.est_2_xyt_permanent[:][0], self.est_2_xyt_permanent[:][1], 'g-', lw=2)
#         # plt.plot(self.est_3_xyt_permanent[:][0], self.est_3_xyt_permanent[:][1], 'b-', lw=2)
#         plt.xlabel('x (m)')
#         plt.ylabel('y (m)')
#         plt.title('agent 1s trajectory estimates')
#         plt.grid(True)
#
#         plt.show(block=False)
#         plt.pause(0.001)
#
#
# def main(args):
#
#     rospy.init_node('dse_plotting_node', anonymous=True)
#     imf = information_filter()
#     try:
#         rospy.spin()
#     except KeyboardInterrupt:
#         print("Shutting down")
#
#
# if __name__ == '__main__':
#     main(sys.argv)
