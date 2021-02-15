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
from gazebo_msgs.msg import LinkStates
import tf_conversions
import tf2_ros

import dse_lib
import dse_constants
roslib.load_manifest('dse_simulation')


class information_filter:

    # Define initial/setup values
    def __init__(self, dim_state):
        self.camera_pose_sub = rospy.Subscriber("/dse/pose_markers", PoseMarkers, self.measurement_callback)
        self.link_states_sub = rospy.Subscriber("/gazebo/link_states", LinkStates, self.gzbo_true_callback)
        self.python_true_sub = rospy.Subscriber("/dse/python_pose_true", PoseMarkers, self.pthn_true_callback)
        self.inf_results_sub = rospy.Subscriber("/dse/inf/results", InfFilterResults, self.results_callback)
        self.meas_vis_pub = rospy.Publisher("/dse/vis/measurement", PoseArray, queue_size=10)
        self.gzbo_est_vis_pub = rospy.Publisher("/dse/vis/gazebo_true", PoseArray, queue_size=10)
        self.pthn_est_vis_pub = rospy.Publisher("/dse/vis/python_true", PoseArray, queue_size=10)
        self.origin_vis_pub = rospy.Publisher("/dse/vis/origin", PoseArray, queue_size=10)
        self.est_vis_pub = rospy.Publisher("/dse/vis/estimates", PoseArray, queue_size=10)

        self.dim_state = dim_state
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

    def flip_measurement(self, x, id_list, ref_obj, poses):
        i_low = self.dim_state * dse_constants.GAZEBO_REFERENCE_OBJECT_ID
        i_high = i_low + self.dim_obs
        ref_obj_est_state = x[i_low:i_high]

        for i in range(len(id_list)):

            i_low = self.dim_state * i
            i_high = i_low + self.dim_obs
            measurement = x[i_low:i_high]

            if id_list[i] != dse_constants.GAZEBO_REFERENCE_OBJECT_ID:
                if self.dim_obs == 3:
                    est_in_ref_frame = dse_lib.agent2_to_frame_agent1_3D(measurement, ref_obj)
                    poses.poses.append(dse_lib.pose_from_state_3D(est_in_ref_frame))
                else:
                    est_in_ref_frame = dse_lib.agent2_to_frame_agent1(measurement, ref_obj)
                    poses.poses.append(dse_lib.pose_from_state(est_in_ref_frame))
        return poses

    # Create pose_array for measurement data
    def measurement_callback(self, data):
        id_list = [1]
        poses = data.pose_array

        x = dse_lib.state_from_pose_array(poses, self.dim_state, self.dim_obs)
        print('measurement: ' + str(x))

        poses = PoseArray()
        poses.header.stamp = rospy.Time.now()
        poses.header.frame_id = 'odom'

        if self.gzbo_ref_obj_state is not None:
            poses = self.flip_measurement(x, id_list, self.gzbo_ref_obj_state, poses)
        elif self.pthn_ref_obj_state is not None:
            poses = self.flip_measurement(x, id_list, self.pthn_ref_obj_state, poses)

        self.meas_vis_pub.publish(poses)

        # poses = PoseArray()
        # poses.header.stamp = rospy.Time.now()
        # poses.header.frame_id = 'odom'
        # self.origin_state = np.zeros((self.dim_state, 1))
        # poses = dse_lib.pose_array_from_state(poses, self.origin_state, self.dim_state, self.dim_obs)
        # self.origin_vis_pub.publish(poses)

    # Create pose_array for the information results
    def gzbo_true_callback(self, data):
        n = len(data.name)
        for i in range(n):
            if data.name[i] == dse_constants.GAZEBO_REFERENCE_OBJECT_NAME:
                self.gzbo_ref_obj_state = dse_lib.state_from_pose_3D(data.pose[i])
                break

    # Create pose_array for the information results
    def pthn_true_callback(self, data):
        n = len(data.ids)
        for i in range(n):
            if data.ids[i] == dse_constants.GAZEBO_REFERENCE_OBJECT_ID:
                self.pthn_ref_obj_state = dse_lib.state_from_pose_3D(data.pose_array.poses[i])
                break

    def values_in_ref_frame(self, x, id_list, ref_obj, poses):
        if self.dim_obs == 3:
            ref_obj_est_state = dse_lib.state_from_id(x, id_list, dse_constants.GAZEBO_REFERENCE_OBJECT_ID, 6)
            ref_obj_est_state = ref_obj_est_state[0:3]
        else:
            ref_obj_est_state = dse_lib.state_from_id(x, id_list, dse_constants.GAZEBO_REFERENCE_OBJECT_ID, 12)
            ref_obj_est_state = ref_obj_est_state[0:6]

        for i in range(len(id_list)):

            i_low = self.dim_state * i
            i_high = i_low + self.dim_obs
            agent_i_state = x[i_low:i_high]

            if id_list[i] != dse_constants.GAZEBO_REFERENCE_OBJECT_ID:
                if self.dim_obs == 3:
                    estimation = dse_lib.agent2_to_frame_agent1_3D(ref_obj_est_state, agent_i_state)
                    est_in_ref_frame = dse_lib.agent2_to_frame_agent1_3D(ref_obj, estimation)
                    poses.poses.append(dse_lib.pose_from_state_3D(est_in_ref_frame))
                else:
                    estimation = dse_lib.agent2_to_frame_agent1(ref_obj_est_state, agent_i_state)
                    est_in_ref_frame = dse_lib.agent2_to_frame_agent1(ref_obj, estimation)
                    poses.poses.append(dse_lib.pose_from_state(est_in_ref_frame))
        return poses

    # Create pose_array for the information results
    def results_callback(self, data):
        inf_id_list = np.array(data.ids)
        inf_Y = dse_lib.multi_array_2d_output(data.inf_matrix)
        inf_y = dse_lib.multi_array_2d_output(data.inf_vector)
        inf_x = np.linalg.inv(inf_Y).dot(inf_y)
        inf_P = np.linalg.inv(inf_Y)

        print('estimations: ' + str(inf_x))

        poses = PoseArray()
        poses.header.stamp = rospy.Time.now()
        poses.header.frame_id = 'odom'

        if self.gzbo_ref_obj_state is not None:
            poses = self.values_in_ref_frame(inf_x, inf_id_list, self.gzbo_ref_obj_state, poses)
        elif self.pthn_ref_obj_state is not None:
            poses = self.values_in_ref_frame(inf_x, inf_id_list, self.pthn_ref_obj_state, poses)

        self.est_vis_pub.publish(poses)

        # poses = PoseArray()
        # poses.header.stamp = rospy.Time.now()
        # poses.header.frame_id = 'odom'
        # self.origin_state = np.zeros((self.dim_state, 1))
        # poses = dse_lib.pose_array_from_state(poses, self.origin_state, self.dim_state, self.dim_obs)
        # self.origin_vis_pub.publish(poses)


def main(args):
    rospy.init_node('dse_gazebo_visualization_node', anonymous=True)
    imf = information_filter(6)
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")


if __name__ == '__main__':
    main(sys.argv)
