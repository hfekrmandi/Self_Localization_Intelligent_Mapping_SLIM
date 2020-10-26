#!/usr/bin/env python
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
    def __init__(self, this_agent_id, dim_state):
        self.camera_pose_sub = rospy.Subscriber("/dse/pose_markers", PoseMarkers, self.measurement_callback)
        self.link_states_sub = rospy.Subscriber("/gazebo/link_states", LinkStates, self.gzbo_true_callback)
        self.python_true_sub = rospy.Subscriber("/dse/python_pose_true", PoseMarkers, self.pthn_true_callback)
        self.inf_results_sub = rospy.Subscriber("/dse/inf/results", InfFilterResults, self.results_callback)
        self.meas_vis_pub = rospy.Publisher("/dse/vis/measurement", PoseArray, queue_size=10)
        self.gzbo_vis_pub = rospy.Publisher("/dse/vis/gazebo_true", PoseArray, queue_size=10)
        self.pthn_vis_pub = rospy.Publisher("/dse/vis/python_true", PoseArray, queue_size=10)
        self.est_vis_pub = rospy.Publisher("/dse/vis/estimates", PoseArray, queue_size=10)

        self.dim_state = dim_state
        if self.dim_state == 6:
            self.dim_obs = 3
        elif self.dim_state == 12:
            self.dim_obs = 6
        else:
            rospy.signal_shutdown('invalid state dimension passed in')

        # Define static variables
        self.this_agent_id = this_agent_id
        self.dt = 0.1
        self.t_last = rospy.get_time()
        self.gzbo_ref_obj_state = None
        self.pthn_ref_obj_state = None

    # Create pose_array for measurement data
    def measurement_callback(self, data):
        poses = PoseArray()
        poses.poses = data.pose_array.poses
        poses.header.stamp = rospy.Time.now()
        poses.header.frame_id = 'base_link'
        self.meas_vis_pub.publish(poses)

    # Create pose_array for the information results
    def gzbo_true_callback(self, data):
        n = len(data.name)
        got_tag = False
        got_robot = False
        for i in range(n):
            if data.name[i] == 'aruco_marker::link':
                tag_state = dse_lib.state_from_pose_3D(data.pose[i])
                got_tag = True
            elif data.name[i] == 'turtlebot3_waffle_pi::base_footprint':
                robot_state = dse_lib.state_from_pose_3D(data.pose[i])
                got_robot = True

        if got_tag and got_robot:
            diff = dse_lib.agent2_to_frame_agent1_3D(robot_state, tag_state)
            poses = PoseArray()
            poses.header.stamp = rospy.Time.now()
            poses.header.frame_id = 'base_link'
            poses = dse_lib.pose_array_from_measurement(poses, diff, self.dim_obs)
            self.gzbo_vis_pub.publish(poses)

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

        print('estimations: ' + str(self.inf_x))

        poses = PoseArray()
        poses.header.stamp = rospy.Time.now()
        poses.header.frame_id = 'base_link'

        estimated_ids, estimated_states = dse_lib.relative_states_from_global_3D(self.this_agent_id, inf_id_list,
                                                                                 self.inf_x, self.dim_state, self.dim_obs)
        poses = dse_lib.pose_array_from_state(poses, estimated_states, self.dim_state, self.dim_obs)
        self.est_vis_pub.publish(poses)


def main(args):
    rospy.init_node('dse_gazebo_visualization_node', anonymous=True)
    imf = information_filter(1, 6)
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")


if __name__ == '__main__':
    main(sys.argv)
