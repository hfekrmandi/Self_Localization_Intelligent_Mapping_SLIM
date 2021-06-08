#!/usr/bin/env python2
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
    def __init__(self):

        # Get parameters from launch file
        self.ros_prefix = rospy.get_param('~prefix')
        if len(self.ros_prefix) != 0 and self.ros_prefix[0] != '/':
            self.ros_prefix = '/' + self.ros_prefix
        self.tf_pretix = self.ros_prefix[1:]
        self.this_agent_id = rospy.get_param('~id')
        self.dim_state = rospy.get_param('~dim_state')


        # self.ros_prefix = '/tb3_0'
        # self.this_agent_id = 5
        # self.dim_state = 6

        self.camera_pose_sub = rospy.Subscriber(self.ros_prefix + "/dse/pose_markers", PoseMarkers, self.measurement_callback)
        self.inf_results_sub = rospy.Subscriber(self.ros_prefix + "/dse/inf/results", InfFilterResults, self.results_callback)
        self.meas_vis_pub = rospy.Publisher(self.ros_prefix + "/dse/vis/measurement", PoseArray, queue_size=10)

        self.est_ids = []
        self.est_vis_pubs = []#rospy.Publisher(self.ros_prefix + "/dse/vis/estimates", PoseArray, queue_size=10)

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

    # Create pose_array for measurement data
    def measurement_callback(self, data):
        poses = PoseArray()
        poses.poses = data.pose_array.poses
        poses.header.stamp = rospy.Time.now()
        if self.ros_prefix == '':
            poses.header.frame_id = 'base_footprint'
        else:
            poses.header.frame_id = self.tf_pretix + '/base_footprint'
        self.meas_vis_pub.publish(poses)

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
            poses.header.frame_id = self.tf_pretix + '/base_footprint'

        estimated_ids, estimated_states = dse_lib.relative_states_from_global_3D(self.this_agent_id, inf_id_list,
                                                                                 self.inf_x, self.dim_state, self.dim_obs)
        poses = dse_lib.pose_array_from_state(poses, estimated_states, self.dim_state, self.dim_obs)

        for id in inf_id_list:
            if id not in self.est_ids:
                self.est_ids.append(id)
                self.est_vis_pubs.append(rospy.Publisher(self.ros_prefix + "/dse/vis/estimates/" + str(id), PoseArray, queue_size=10))

        for id in estimated_ids:
            i = np.where(estimated_ids == id)[0][0]
            j = self.est_ids.index(id)

            i_min = i * self.dim_state
            i_max = i_min + self.dim_state
            mean = dse_lib.state_to_xyzypr(estimated_states[i_min:i_max])

            cov = dse_lib.sub_matrix(inf_P, estimated_ids, id, self.dim_state)
            cov = dse_lib.state_cov_to_covariance_matrix(cov)
            cov = dse_lib.rotate_covariance_xyzypr_state(cov, mean)

            estimates = np.random.multivariate_normal(mean, cov, 50)
            poses.poses = []
            for est in estimates:
                poses.poses.append(dse_lib.pose_from_state(est[:, None]))

            self.est_vis_pubs[j].publish(poses)


def main(args):

    rospy.init_node('dse_gazebo_visualization_node', anonymous=True)
    imf = information_filter()
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")


if __name__ == '__main__':
    main(sys.argv)
