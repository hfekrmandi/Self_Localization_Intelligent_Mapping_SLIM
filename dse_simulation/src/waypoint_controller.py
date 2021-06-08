#!/usr/bin/env python2
from __future__ import print_function
import roslib
import sys
import rospy
import numpy as np
import datetime
import time
import tf2_ros
from geometry_msgs.msg import Twist
from std_msgs.msg import Bool
from dse_msgs.msg import PoseMarkers
from std_msgs.msg import Float64MultiArray
from std_msgs.msg import MultiArrayLayout
from std_msgs.msg import MultiArrayDimension
from dse_msgs.msg import InfFilterPartials
from dse_msgs.msg import InfFilterResults
from scipy.spatial.transform import Rotation as R

import dse_lib
import dse_constants
import gazebo_lib
roslib.load_manifest('dse_simulation')


class waypoint_controller:

    # Set up initial variables
    # Pass in the ID of this agent and the state dimension (6 or 12)
    def __init__(self):

        # self.v_nom = 0.5
        # self.radius = 2
        # self.ros_prefix = 'tb3_0'

        self.v_nom = rospy.get_param('~fwd_vel', 0.2)
        self.ros_prefix = rospy.get_param('~prefix', '')
        if len(self.ros_prefix) != 0 and self.ros_prefix[0] != '/':
            self.ros_prefix = '/' + self.ros_prefix
        self.points = np.array(rospy.get_param('~points_array'))
        self.t_settle = rospy.get_param('~t_settle', 1)
        self.robot_d_sq = rospy.get_param('~threshold_dist', 0.1) ** 2
        self.robot_tf = self.ros_prefix[1:] + '/base_footprint'
        self.current_point = 0

        # Define static variables
        self.dt = 0.1
        self.t_last = rospy.get_time()
        self.euler_order = dse_constants.EULER_ORDER

        self.theta_error_to_v_theta = self.dt / self.t_settle
        self.do_control = False

        self.tfBuffer = tf2_ros.Buffer()
        self.listener = tf2_ros.TransformListener(self.tfBuffer)
        time.sleep(5)

        # Define publishers and subscribers
        # Publishes robot control signals
        self.control_pub = rospy.Publisher(self.ros_prefix + '/cmd_vel', Twist, queue_size=10)
        self.control_on_off_sub = rospy.Subscriber('/control_on', Bool, self.control_on_off)
        # Returns the angle difference between the current trajectory and the goal, measured CCW from the current trajectory

    def theta_error(self, x, y, t, x_d, y_d):
        t_goal = np.arctan2(y_d - y, x_d - x)
        e = t_goal - t
        ## CRITICAL: ENSURE THAT THE ERROR IS BETWEEN -PI, PI OTHERWISE IT BEHAVES WEIRD
        if e > np.pi:
            e = -np.pi * 2 + e
        elif e < -np.pi:
            e = np.pi * 2 + e
        return e

    def publish_control(self):
        # Publish control message
        control = Twist()
        if self.do_control and self.current_point < self.points.shape[0] - 1:
            control.linear.x = self.v_nom

            our_pose = gazebo_lib.object_pose_in_world(self.robot_tf, self.tfBuffer)
            our_eul = dse_lib.quat_from_pose2eul(our_pose.pose.orientation)
            loc = [our_pose.pose.position.x, our_pose.pose.position.y, our_eul[0]]

            # print('target waypoint: ', self.points[self.current_point + 1])
            d = (self.points[self.current_point + 1][0] - loc[0]) ** 2 + \
                (self.points[self.current_point + 1][1] - loc[1]) ** 2
            # print('distance', np.sqrt(d))

            ## Compute the angle error
            e = self.theta_error(loc[0], loc[1], loc[2],
                                 self.points[self.current_point + 1][0], self.points[self.current_point + 1][1])
            # print('Angle error is: ' + str(e))
            # Compute new wheel angle and send it to the car
            control.angular.z = e * self.theta_error_to_v_theta

            ## Determine if we passed the obstacle
            d = (self.points[self.current_point + 1][0] - loc[0]) ** 2 + \
                (self.points[self.current_point + 1][1] - loc[1]) ** 2
            if d < self.robot_d_sq:
                self.current_point += 1

        self.control_pub.publish(control)

    def control_on_off(self, data):
        self.do_control = data.data


def main(args):
    rospy.init_node('waypoint_controller_node', anonymous=True)
    il = waypoint_controller()
    r = rospy.Rate(10)
    il.dt = 1 / 10.0
    try:
        while True:
            r.sleep()
            il.publish_control()
    except KeyboardInterrupt:
        print("Shutting down")


if __name__ == '__main__':
    main(sys.argv)
