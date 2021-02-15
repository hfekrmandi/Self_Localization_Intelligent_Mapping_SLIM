#!/usr/bin/env python3
from __future__ import print_function
import roslib
import sys
import rospy
import numpy as np
import datetime
import time
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
roslib.load_manifest('dse_simulation')


class to_tag_controller:

    # Set up initial variables
    # Pass in the ID of this agent and the state dimension (6 or 12)
    def __init__(self):

        # self.v_nom = 0.5
        # self.radius = 2
        # self.ros_prefix = 'tb3_0'

        self.v_nom = rospy.get_param('~fwd_vel', 0.2)
        # Positive is to the left, negative is to the right
        self.radius = rospy.get_param('~radius', np.inf)
        self.ros_prefix = rospy.get_param('~prefix', '')
        if len(self.ros_prefix) != 0 and self.ros_prefix[0] != '/':
            self.ros_prefix = '/' + self.ros_prefix

        # Define publishers and subscribers
        # Publishes robot control signals
        self.control_pub = rospy.Publisher(self.ros_prefix + '/cmd_vel', Twist, queue_size=10)
        self.control_on_off_sub = rospy.Subscriber('/control_on', Bool, self.control_on_off)

        # Define static variables
        self.dt = 0.1
        self.t_last = rospy.get_time()
        self.euler_order = dse_constants.EULER_ORDER

        # Define controller parameters
        # time for 1 full circle is circumference / speed
        # angular velocity = 2*pi/time
        self.v_theta = self.v_nom / self.radius
        self.do_control = False

    def publish_control(self):
        # Publish control message
        control = Twist()
        if self.do_control:
            control.linear.x = self.v_nom
            control.angular.z = self.v_theta

        self.control_pub.publish(control)

    def control_on_off(self, data):
        self.do_control = data.data

def main(args):
    rospy.init_node('circle_controller_node', anonymous=True)
    il = to_tag_controller()
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
