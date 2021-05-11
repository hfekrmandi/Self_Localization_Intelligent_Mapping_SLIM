#!/usr/bin/env python2
from __future__ import print_function
import roslib
import sys
import rospy
import numpy as np
import datetime
import time
from dse_msgs.msg import PoseMarkers
from std_msgs.msg import Float64MultiArray
from std_msgs.msg import MultiArrayLayout
from std_msgs.msg import MultiArrayDimension
from dse_msgs.msg import InfFilterPartials
from dse_msgs.msg import InfFilterResults
from scipy.spatial.transform import Rotation as R

from dse_lib import *

roslib.load_manifest('dse_simulation')

class direct_estimator:

    # Define setup/initial values
    def __init__(self):

        # Get parameters from launch file
        self.ros_prefix = rospy.get_param('~prefix', '')
        if len(self.ros_prefix) != 0 and self.ros_prefix[0] != '/':
            self.ros_prefix = '/' + self.ros_prefix

        self.results_pub = rospy.Publisher(self.ros_prefix + "/dse/inf/results", InfFilterResults, queue_size=10)
        self.inf_sub = rospy.Subscriber(self.ros_prefix + "/dse/inf/partial", InfFilterPartials, self.information_callback)

    # When the information filter sends partials (prior and measurement), combine and return them
    def information_callback(self, data):
        inf_id_list = data.ids
        inf_Y = multi_array_2d_output(data.inf_matrix_prior)
        inf_y = multi_array_2d_output(data.inf_vector_prior)
        inf_I = multi_array_2d_output(data.obs_matrix)
        inf_i = multi_array_2d_output(data.obs_vector)

        inf_Y = inf_Y + inf_I
        inf_y = inf_y + inf_i

        inf_results = InfFilterResults()
        inf_results.ids = inf_id_list
        inf_results.inf_matrix = multi_array_2d_input(inf_Y, inf_results.inf_matrix)
        inf_results.inf_vector = multi_array_2d_input(inf_y, inf_results.inf_vector)
        self.results_pub.publish(inf_results)


def main(args):
    rospy.init_node('direct_estimator_node', anonymous=True)
    de = direct_estimator()
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")


if __name__ == '__main__':
    main(sys.argv)
