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
from dse_msgs.msg import InfFilterPartials
from visualization_msgs.msg import Marker
from scipy.spatial.transform import Rotation as R
import tf_conversions
import tf2_ros
import message_filters

import dse_lib
import dse_constants

roslib.load_manifest('dse_simulation')


class central_est:

    # Define initial/setup values
    def __init__(self):

        # Get parameters from launch file

        self.n_params = 3
        self.dim_state = 6
        # self.dim_state = rospy.get_param('~dim_state', 6)
        # self.n_params = rospy.get_param('~n_params')
        # self.object_names = []
        self.inf_subs = []
        self.inf_pubs = []
        self.inf_id_list = []
        self.inf_Y = []
        self.inf_y = []
        self.inf_I = []
        self.inf_i = []
        self.object_names = ['tb3_0', 'tb3_1', 'tb3_2']
        self.got_information = []
        # self.agent_ids = [2000, 2001, 2002]
        for i in range(self.n_params):
            # self.object_names.append(rospy.get_param('~objects')[i])
            self.inf_pubs.append(
                rospy.Publisher(self.object_names[i] + "/dse/inf/results", InfFilterResults, queue_size=10))
            self.inf_subs.append(rospy.Subscriber(self.object_names[i] + "/dse/inf/partial", InfFilterPartials,
                                                  self.information_callback, i))
            self.inf_id_list.append([0, 0])
            self.inf_Y.append([[0, 0], [0, 0]])
            self.inf_y.append([[0, 0], [0, 0]])
            self.inf_I.append([[0, 0], [0, 0]])
            self.inf_i.append([[0, 0], [0, 0]])
            self.got_information.append(False)

        print('n_params: %d' % self.n_params)
        print('state dim: %d' % self.dim_state)
        print('object names: ' + str(self.object_names))

    # When the information filter sends partials (prior and measurement), combine and return them
    def information_callback(self, data, agent_index):
        print('got callback')
        self.inf_id_list[agent_index] = data.ids
        self.inf_Y[agent_index] = dse_lib.multi_array_2d_output(data.inf_matrix_prior)
        self.inf_y[agent_index] = dse_lib.multi_array_2d_output(data.inf_vector_prior)
        self.inf_I[agent_index] = dse_lib.multi_array_2d_output(data.obs_matrix)
        self.inf_i[agent_index] = dse_lib.multi_array_2d_output(data.obs_vector)
        self.got_information[agent_index] = True

    # When the information filter sends partials (prior and measurement), combine and return them
    def estimate_and_send(self):

        array_ids = []
        array_Y = []
        array_y = []
        array_I = []
        array_i = []
        for i in range(self.n_params):
            if self.got_information[i]:
                array_ids.append(self.inf_id_list[i])
                array_Y.append(self.inf_Y[i])
                array_y.append(self.inf_y[i])
                array_I.append(self.inf_I[i])
                array_i.append(self.inf_i[i])

        array_ids, array_Y, array_y, array_I, array_i = \
            dse_lib.get_sorted_agent_states(array_ids, array_Y, array_y, array_I, array_i, self.dim_state)

        inf_id_list = array_ids[0]
        inf_Y = array_Y[0]
        inf_y = array_y[0]
        inf_I = array_I[0]
        inf_i = array_i[0]
        for i in range(1, len(array_ids)):
            inf_I = np.add(inf_I, array_I[i])
            inf_i = np.add(inf_i, array_i[i])

        inf_Y = np.add(inf_Y, inf_I)
        inf_y = np.add(inf_y, inf_i)

        for i in range(self.n_params):
            inf_results = InfFilterResults()
            inf_results.ids = inf_id_list
            inf_results.inf_matrix = dse_lib.multi_array_2d_input(inf_Y, inf_results.inf_matrix)
            inf_results.inf_vector = dse_lib.multi_array_2d_input(inf_y, inf_results.inf_vector)
            self.inf_pubs[i].publish(inf_results)
            self.got_information[i] = False


def main(args):
    rospy.init_node('dse_centralized_estimator_node', anonymous=True)
    est = central_est()
    r = rospy.Rate(10)
    try:
        while True:
            r.sleep()
            est.estimate_and_send()
    except KeyboardInterrupt:
        print("Shutting down")


if __name__ == '__main__':
    main(sys.argv)
