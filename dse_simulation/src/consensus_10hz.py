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
import copy

import dse_lib
import dse_constants

roslib.load_manifest('dse_simulation')


class hybrid_consensus:

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
        self.inf_indices = []
        self.object_names = ['tb3_0', 'tb3_1', 'tb3_2']
        # self.object_names = rospy.get_param('~objects')
        for i in range(self.n_params):
            if len(self.object_names[i]) != 0 and self.object_names[i][0] != '/':
                self.object_names[i] = '/' + self.object_names[i]
        # self.agent_ids = [2000, 2001, 2002]
        for i in range(self.n_params):
            self.inf_pubs.append(rospy.Publisher(
                self.object_names[i] + "/dse/inf/results", InfFilterResults, queue_size=10))
            self.inf_subs.append(rospy.Subscriber(
                self.object_names[i] + "/dse/inf/partial", InfFilterPartials, self.information_callback, i))

    # When the information filter sends partials (prior and measurement), combine and return them
    def information_callback(self, data, agent_index):
        if agent_index in self.inf_indices:
            inf_index = np.where(np.array(self.inf_indices) == agent_index)[0][0]
            self.inf_id_list[inf_index] = data.ids
            self.inf_Y[inf_index] = dse_lib.multi_array_2d_output(data.inf_matrix_prior)
            self.inf_y[inf_index] = dse_lib.multi_array_2d_output(data.inf_vector_prior)
            self.inf_I[inf_index] = dse_lib.multi_array_2d_output(data.obs_matrix)
            self.inf_i[inf_index] = dse_lib.multi_array_2d_output(data.obs_vector)
        else:
            self.inf_indices.append(agent_index)
            self.inf_id_list.append(data.ids)
            self.inf_Y.append(dse_lib.multi_array_2d_output(data.inf_matrix_prior))
            self.inf_y.append(dse_lib.multi_array_2d_output(data.inf_vector_prior))
            self.inf_I.append(dse_lib.multi_array_2d_output(data.obs_matrix))
            self.inf_i.append(dse_lib.multi_array_2d_output(data.obs_vector))

    # Option 1:
    #   Create a communication graph for the whole system
    # Option 2:
    #   Split the system into isolated groups
    #   Create a graph for each group


    def apply_comm_model(self, ):
        #
        return

    # When the information filter sends partials (prior and measurement), combine and return them
    def estimate_and_send(self):
        if len(self.inf_indices) > 0:

            array_ids = copy.deepcopy(self.inf_id_list)
            array_Y = copy.deepcopy(self.inf_Y)
            array_y = copy.deepcopy(self.inf_y)
            array_I = copy.deepcopy(self.inf_I)
            array_i = copy.deepcopy(self.inf_i)

            array_ids, array_Y, array_y, array_I, array_i = \
                dse_lib.get_sorted_agent_states(array_ids, array_Y, array_y, array_I, array_i, self.dim_state)


            # for i in range(len(self.inf_indices)):
            #     inf_id_list = array_ids[i]
            #     inf_Y = array_Y[i]
            #     inf_y = array_y[i]
            #     inf_I = array_I[i]
            #     inf_i = array_i[i]
            #
            #     inf_Y = np.add(inf_Y, inf_I)
            #     inf_y = np.add(inf_y, inf_i)
            #
            #     inf_x = np.linalg.inv(inf_Y).dot(inf_y)
            #     inf_P = np.linalg.inv(inf_Y)
            #
            #     inf_results = InfFilterResults()
            #     inf_results.ids = inf_id_list
            #     inf_results.inf_matrix = dse_lib.multi_array_2d_input(inf_Y, inf_results.inf_matrix)
            #     inf_results.inf_vector = dse_lib.multi_array_2d_input(inf_y, inf_results.inf_vector)
            #     self.inf_pubs[self.inf_indices[i]].publish(inf_results)


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

            for i in self.inf_indices:
                inf_results = InfFilterResults()
                inf_results.ids = inf_id_list
                inf_results.inf_matrix = dse_lib.multi_array_2d_input(inf_Y, inf_results.inf_matrix)
                inf_results.inf_vector = dse_lib.multi_array_2d_input(inf_y, inf_results.inf_vector)
                self.inf_pubs[i].publish(inf_results)


            self.inf_indices = []
            self.inf_id_list = []
            self.inf_Y = []
            self.inf_y = []
            self.inf_I = []
            self.inf_i = []


def main(args):
    rospy.init_node('dse_consensus_node', anonymous=True)
    est = hybrid_consensus()
    r = rospy.Rate(10)
    try:
        while True:
            r.sleep()
            est.estimate_and_send()
    except KeyboardInterrupt:
        print("Shutting down")


if __name__ == '__main__':
    main(sys.argv)
