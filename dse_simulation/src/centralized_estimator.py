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
        self.object_names = ['tb3_0', 'tb3_1', 'tb3_2']
        for i in range(self.n_params):
            # self.object_names.append(rospy.get_param('~objects')[i])
            self.inf_pubs.append(rospy.Publisher(self.object_names[i] + "/dse/inf/results", InfFilterResults, queue_size=10))
            self.inf_subs.append(message_filters.Subscriber(self.object_names[i] + "/dse/inf/partial", InfFilterPartials))

        print('n_params: %d' % self.n_params)
        print('state dim: %d' % self.dim_state)
        print('object names: ' + str(self.object_names))

        ts = message_filters.ApproximateTimeSynchronizer(self.inf_subs, 10, 1, allow_headerless=True)
        ts.registerCallback(self.information_callback)

    # When the information filter sends partials (prior and measurement), combine and return them
    def information_callback(self, *argv):
        print('got callback')
        # first = True
        # all_same_ids = True
        # inf_Y = dse_lib.multi_array_2d_output(argv[0].inf_matrix_prior)
        # inf_y = dse_lib.multi_array_2d_output(argv[0].inf_vector_prior)
        # inf_I = dse_lib.multi_array_2d_output(argv[0].obs_matrix)
        # inf_i = dse_lib.multi_array_2d_output(argv[0].obs_vector)
        # inf_id_list = argv[0].ids
        array_ids = []
        array_Y = []
        array_y = []
        array_I = []
        array_i = []

        for i in range(len(argv)):
            arg = argv[i]
            array_ids.append(arg.ids)
            array_Y.append(dse_lib.multi_array_2d_output(arg.inf_matrix_prior))
            array_y.append(dse_lib.multi_array_2d_output(arg.inf_vector_prior))
            array_I.append(dse_lib.multi_array_2d_output(arg.obs_matrix))
            array_i.append(dse_lib.multi_array_2d_output(arg.obs_vector))
            # inf_id_list, inf_Y, inf_y, P_11, x_11 = dse_lib.extend_arrays(arg.ids, inf_id_list, inf_Y, inf_y, self.dim_state)
            # rcvd_I = dse_lib.multi_array_2d_output(arg.obs_matrix)
            # rcvd_i = dse_lib.multi_array_2d_output(arg.obs_vector)
            # inf_I = np.add(rcvd_I, inf_I)
            # inf_i = np.add(rcvd_i, inf_i)

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

        # inf_id_list = data.ids
        # inf_Y = dse_lib.multi_array_2d_output(data.inf_matrix_prior)
        # inf_y = dse_lib.multi_array_2d_output(data.inf_vector_prior)
        # inf_I = dse_lib.multi_array_2d_output(data.obs_matrix)
        # inf_i = dse_lib.multi_array_2d_output(data.obs_vector)
        #
        # inf_Y = inf_Y + inf_I
        # inf_y = inf_y + inf_i
        #
        # inf_results = InfFilterResults()
        # inf_results.ids = inf_id_list
        # inf_results.inf_matrix = dse_lib.multi_array_2d_input(inf_Y, inf_results.inf_matrix)
        # inf_results.inf_vector = dse_lib.multi_array_2d_input(inf_y, inf_results.inf_vector)
        # self.inf_pubs[index].publish(inf_results)


def main(args):

    rospy.init_node('dse_centralized_estimator_node', anonymous=True)
    vis = central_est()
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")


if __name__ == '__main__':
    main(sys.argv)
