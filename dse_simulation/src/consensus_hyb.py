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
from dse_msgs.msg import InfFilterPartials
from visualization_msgs.msg import Marker
from scipy.spatial.transform import Rotation as R
import tf_conversions
import tf2_ros
import message_filters
import copy
from threading import Thread, Lock

import dse_lib
import consensus_lib
import dse_constants

roslib.load_manifest('dse_simulation')


class hybrid_consensus:

    # Define initial/setup values
    def __init__(self):

        self.inf_subs = []
        self.inf_pubs = []
        self.inf_id = []
        self.inf_id_list = []
        self.inf_Y = []
        self.inf_y = []
        self.inf_I = []
        self.inf_i = []
        self.inf_indices = []
        self.mutex = Lock()

        # Get parameters from launch file
        self.object_names = rospy.get_param('~objects')
        self.dim_state = rospy.get_param('~dim_state', 6)
        self.comm_threshold = rospy.get_param('~comm_threshold', 5)
        self.fail_prob = rospy.get_param('~fail_prob', 0.1)
        self.r = rospy.Rate(rospy.get_param('~rate', 10))
        id_to_tf_arr = rospy.get_param('~id_to_tf')

        # self.dim_state = 6
        # self.object_names = ['tb3_0', 'tb3_1', 'tb3_2']
        # self.comm_threshold = 5
        # self.fail_prob = 0
        # self.r = rospy.Rate(10)
        # id_to_tf_arr = [[5, 'tb3_0/base_footprint'], [6, 'tb3_1/base_footprint'], [7, 'tb3_2/base_footprint']]

        self.id_to_tf = {}
        for val in id_to_tf_arr:
            self.id_to_tf[val[0]] = val[1]

        # self.id_to_tf = {
        #     5:'tb3_0/base_footprint',
        #     6:'tb3_1/base_footprint',
        #     7:'tb3_2/base_footprint'
        # }

        self.tfBuffer = tf2_ros.Buffer()
        self.listener = tf2_ros.TransformListener(self.tfBuffer)
        time.sleep(5)

        for i in range(len(self.object_names)):
            if len(self.object_names[i]) != 0 and self.object_names[i][0] != '/':
                self.object_names[i] = '/' + self.object_names[i]

            self.inf_pubs.append(rospy.Publisher(
                self.object_names[i] + "/dse/inf/results", InfFilterResults, queue_size=10))
            self.inf_subs.append(rospy.Subscriber(
                self.object_names[i] + "/dse/inf/partial", InfFilterPartials, self.information_callback, i))
        # self.agent_ids = [2000, 2001, 2002]
        self.last = None

    # def gzbo_true_callback(self, data):
    #     n = len(data.name)
    #     for i in range(n):
    #         if data.name[i] in self.object_names:
    #             index = np.where(self.object_names == data.name[i])[0][0]
    #             position = (data.pose[i].position.x, data.pose[i].position.y, data.pose[i].position.z)

    # When the information filter sends partials (prior and measurement), combine and return them
    def information_callback(self, data, agent_index):
        self.mutex.acquire()
        if agent_index in self.inf_indices:
            inf_index = np.where(np.array(self.inf_indices) == agent_index)[0][0]
            self.inf_id_list[inf_index] = copy.deepcopy(data.ids)
            self.inf_Y[inf_index] = copy.deepcopy(dse_lib.multi_array_2d_output(data.inf_matrix_prior))
            self.inf_y[inf_index] = copy.deepcopy(dse_lib.multi_array_2d_output(data.inf_vector_prior))
            self.inf_I[inf_index] = copy.deepcopy(dse_lib.multi_array_2d_output(data.obs_matrix))
            self.inf_i[inf_index] = copy.deepcopy(dse_lib.multi_array_2d_output(data.obs_vector))
        else:
            self.inf_indices.append(agent_index)
            self.inf_id.append(data.sender_id)
            self.inf_id_list.append(data.ids)
            self.inf_Y.append(dse_lib.multi_array_2d_output(data.inf_matrix_prior))
            self.inf_y.append(dse_lib.multi_array_2d_output(data.inf_vector_prior))
            self.inf_I.append(dse_lib.multi_array_2d_output(data.obs_matrix))
            self.inf_i.append(dse_lib.multi_array_2d_output(data.obs_vector))
        self.mutex.release()

    # Option 1:
    #   Create a communication graph for the whole system
    # Option 2:
    #   Split the system into isolated groups
    #   Create a graph for each group

    # When the information filter sends partials (prior and measurement), combine and return them
    def estimate_and_send(self):
        self.mutex.acquire()
        if len(self.inf_indices) > 0:
            start = rospy.Time.now().to_sec()


            agents = copy.deepcopy(self.inf_indices)
            order_to_id = copy.deepcopy(self.inf_id)
            array_ids = copy.deepcopy(self.inf_id_list)
            array_Y = copy.deepcopy(self.inf_Y)
            array_y = copy.deepcopy(self.inf_y)
            array_I = copy.deepcopy(self.inf_I)
            array_i = copy.deepcopy(self.inf_i)

            # adj = np.ones((len(self.inf_indices), len(self.inf_indices)))
            adj = consensus_lib.get_communication_graph(len(agents), order_to_id, self.id_to_tf, self.tfBuffer,
                                                        self.comm_threshold, self.fail_prob)
            print('adj graph:')
            print(adj)

            groups = consensus_lib.break_agents_into_groups(agents, order_to_id, adj, array_ids, array_Y, array_y,
                                                            array_I, array_i)

            if self.last is None:
                self.last = [None]*len(agents)

            count = 0
            for group in groups:
                [group_agents, group_order_to_id, group_adj, group_array_ids, group_array_Y, group_array_y,
                 group_array_I, group_array_i] = copy.deepcopy(group)

                np.fill_diagonal(group_adj, 0)
                # Translate that into a graph ajacency matrix
                inf_id_list, inf_Y, inf_y = consensus_lib.consensus(group_order_to_id, group_array_ids, group_array_Y,
                                                                    group_array_y, group_array_I, group_array_i,
                                                                    group_adj, self.dim_state)

                test_Y = group_array_Y[0] + group_array_I[0]
                test_y = group_array_y[0] + group_array_i[0]
                test_x = np.linalg.inv(test_Y).dot(test_y[:, 0])

                for i, agent in enumerate(group_agents):
                    inf_results = InfFilterResults()
                    inf_results.ids = inf_id_list[i]
                    inf_results.inf_matrix = dse_lib.multi_array_2d_input(inf_Y[i], inf_results.inf_matrix)
                    inf_results.inf_vector = dse_lib.multi_array_2d_input(inf_y[i], inf_results.inf_vector)
                    self.inf_pubs[agent].publish(inf_results)

                this_x = np.linalg.inv(inf_Y).dot(inf_y)[0,:,0,0]
                # if self.last[count] is not None and np.any(np.abs((this_x - self.last[count])) > 1):
                #     print('$$$$$$$$$$$$$$$$$$$$ ERROR $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$')
                self.last[count] = this_x
                count += 1

            self.inf_indices = []
            self.inf_id_list = []
            self.inf_id = []
            self.inf_Y = []
            self.inf_y = []
            self.inf_I = []
            self.inf_i = []

            current = rospy.Time.now().to_sec()
            diff = current - start
            print('consensus time: ', diff)
        self.mutex.release()


def main(args):
    rospy.init_node('dse_consensus_node', anonymous=True)
    est = hybrid_consensus()
    try:
        while True:
            est.r.sleep()
            est.estimate_and_send()
    except KeyboardInterrupt:
        print("Shutting down")


if __name__ == '__main__':
    main(sys.argv)
