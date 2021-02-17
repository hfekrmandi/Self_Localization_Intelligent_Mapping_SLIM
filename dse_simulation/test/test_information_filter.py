#!/usr/bin/env python
from __future__ import print_function

import roslib
import os
import sys
import unittest
import rospy
import rostest
from optparse import OptionParser
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
import copy

sys.path.append(os.path.join(sys.path[0], "../src"))
import dse_lib
import consensus_lib

PKG = 'dse_simulation'
roslib.load_manifest(PKG)

##############################################################################
##############################################################################
class TestInformationFilterCommon(unittest.TestCase):
    ##############################################################################
    ##############################################################################
    # def set_up(self):
    ##############################################################################
    def __init__(self, *args):
        ##############################################################################
        # rospy.loginfo("-D- TestRangeFilter __init__")
        # super(TestRangeFilterCommon, self).__init__(*args)
        self.set_up()
        super(TestInformationFilterCommon, self).__init__(*args)

    ##############################################################################
    def set_up(self):
        ##############################################################################
        rospy.init_node("test_observation_jacobian")
        # self.coefficient = rospy.get_param("range_filter/coefficient", 266)
        # self.exponent = rospy.get_param("range_filter/exponent", -1.31)
        # self.rolling_pts = rospy.get_param("range_filter/rolling_pts", 4)
        self.test_rate = rospy.get_param("~test_rate", 100)
        self.results_sub = rospy.Subscriber("/tb3_0/dse/inf/results", InfFilterResults, self.estimator_results_callback)
        self.inf_pub = rospy.Publisher("/tb3_0/dse/inf/partial", InfFilterPartials, queue_size=10)

        # self.latest_filtered = 1e10
        # self.latest_std = 2e10
        self.dim_state = 6
        self.dim_obs = 3
        self.euler_order = 'zyx'
        self.got_callback = False

    ##############################################################################
    def send_poses(self, poses, rate):
        ##############################################################################
        r = rospy.Rate(rate)
        # rospy.loginfo("-D- sendmsgs: sending %s" % str(msgs))
        for pose in poses:
            rospy.loginfo("-D- publishing %d" % pose)
            self.pose_pub.publish()
            r.sleep()

    # When the information filter sends back results, store them locally
    def information_callback(self, data):
        rospy.loginfo("-D- information_filter.py sent back data")
        inf_id_list = data.ids
        self.inf_Y_prior = dse_lib.multi_array_2d_output(data.inf_matrix_prior)
        self.inf_y_prior = dse_lib.multi_array_2d_output(data.inf_vector_prior)
        self.inf_I = dse_lib.multi_array_2d_output(data.obs_matrix)
        self.inf_i = dse_lib.multi_array_2d_output(data.obs_vector)

    # When the direct estimator or consensus returns the combined information variables
    def estimator_results_callback(self, data):
        rospy.loginfo("-D- information_filter.py sent back data")
        self.inf_id_list = np.array(data.ids)
        self.inf_Y = dse_lib.multi_array_2d_output(data.inf_matrix)
        self.inf_y = dse_lib.multi_array_2d_output(data.inf_vector)
        self.got_callback = True


##############################################################################
##############################################################################
class TestInformationFilterValid(TestInformationFilterCommon):
    ##############################################################################
    ##############################################################################

    ##############################################################################
    def test_one_Equal_one(self):
        ##############################################################################
        rospy.loginfo("-D- test_one_Equal_one")
        self.assertEqual(1, 1, "1!=1")

    def test_theta_2_rotm_zero(self):
        ##############################################################################
        rospy.loginfo("-D- test_theta_2_rotm_0")
        rotm = dse_lib.theta_2_rotm(0)
        x_0 = np.transpose([1, 2])
        x_rotm = rotm.dot(x_0)
        x_true = x_0
        self.assertEqual(True, np.allclose(x_true, x_rotm))

    def test_theta_2_rotm_90(self):
        ##############################################################################
        rospy.loginfo("-D- test_theta_2_rotm_0")
        theta = 90
        rotm = dse_lib.theta_2_rotm(theta * np.pi / 180.0)
        x_0 = np.transpose([1, 2])
        x_rotm = rotm.dot(x_0)
        x_true = np.transpose([-2, 1])
        self.assertEqual(True, np.allclose(x_true, x_rotm))

    def test_theta_2_rotm_45(self):
        ##############################################################################
        rospy.loginfo("-D- test_theta_2_rotm_0")
        theta = 45
        rotm = dse_lib.theta_2_rotm(theta * np.pi / 180.0)
        x_0 = np.transpose([1, 1])
        x_rotm = rotm.dot(x_0)
        x_true = np.transpose([0, np.sqrt(2)])
        self.assertEqual(True, np.allclose(x_true, x_rotm))

    def test_to_frame_1(self):
        ##############################################################################
        rospy.loginfo("-D- test_from_frame_1")
        agent1_global = np.array([[0], [0], [np.pi]])
        agent2_global = np.array([[1], [0], [0]])
        agent2_in_frame_agent1_true = np.array([[-1], [0], [np.pi]])
        agent1_in_frame_agent2_true = np.array([[-1], [0], [np.pi]])

        agent2_in_frame_agent1_est = dse_lib.agent2_to_frame_agent1_3D(agent1_global, agent2_global)
        agent1_in_frame_agent2_est = dse_lib.agent2_to_frame_agent1_3D(agent2_global, agent1_global)

        if agent2_in_frame_agent1_est[2, 0] < 0:
            agent2_in_frame_agent1_est[2, 0] += 2*np.pi
        if agent1_in_frame_agent2_est[2, 0] < 0:
            agent1_in_frame_agent2_est[2, 0] += 2*np.pi

        self.assertEqual(True, np.allclose(agent2_in_frame_agent1_true, agent2_in_frame_agent1_est))
        self.assertEqual(True, np.allclose(agent1_in_frame_agent2_true, agent1_in_frame_agent2_est))

    def test_to_frame_2(self):
        ##############################################################################
        rospy.loginfo("-D- test_from_frame_1")
        agent1_global = np.array([[0], [0], [0]])
        agent2_global = np.array([[-1], [1], [0]])
        agent2_in_frame_agent1_true = np.array([[-1], [1], [0]])
        agent1_in_frame_agent2_true = np.array([[1], [-1], [0]])

        agent2_in_frame_agent1_est = dse_lib.agent2_to_frame_agent1_3D(agent1_global, agent2_global)
        agent1_in_frame_agent2_est = dse_lib.agent2_to_frame_agent1_3D(agent2_global, agent1_global)

        if agent2_in_frame_agent1_est[2, 0] < 0:
            agent2_in_frame_agent1_est[2, 0] += 2*np.pi
        if agent1_in_frame_agent2_est[2, 0] < 0:
            agent1_in_frame_agent2_est[2, 0] += 2*np.pi

        self.assertEqual(True, np.allclose(agent2_in_frame_agent1_true, agent2_in_frame_agent1_est))
        self.assertEqual(True, np.allclose(agent1_in_frame_agent2_true, agent1_in_frame_agent2_est))

    def test_to_frame_3(self):
        ##############################################################################
        rospy.loginfo("-D- test_from_frame_1")
        agent1_global = np.array([[0], [0], [np.pi]])
        agent2_global = np.array([[1], [0], [np.pi/2]])
        agent2_in_frame_agent1_true = np.array([[-1], [0], [3*np.pi/2]])
        agent1_in_frame_agent2_true = np.array([[0], [1], [np.pi/2]])

        agent2_in_frame_agent1_est = dse_lib.agent2_to_frame_agent1_3D(agent1_global, agent2_global)
        agent1_in_frame_agent2_est = dse_lib.agent2_to_frame_agent1_3D(agent2_global, agent1_global)

        if agent2_in_frame_agent1_est[2, 0] < 0:
            agent2_in_frame_agent1_est[2, 0] += 2*np.pi
        if agent1_in_frame_agent2_est[2, 0] < 0:
            agent1_in_frame_agent2_est[2, 0] += 2*np.pi

        self.assertEqual(True, np.allclose(agent2_in_frame_agent1_true, agent2_in_frame_agent1_est))
        self.assertEqual(True, np.allclose(agent1_in_frame_agent2_true, agent1_in_frame_agent2_est))

    def test_to_frame_4(self):
        ##############################################################################
        rospy.loginfo("-D- test_from_frame_1")
        agent1_global = np.array([[1], [1], [7/4.0*np.pi]])
        agent2_global = np.array([[0.4], [-0.6], [5/4.0*np.pi]])
        agent2_in_frame_agent1_true = np.array([[0.5*np.sqrt(2)], [-1.1*np.sqrt(2)], [3/2.0*np.pi]])
        agent1_in_frame_agent2_true = np.array([[-1.1*np.sqrt(2)], [-0.5*np.sqrt(2)], [1/2.0*np.pi]])

        agent2_in_frame_agent1_est = dse_lib.agent2_to_frame_agent1_3D(agent1_global, agent2_global)
        agent1_in_frame_agent2_est = dse_lib.agent2_to_frame_agent1_3D(agent2_global, agent1_global)

        if agent2_in_frame_agent1_est[2, 0] < 0:
            agent2_in_frame_agent1_est[2, 0] += 2*np.pi
        if agent1_in_frame_agent2_est[2, 0] < 0:
            agent1_in_frame_agent2_est[2, 0] += 2*np.pi

        self.assertEqual(True, np.allclose(agent2_in_frame_agent1_true, agent2_in_frame_agent1_est))
        self.assertEqual(True, np.allclose(agent1_in_frame_agent2_true, agent1_in_frame_agent2_est))

    def test_from_frame_0(self):
        ##############################################################################
        rospy.loginfo("-D- test_from_frame_1")
        agent1_global = np.array([[1], [1], [7/4.0*np.pi]])
        agent2_global = np.array([[0.4], [-0.6], [5/4.0*np.pi]])
        agent2_in_frame_agent1_true = np.array([[0.5*np.sqrt(2)], [-1.1*np.sqrt(2)], [3/2.0*np.pi]])
        agent1_in_frame_agent2_true = np.array([[-1.1*np.sqrt(2)], [-0.5*np.sqrt(2)], [1/2.0*np.pi]])

        agent2_in_frame_agent1_est = dse_lib.agent2_to_frame_agent1_3D(agent1_global, agent2_global)
        agent2_global_est = dse_lib.agent2_from_frame_agent1_3D(agent1_global, agent2_in_frame_agent1_est)

        if agent2_global_est[2, 0] < 0:
            agent2_global_est[2, 0] += 2*np.pi

        self.assertEqual(True, np.allclose(agent2_global, agent2_global_est))

    def test_from_frame_1(self):
        ##############################################################################
        rospy.loginfo("-D- test_from_frame_1")

        # 1 is fixed, 2 is this, 3 is object
        agent1_global = np.array([[1], [1], [7 / 4.0 * np.pi]])
        agent2_global = np.array([[0.4], [-0.6], [5 / 4.0 * np.pi]])
        agent3_global = np.array([[1], [0], [np.pi/2]])

        agent1_in_frame_agent2 = dse_lib.agent2_to_frame_agent1_3D(agent2_global, agent1_global)
        agent1_in_frame_agent3 = dse_lib.agent2_to_frame_agent1_3D(agent3_global, agent1_global)
        agent2_in_frame_agent1 = dse_lib.agent2_to_frame_agent1_3D(agent1_global, agent2_global)
        agent2_in_frame_agent3 = dse_lib.agent2_to_frame_agent1_3D(agent3_global, agent2_global)
        agent3_in_frame_agent1 = dse_lib.agent2_to_frame_agent1_3D(agent1_global, agent3_global)
        agent3_in_frame_agent2 = dse_lib.agent2_to_frame_agent1_3D(agent2_global, agent3_global)

        z_true = agent3_in_frame_agent2
        x = agent3_in_frame_agent1

        agent2_global_est = dse_lib.agent2_from_frame_agent1_3D(agent1_global, agent2_in_frame_agent1)
        agent3_global_est = dse_lib.agent2_from_frame_agent1_3D(agent1_global, agent3_in_frame_agent1)
        z_est = dse_lib.agent2_to_frame_agent1_3D(agent2_global_est, agent3_global_est)

        z_est_2 = dse_lib.agent2_to_frame_agent1_3D(agent2_in_frame_agent1, agent3_in_frame_agent1)

        self.assertEqual(True, np.allclose(z_true, z_est))
        self.assertEqual(True, np.allclose(z_true, z_est_2))

    def test_dual_relative_obs_jacobian_3D_0(self):
        ##############################################################################
        rospy.loginfo("-D- test_from_frame_1")

        # 1 is fixed, 2 is this, 3 is object
        agent1_global = np.array([[1], [1], [7 / 4.0 * np.pi]])
        agent2_global = np.array([[0.4], [-0.6], [5 / 4.0 * np.pi]])
        agent3_global = np.array([[1.5], [0321], [np.pi/2]])

        agent1_in_frame_agent2 = dse_lib.agent2_to_frame_agent1_3D(agent2_global, agent1_global)
        agent1_in_frame_agent3 = dse_lib.agent2_to_frame_agent1_3D(agent3_global, agent1_global)
        agent2_in_frame_agent1 = dse_lib.agent2_to_frame_agent1_3D(agent1_global, agent2_global)
        agent2_in_frame_agent3 = dse_lib.agent2_to_frame_agent1_3D(agent3_global, agent2_global)
        agent3_in_frame_agent1 = dse_lib.agent2_to_frame_agent1_3D(agent1_global, agent3_global)
        agent3_in_frame_agent2 = dse_lib.agent2_to_frame_agent1_3D(agent2_global, agent3_global)

        z_true = agent3_in_frame_agent2

        z_fun = dse_lib.agent2_to_frame_agent1_3D(agent2_global, agent3_global)

        H = np.array(dse_lib.dual_relative_obs_jacobian_3D(agent2_global, agent3_global))
        x = np.append(agent2_global, agent3_global)[:, None]
        z_h = H.dot(x)
        z_h = np.array([z_h[0][0][0], z_h[1][0][0], z_h[2][0]])[:, None]

        self.assertEqual(True, np.allclose(z_true, z_fun))
        self.assertEqual(True, np.allclose(z_true, z_h))

    def test_jacobian_fixed_to_obs_3D_0(self):
        ##############################################################################
        rospy.loginfo("-D- test_from_frame_1")

        # 1 is fixed, 2 is this, 3 is object
        agent1_global = np.array([[1], [1], [7 / 4.0 * np.pi]])
        agent2_global = np.array([[0.4], [-0.6], [5 / 4.0 * np.pi]])
        agent3_global = np.array([[1.5], [0321], [30.1234*np.pi/2]])

        agent1_in_frame_agent2 = dse_lib.agent2_to_frame_agent1_3D(agent2_global, agent1_global)
        agent1_in_frame_agent3 = dse_lib.agent2_to_frame_agent1_3D(agent3_global, agent1_global)
        agent2_in_frame_agent1 = dse_lib.agent2_to_frame_agent1_3D(agent1_global, agent2_global)
        agent2_in_frame_agent3 = dse_lib.agent2_to_frame_agent1_3D(agent3_global, agent2_global)
        agent3_in_frame_agent1 = dse_lib.agent2_to_frame_agent1_3D(agent1_global, agent3_global)
        agent3_in_frame_agent2 = dse_lib.agent2_to_frame_agent1_3D(agent2_global, agent3_global)

        z_true = agent3_in_frame_agent2
        x = agent3_in_frame_agent1

        z_fun = dse_lib.agent2_to_frame_agent1_3D(agent2_in_frame_agent1, agent3_in_frame_agent1)

        H = np.array(dse_lib.jacobian_fixed_to_obs_3D(agent2_in_frame_agent1, agent3_in_frame_agent1))
        x = np.append(agent2_in_frame_agent1, agent3_in_frame_agent1)[:, None]
        z_h = H.dot(x)
        z_h = np.array([z_h[0][0][0], z_h[1][0][0], z_h[2][0]])[:, None]

        self.assertEqual(True, np.allclose(z_true, z_fun))
        self.assertEqual(True, np.allclose(z_true, z_h))

    def test_observation_jacobian_zeros(self):
        ##############################################################################
        rospy.loginfo("-D- test_observation_jacobian_0")
        agent1 = 0
        agent2 = 1
        x = np.zeros((12, 1))
        H = np.zeros((3, 12))
        H = dse_lib.h_camera_3D(H, x, 0, agent1, agent2, self.dim_state, self.dim_obs)
        z_jac = H.dot(x)

        agent1_row_min = self.dim_state * agent1
        agent1_row_max = agent1_row_min + self.dim_obs
        agent2_row_min = self.dim_state * agent2
        agent2_row_max = agent2_row_min + self.dim_obs
        x1 = x[agent1_row_min:agent1_row_max]
        t1 = x1[0:2]
        R1 = dse_lib.theta_2_rotm(x1[2, 0])

        x2 = x[agent2_row_min:agent2_row_max]
        t2 = x2[0:2]
        R2 = dse_lib.theta_2_rotm(x2[2, 0])

        zt = (np.transpose(R1).dot(t2) - np.transpose(R1).dot(t1))[:, 0]
        zR = np.transpose(R1).dot(R2)
        zr = [-np.arccos(zR[0, 0])]
        z_true = np.concatenate((zt, zr))[:, None]

        rospy.loginfo("-D- z_jac (%d, %d)" % (np.shape(z_jac)[0], np.shape(z_jac)[1]))
        rospy.loginfo("-D- z_jac (%d, %d)" % (np.shape(z_true)[0], np.shape(z_true)[1]))
        self.assertEqual(True, np.allclose(z_true, z_jac))

    def test_observation_jacobian_translation(self):
        ##############################################################################
        rospy.loginfo("-D- test_observation_jacobian_0")
        agent1 = 0
        agent2 = 1
        x = np.transpose([1, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])[:, None]
        H = np.zeros((3, 12))
        H = dse_lib.h_camera_3D(H, x, 0, agent1, agent2, self.dim_state, self.dim_obs)
        z_jac = H.dot(x)

        agent1_row_min = self.dim_state * agent1
        agent1_row_max = agent1_row_min + self.dim_obs
        agent2_row_min = self.dim_state * agent2
        agent2_row_max = agent2_row_min + self.dim_obs
        x1 = x[agent1_row_min:agent1_row_max]
        t1 = x1[0:2]
        R1 = dse_lib.theta_2_rotm(x1[2, 0])

        x2 = x[agent2_row_min:agent2_row_max]
        t2 = x2[0:2]
        R2 = dse_lib.theta_2_rotm(x2[2, 0])

        zt = (np.transpose(R1).dot(t2) - np.transpose(R1).dot(t1))[:, 0]
        zR = np.transpose(R1).dot(R2)
        zr = [-np.arccos(zR[0, 0])]
        z_true = np.concatenate((zt, zr))[:, None]

        rospy.loginfo("-D- z_jac (%d, %d)" % (np.shape(z_jac)[0], np.shape(z_jac)[1]))
        rospy.loginfo("-D- z_jac (%d, %d)" % (np.shape(z_true)[0], np.shape(z_true)[1]))
        self.assertEqual(True, np.allclose(z_true, z_jac))

    def test_observation_jacobian_translation_rotation(self):
        ##############################################################################
        rospy.loginfo("-D- test_observation_jacobian_0")
        agent1 = 0
        agent2 = 1
        x = np.transpose([1, 2, np.pi/2, 0, 0, 0, 0, 0, -np.pi/2, 0, 0, 0])[:, None]
        H = np.zeros((3, 12))
        H = dse_lib.h_camera_3D(H, x, 0, agent1, agent2, self.dim_state, self.dim_obs)
        z_jac = H.dot(x)

        agent1_row_min = self.dim_state * agent1
        agent1_row_max = agent1_row_min + self.dim_obs
        agent2_row_min = self.dim_state * agent2
        agent2_row_max = agent2_row_min + self.dim_obs
        x1 = x[agent1_row_min:agent1_row_max]
        t1 = x1[0:2]
        R1 = dse_lib.theta_2_rotm(x1[2, 0])

        x2 = x[agent2_row_min:agent2_row_max]
        t2 = x2[0:2]
        R2 = dse_lib.theta_2_rotm(x2[2, 0])

        zt = (np.transpose(R1).dot(t2) - np.transpose(R1).dot(t1))[:, 0]
        zR = np.transpose(R1).dot(R2)
        zr = [-np.arccos(zR[0, 0])]
        z_true = np.concatenate((zt, zr))[:, None]

        self.assertEqual(True, np.allclose(z_true, z_jac))

    def test_extend_arrays_no_extension(self):
        ##############################################################################
        rospy.loginfo("-D- test_extend_arrays_0")

        dim_state = 12

        id_list = np.arange(5)
        observed_ids = id_list
        n_ids = len(id_list)

        Y_11 = np.eye((dim_state * n_ids))
        y_11 = np.ones((dim_state * n_ids, 1))
        x_11 = np.linalg.inv(Y_11).dot(y_11)
        P_11 = np.linalg.inv(Y_11)

        id_list_2, Y_11_2, y_11_2, P_11_2, x_11_2 = dse_lib.extend_arrays(observed_ids, id_list, Y_11, y_11, dim_state)

        self.assertEqual(True, np.allclose(P_11, P_11_2))
        self.assertEqual(True, np.allclose(x_11, x_11_2))
        self.assertEqual(True, np.allclose(Y_11, Y_11_2))
        self.assertEqual(True, np.allclose(y_11, y_11_2))
        self.assertEqual(True, np.allclose(id_list, id_list_2))

    def test_extend_arrays_add_1(self):
        ##############################################################################
        rospy.loginfo("-D- test_extend_arrays_0")

        dim_state = 12

        id_list = np.arange(5)
        observed_ids = np.arange(6)
        n_ids = len(id_list)

        Y_11 = np.eye((dim_state * n_ids))
        y_11 = np.ones((dim_state * n_ids, 1))
        x_11 = np.linalg.inv(Y_11).dot(y_11)
        P_11 = np.linalg.inv(Y_11)

        id_list_2, Y_11_2, y_11_2, P_11_2, x_11_2 = dse_lib.extend_arrays(observed_ids, id_list, Y_11, y_11, dim_state)

        self.assertNotEqual(True, np.allclose(np.shape(P_11), np.shape(P_11_2)))
        self.assertNotEqual(True, np.allclose(np.shape(x_11), np.shape(x_11_2)))
        self.assertNotEqual(True, np.allclose(np.shape(Y_11), np.shape(Y_11_2)))
        self.assertNotEqual(True, np.allclose(np.shape(y_11), np.shape(y_11_2)))
        self.assertNotEqual(True, np.allclose(np.shape(id_list), np.shape(id_list_2)))

        x_11_3 = np.linalg.inv(Y_11_2).dot(y_11_2)
        P_11_3 = np.linalg.inv(Y_11_2)
        self.assertEqual(True, np.allclose(P_11_3, P_11_2))
        self.assertEqual(True, np.allclose(x_11_3, x_11_2))

        state_dim = dim_state * len(observed_ids)
        self.assertEqual(state_dim, np.shape(Y_11_2)[0])
        self.assertEqual(state_dim, np.shape(Y_11_2)[1])
        self.assertEqual(state_dim, np.shape(y_11_2)[0])
        self.assertEqual(len(observed_ids), len(id_list_2))

    def test_sort_arrays_0(self):
        ##############################################################################
        rospy.loginfo("-D- test_extend_arrays_0")

        dim_state = 2

        id_list = np.arange(5)

        id_list = []
        id_list.append([0, 1])
        id_list.append([1, 0])
        id_list.append([0])

        # Starting data
        Y_11 = []
        Y_11.append(np.array([[0, 1, 2, 3], [1, 2, 3, 4], [2, 3, 4, 5], [3, 4, 5, 6]]))
        Y_11.append(np.array([[3, 2, 1, 0], [4, 3, 2, 1], [5, 4, 6, 8], [6, 5, 4, 3]]))
        Y_11.append(np.array([[2, 6], [4, 3]]))

        y_11 = []
        y_11.append(np.array([0, 1, 2, 3])[:, None])
        y_11.append(np.array([3, 2, 1, 0])[:, None])
        y_11.append(np.array([2, 6])[:, None])

        I_11 = []
        I_11.append(np.array([[0, 1, 2, 3], [1, 2, 3, 4], [2, 3, 4, 5], [3, 4, 5, 6]]))
        I_11.append(np.array([[3, 2, 1, 0], [4, 3, 2, 1], [5, 4, 6, 8], [6, 5, 4, 3]]))
        I_11.append(np.array([[2, 6], [4, 3]]))

        i_11 = []
        i_11.append(np.array([0, 1, 2, 3])[:, None])
        i_11.append(np.array([3, 2, 1, 0])[:, None])
        i_11.append(np.array([2, 6])[:, None])

        # True result data
        id_list_true = [0, 1]

        Y_11_true = []
        Y_11_true.append(np.array([[0, 1, 2, 3], [1, 2, 3, 4], [2, 3, 4, 5], [3, 4, 5, 6]]))
        Y_11_true.append(np.array([[6, 8, 5, 4], [4, 3, 6, 5], [1, 0, 3, 2], [2, 1, 4, 3]]))
        Y_11_true.append(np.array([[2, 6, 0, 0], [4, 3, 0, 0], [0, 0, 0.01, 0], [0, 0, 0, 0.01]]))

        y_11_true = []
        y_11_true.append(np.array([0, 1, 2, 3])[:, None])
        y_11_true.append(np.array([1, 0, 3, 2])[:, None])
        y_11_true.append(np.array([2, 6, 0, 0])[:, None])

        I_11_true = []
        I_11_true.append(np.array([[0, 1, 2, 3], [1, 2, 3, 4], [2, 3, 4, 5], [3, 4, 5, 6]]))
        I_11_true.append(np.array([[6, 8, 5, 4], [4, 3, 6, 5], [1, 0, 3, 2], [2, 1, 4, 3]]))
        I_11_true.append(np.array([[2, 6, 0, 0], [4, 3, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]))

        i_11_true = []
        i_11_true.append(np.array([0, 1, 2, 3])[:, None])
        i_11_true.append(np.array([1, 0, 3, 2])[:, None])
        i_11_true.append(np.array([2, 6, 0, 0])[:, None])

        array_ids, array_Y, array_y, array_I, array_i = dse_lib.get_sorted_agent_states(
            id_list, Y_11, y_11, I_11, i_11, dim_state)

        self.assertEqual(True, np.allclose(id_list_true, array_ids[0]))
        self.assertEqual(True, np.allclose(id_list_true, array_ids[1]))
        self.assertEqual(True, np.allclose(id_list_true, array_ids[2]))

        for i in range(3):
            self.assertEqual(True, np.allclose(Y_11_true[i], array_Y[i]))
            self.assertEqual(True, np.allclose(y_11_true[i], array_y[i]))
            self.assertEqual(True, np.allclose(I_11_true[i], array_I[i]))
            self.assertEqual(True, np.allclose(i_11_true[i], array_i[i]))

    def test_centralized_estimator_0(self):
        ##############################################################################
        rospy.loginfo("-D- test_centralized_estimator_0")

        state_dim = 6
        num_agents = 3
        inf_ids = np.arange(3)[:, None]
        inf_Y = np.eye(state_dim*num_agents)
        inf_y = np.ones((state_dim*num_agents, 1))
        inf_I = np.eye(state_dim*num_agents)
        inf_i = np.ones((state_dim*num_agents, 1))
        # inf_Y = np.random.rand(state_dim*num_agents, state_dim*num_agents)
        # inf_y = np.random.rand(state_dim*num_agents, 1)
        # inf_I = np.random.rand(state_dim*num_agents, state_dim*num_agents)
        # inf_i = np.random.rand(state_dim*num_agents, 1)

        target_Y = np.add(inf_Y, inf_I)
        target_y = np.add(inf_y, inf_i)

        # Write the consensus variables to the publisher
        inf_partial = InfFilterPartials()
        inf_partial.ids = inf_ids
        inf_partial.inf_matrix_prior = dse_lib.multi_array_2d_input(inf_Y, inf_partial.inf_matrix_prior)
        inf_partial.inf_vector_prior = dse_lib.multi_array_2d_input(inf_y, inf_partial.inf_vector_prior)
        inf_partial.obs_matrix = dse_lib.multi_array_2d_input(inf_I, inf_partial.obs_matrix)
        inf_partial.obs_vector = dse_lib.multi_array_2d_input(inf_i, inf_partial.obs_vector)
        self.inf_pub.publish(inf_partial)

        # r = rospy.Rate(10)
        # while not self.got_callback:
        #     r.sleep()

        # target_Y = np.add(inf_Y, inf_I)
        # target_y = np.add(inf_y, inf_i)
        #
        # self.assertEqual(True, np.allclose(inf_ids, self.inf_id_list))
        # self.assertEqual(True, np.allclose(target_Y, self.inf_Y))
        # self.assertEqual(True, np.allclose(target_y, self.inf_y))

    # def test_relative_states_from_global_3D_0(self):
    #     ##############################################################################
    #     rospy.loginfo("-D- test_relative_states_from_global_3D_0")
    #
    #     dim_state = 6
    #     dim_obs = 3
    #
    #     id_list = np.arange(5)
    #     n_ids = len(id_list)
    #     our_id = 1
    #     states = np.random.rand(dim_state * n_ids, 1)
    #     states[dim_state*our_id:dim_state*(our_id+1)] = np.zeros((dim_state, 1))
    #
    #     obs_ids, obs_states = dse_lib.relative_states_from_global_3D(our_id, id_list, states, dim_state, dim_obs)
    #     tmp = 0

    def test_consensus_1_agent(self):
        ##############################################################################
        rospy.loginfo("-D- test_consensus_0")

        state_dim = 6
        num_agents = 3
        inf_ids = np.arange(3)[:, None]
        inf_Y = np.random.rand(state_dim*num_agents, state_dim*num_agents)
        inf_y = np.random.rand(state_dim*num_agents, 1)
        inf_I = np.random.rand(state_dim*num_agents, state_dim*num_agents)
        inf_i = np.random.rand(state_dim*num_agents, 1)

        target_Y = np.add(inf_Y, inf_I)
        target_y = np.add(inf_y, inf_i)

        # Write the consensus variables to the publisher
        inf_partial = InfFilterPartials()
        inf_partial.ids = inf_ids
        inf_partial.inf_matrix_prior = dse_lib.multi_array_2d_input(inf_Y, inf_partial.inf_matrix_prior)
        inf_partial.inf_vector_prior = dse_lib.multi_array_2d_input(inf_y, inf_partial.inf_vector_prior)
        inf_partial.obs_matrix = dse_lib.multi_array_2d_input(inf_I, inf_partial.obs_matrix)
        inf_partial.obs_vector = dse_lib.multi_array_2d_input(inf_i, inf_partial.obs_vector)
        self.inf_pub.publish(inf_partial)

        # r = rospy.Rate(10)
        # while not self.got_callback:
        #     r.sleep()

        # target_Y = np.add(inf_Y, inf_I)
        # target_y = np.add(inf_y, inf_i)
        #
        # self.assertEqual(True, np.allclose(inf_ids, self.inf_id_list))
        # self.assertEqual(True, np.allclose(target_Y, self.inf_Y))
        # self.assertEqual(True, np.allclose(target_y, self.inf_y))

    def test_networkComponents_0(self):
        ##############################################################################
        rospy.loginfo("-D- test_networkComponents_0")

        adj = np.array([[1, 1, 0], [1, 1, 0], [0, 0, 1]])
        true_sizes = [2, 2, 1]
        true_nComponents = 2
        true_members = [[0, 1], 2]

        sizes, nComponents, members = consensus_lib.networkComponents(adj)

        self.assertEqual(True, np.allclose(true_sizes, sizes))
        self.assertEqual(True, np.allclose(true_nComponents, nComponents))
        for i in range(nComponents):
            self.assertEqual(True, np.allclose(true_members[i], members[i]))


    def test_networkComponents_1(self):
        ##############################################################################
        rospy.loginfo("-D- test_networkComponents_1")

        adj = np.array([[1, 1, 0], [1, 1, 1], [0, 1, 1]])
        true_sizes = [3, 3, 3]
        true_nComponents = 1
        true_members = [[0, 1, 2]]

        sizes, nComponents, members = consensus_lib.networkComponents(adj)

        self.assertEqual(True, np.allclose(true_sizes, sizes))
        self.assertEqual(True, np.allclose(true_nComponents, nComponents))
        for i in range(nComponents):
            self.assertEqual(True, np.allclose(true_members[i], members[i]))


    def test_consensus_0(self):
        ##############################################################################
        rospy.loginfo("-D- test_consensus_0")

        # INPUTS
        #   order_to_id = useless, []
        #   array_ids = array of all known ids, ordered for Y, y, I, i, can be empty
        #   array_Y = dim_state*n ^ 2, identity
        #   array_y = dim_state*n, 1s
        #   array_I = dim_state*n ^ 2, identity
        #   array_i = dim_state*n, 1s
        #   array_comm = [[1, 1, 0], [1, 1, 1], [0, 1, 1]]
        #   self.dim_state = 12
        # OUTPUTS
        #   inf_id_list
        #   inf_Y
        #   inf_y
        order_to_id = []
        array_ids = []
        array_Y = []
        array_y = []
        array_I = []
        array_i = []
        for i in range(3):
            array_Y.append(np.eye(36))
            array_y.append(np.ones((36, 1)))
            array_I.append(np.eye(36))
            array_i.append(np.ones((36, 1)))
        adj = np.array([[1, 1, 0], [1, 1, 1], [0, 1, 1]])
        dim_state = 12
        inf_id_list, inf_Y, inf_y = consensus_lib.consensus(order_to_id, array_ids, array_Y, array_y, array_I, array_i, adj, dim_state)
        print('hi')


if __name__ == '__main__':
    rospy.loginfo("-I- test_information_filter started")
    rospy.loginfo("-D- sys.argv: %s" % str(sys.argv))
    rostest.rosrun(PKG, 'test_information_filter_valid', TestInformationFilterValid, sys.argv)
