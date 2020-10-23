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

sys.path.append(os.path.join(sys.path[0], "../src"))
import dse_lib

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
        # self.latest_filtered = 1e10
        # self.latest_std = 2e10
        self.pose_pub = rospy.Publisher("/dse/pose_markers", PoseMarkers, queue_size=10)
        self.inf_sub = rospy.Subscriber("/dse/inf/partial", InfFilterPartials, self.information_callback)
        self.dim_state = 6
        self.dim_obs = 3
        self.euler_order = 'zyx'

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
        rospy.loginfo("-D- information_filter sent back data")
        inf_id_list = data.ids
        self.inf_Y_prior = dse_lib.multi_array_2d_output(data.inf_matrix_prior)
        self.inf_y_prior = dse_lib.multi_array_2d_output(data.inf_vector_prior)
        self.inf_I = dse_lib.multi_array_2d_output(data.obs_matrix)
        self.inf_i = dse_lib.multi_array_2d_output(data.obs_vector)


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

    def test_from_to_from_frame_1(self):
        ##############################################################################
        rospy.loginfo("-D- test_from_frame_1")
        agent1_global = np.array([[0.5], [-7], [2.587394]])
        agent2_global = np.array([[-6], [-1.42], [5.234]])

        agent2_in_frame_agent1_est = dse_lib.agent2_to_frame_agent1_3D(agent1_global, agent2_global)
        agent1_in_frame_agent2_est = dse_lib.agent2_to_frame_agent1_3D(agent2_global, agent1_global)

        agent1_global_est = dse_lib.agent2_from_frame_agent1_3D(agent2_global, agent1_in_frame_agent2_est)
        agent2_global_est = dse_lib.agent2_from_frame_agent1_3D(agent1_global, agent2_in_frame_agent1_est)

        self.assertEqual(True, np.allclose(agent1_global, agent1_global_est))
        self.assertEqual(True, np.allclose(agent2_global, agent2_global_est))

    def test_observation_jacobian_zeros(self):
        ##############################################################################
        rospy.loginfo("-D- test_observation_jacobian_0")
        agent1 = 0
        agent2 = 1
        x = np.zeros((12, 1))
        H = np.zeros((3, 12))
        H = dse_lib.h_camera_3D(H, x, agent1, agent2, self.dim_state, self.dim_obs)
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
        zr = [np.arccos(zR[0, 0])]
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
        H = dse_lib.h_camera_3D(H, x, agent1, agent2, self.dim_state, self.dim_obs)
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
        zr = [np.arccos(zR[0, 0])]
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
        H = dse_lib.h_camera_3D(H, x, agent1, agent2, self.dim_state, self.dim_obs)
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
        zr = [np.arccos(zR[0, 0])]
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


if __name__ == '__main__':
    rospy.loginfo("-I- test_information_filter started")
    rospy.loginfo("-D- sys.argv: %s" % str(sys.argv))
    rostest.rosrun(PKG, 'test_information_filter_valid', TestInformationFilterValid, sys.argv)
