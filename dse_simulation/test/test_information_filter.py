#!/usr/bin/env python
# Software License Agreement (BSD License)
#
# Copyright (c) 2008, Willow Garage, Inc.
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above
#    copyright notice, this list of conditions and the following
#    disclaimer in the documentation and/or other materials provided
#    with the distribution.
#  * Neither the name of Willow Garage, Inc. nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
# FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
# COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
# BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
# LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
# ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
#
# Revision $Id$

## Integration test for add_two_ints

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
from dse_lib import *

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

    # When the information filter sends partials (prior and measurement), combine and return them
    def information_callback(self, data):
        rospy.loginfo("-D- information_filter sent back data")
        inf_id_list = data.ids
        self.inf_Y_prior = multi_array_2d_output(data.inf_matrix_prior)
        self.inf_y_prior = multi_array_2d_output(data.inf_vector_prior)
        self.inf_I = multi_array_2d_output(data.obs_matrix)
        self.inf_i = multi_array_2d_output(data.obs_vector)


##############################################################################
##############################################################################
class TestInformationFilterValid(TestInformationFilterCommon):
    ##############################################################################
    ##############################################################################

    ##############################################################################
    def test_one_equals_one(self):
        ##############################################################################
        rospy.loginfo("-D- test_one_equals_one")
        self.assertEquals(1, 1, "1!=1")

    def test_theta_2_rotm_0(self):
        ##############################################################################
        rospy.loginfo("-D- test_theta_2_rotm_0")
        rotm = theta_2_rotm(0)
        x_0 = np.transpose([1, 2])
        x_rotm = rotm.dot(x_0)
        x_true = x_0
        self.assertEquals(True, np.allclose(x_true, x_rotm))

    def test_theta_2_rotm_1(self):
        ##############################################################################
        rospy.loginfo("-D- test_theta_2_rotm_0")
        theta = 90
        rotm = theta_2_rotm(theta * np.pi / 180.0)
        x_0 = np.transpose([1, 2])
        x_rotm = rotm.dot(x_0)
        x_true = np.transpose([-2, 1])
        self.assertEquals(True, np.allclose(x_true, x_rotm))

    def test_theta_2_rotm_2(self):
        ##############################################################################
        rospy.loginfo("-D- test_theta_2_rotm_0")
        theta = 45
        rotm = theta_2_rotm(theta * np.pi / 180.0)
        x_0 = np.transpose([1, 1])
        x_rotm = rotm.dot(x_0)
        x_true = np.transpose([0, np.sqrt(2)])
        self.assertEquals(True, np.allclose(x_true, x_rotm))

    def test_observation_jacobian_0(self):
        ##############################################################################
        rospy.loginfo("-D- test_observation_jacobian_0")
        agent1 = 0
        agent2 = 1
        x = np.zeros((12, 1))
        H = np.zeros((3, 12))
        H = h_camera_3D(H, x, agent1, agent2, self.dim_state, self.dim_obs)
        z_jac = H.dot(x)

        agent1_row_min = self.dim_state * agent1
        agent1_row_max = agent1_row_min + self.dim_obs
        agent2_row_min = self.dim_state * agent2
        agent2_row_max = agent2_row_min + self.dim_obs
        x1 = x[agent1_row_min:agent1_row_max]
        t1 = x1[0:2]
        R1 = theta_2_rotm(x1[2, 0])

        x2 = x[agent2_row_min:agent2_row_max]
        t2 = x2[0:2]
        R2 = theta_2_rotm(x2[2, 0])

        zt = (np.transpose(R1).dot(t2) - np.transpose(R1).dot(t1))[:, 0]
        zR = np.transpose(R1).dot(R2)
        zr = [np.arccos(zR[0, 0])]
        z_true = np.concatenate((zt, zr))[:, None]

        rospy.loginfo("-D- z_jac (%d, %d)" % (np.shape(z_jac)[0], np.shape(z_jac)[1]))
        rospy.loginfo("-D- z_jac (%d, %d)" % (np.shape(z_true)[0], np.shape(z_true)[1]))
        self.assertEquals(True, np.allclose(z_true, z_jac))

    def test_observation_jacobian_1(self):
        ##############################################################################
        rospy.loginfo("-D- test_observation_jacobian_0")
        agent1 = 0
        agent2 = 1
        x = np.transpose([1, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])[:, None]
        H = np.zeros((3, 12))
        H = h_camera_3D(H, x, agent1, agent2, self.dim_state, self.dim_obs)
        z_jac = H.dot(x)

        agent1_row_min = self.dim_state * agent1
        agent1_row_max = agent1_row_min + self.dim_obs
        agent2_row_min = self.dim_state * agent2
        agent2_row_max = agent2_row_min + self.dim_obs
        x1 = x[agent1_row_min:agent1_row_max]
        t1 = x1[0:2]
        R1 = theta_2_rotm(x1[2, 0])

        x2 = x[agent2_row_min:agent2_row_max]
        t2 = x2[0:2]
        R2 = theta_2_rotm(x2[2, 0])

        zt = (np.transpose(R1).dot(t2) - np.transpose(R1).dot(t1))[:, 0]
        zR = np.transpose(R1).dot(R2)
        zr = [np.arccos(zR[0, 0])]
        z_true = np.concatenate((zt, zr))[:, None]

        rospy.loginfo("-D- z_jac (%d, %d)" % (np.shape(z_jac)[0], np.shape(z_jac)[1]))
        rospy.loginfo("-D- z_jac (%d, %d)" % (np.shape(z_true)[0], np.shape(z_true)[1]))
        self.assertEquals(True, np.allclose(z_true, z_jac))

    def test_observation_jacobian_2(self):
        ##############################################################################
        rospy.loginfo("-D- test_observation_jacobian_0")
        agent1 = 0
        agent2 = 1
        x = np.transpose([1, 2, np.pi/2, 0, 0, 0, 0, 0, -np.pi/2, 0, 0, 0])[:, None]
        H = np.zeros((3, 12))
        H = h_camera_3D(H, x, agent1, agent2, self.dim_state, self.dim_obs)
        z_jac = H.dot(x)

        agent1_row_min = self.dim_state * agent1
        agent1_row_max = agent1_row_min + self.dim_obs
        agent2_row_min = self.dim_state * agent2
        agent2_row_max = agent2_row_min + self.dim_obs
        x1 = x[agent1_row_min:agent1_row_max]
        t1 = x1[0:2]
        R1 = theta_2_rotm(x1[2, 0])

        x2 = x[agent2_row_min:agent2_row_max]
        t2 = x2[0:2]
        R2 = theta_2_rotm(x2[2, 0])

        zt = (np.transpose(R1).dot(t2) - np.transpose(R1).dot(t1))[:, 0]
        zR = np.transpose(R1).dot(R2)
        zr = [np.arccos(zR[0, 0])]
        z_true = np.concatenate((zt, zr))[:, None]

        self.assertEquals(True, np.allclose(z_true, z_jac))

    # ##############################################################################
    # def test_filtered_ones(self):
    #     ##############################################################################
    #
    #     self.latest_filtered = 1e10
    #     self.latest_std = 2e10
    #     msgs = [1] * (self.rolling_pts)
    #     rospy.loginfo("-D- test_filtered_ones")
    #     self.sendmsgs(msgs, self.test_rate)
    #     rospy.sleep(1)
    #     self.assertEquals(self.latest_filtered, self.coefficient,
    #                       "filtered_ones != expected (%0.3f != %0.3f)" % (self.latest_filtered, self.coefficient))
    #     self.assertEquals(self.latest_std, 0, "filtered_ones: std != 0 (%0.3f != 0)" % self.latest_std)
    #
    # ##############################################################################
    # def const_value_test(self, val, testname):
    #     ##############################################################################
    #
    #     self.latest_filtered = 1e10
    #     self.latest_std = 2e10
    #     msgs = [val] * (self.rolling_pts)
    #     rospy.loginfo("-D- test_filtered_ones")
    #     self.sendmsgs(msgs, self.test_rate)
    #     rospy.sleep(1)
    #     expected = self.coefficient * val ** self.exponent
    #     self.assertAlmostEqual(self.latest_filtered, expected, 4, "%s: filtered_ones != expected (%0.3f != %0.3f)" % (
    #     testname, self.latest_filtered, expected))
    #     self.assertEquals(self.latest_std, 0, "%s; filtered_ones: std != 0 (%0.3f != 0)" % (testname, self.latest_std))
    #
    # ##############################################################################
    # def test_filtered_ones(self):
    #     ##############################################################################
    #     self.const_value_test(1, "constant_ones")
    #     self.const_value_test(2, "constant_twos")
    #     self.const_value_test(10, "constant_tens")
    #     self.const_value_test(100, "constant_hundreds")
    #     self.const_value_test(572, "constant_572")
    #
    # ##############################################################################
    # def test_invalid_zeros(self):
    #     ##############################################################################
    #
    #     msgs = [0] * (self.rolling_pts)
    #     self.latest_filtered = 1e10
    #     self.latest_std = 2e10
    #     self.sendmsgs(msgs, self.test_rate)
    #     rospy.sleep(1)
    #     self.assertEquals(self.latest_filtered, 1e10, "filtered_zeros: value != 1e10 (%0.3f != 0)" % self.latest_std)
    #     self.assertEquals(self.latest_std, 2e10, "filtered_zeros: std != 2e10 (%0.3f != 0)" % self.latest_std)
    #
    # ##############################################################################
    # def test_invalid_neg(self):
    #     ##############################################################################
    #
    #     msgs = [-1] * (self.rolling_pts)
    #     # rospy.loginfo("-D- test_invalid_neg")
    #     self.sendmsgs(msgs, self.test_rate)
    #     self.latest_filtered = 1e10
    #     self.latest_std = 2e10
    #     rospy.sleep(1)
    #     self.assertEquals(self.latest_filtered, 1e10, "filtered_neg: std != 0 (%0.3f != 1e10)" % self.latest_std)
    #     self.assertEquals(self.latest_std, 2e10, "filtered_neg: std != 0 (%0.3f != 2e10)" % self.latest_std)


if __name__ == '__main__':
    rospy.loginfo("-I- test_information_filter started")
    #rospy.set_param("test_rate", 100)
    rospy.loginfo("-D- sys.argv: %s" % str(sys.argv))
    rostest.rosrun(PKG, 'test_information_filter_valid', TestInformationFilterValid, sys.argv)
