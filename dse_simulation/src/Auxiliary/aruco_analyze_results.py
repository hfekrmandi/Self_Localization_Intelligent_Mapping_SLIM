#!/usr/bin/env python2
from __future__ import print_function
import roslib
import sys
import rospy
import numpy as np
import datetime
import time
import os
import pickle
from geometry_msgs.msg import PoseArray
from geometry_msgs.msg import Pose
from geometry_msgs.msg import PoseWithCovariance
from std_msgs.msg import Bool
from nav_msgs.msg import Odometry
from dse_msgs.msg import PoseMarkers
from std_msgs.msg import Float64MultiArray
from std_msgs.msg import MultiArrayLayout
from std_msgs.msg import MultiArrayDimension
from dse_msgs.msg import InfFilterResults
from visualization_msgs.msg import Marker
from scipy.spatial.transform import Rotation as R
from gazebo_msgs.msg import LinkStates
import tf_conversions
import tf2_ros
import matplotlib.pyplot as plt

import gazebo_lib
import dse_lib
import dse_constants
roslib.load_manifest('dse_simulation')


def main(args):

    dump_file = "../aruco_data/aruco_data_plane_linear.p"
    cal = pickle.load(open(os.path.join(sys.path[0], dump_file), "rb"))
    [header, [true_poses, est_poses, error_poses, metrics]] = cal
    print('got data')

    time = np.arange(len(true_poses))
    true_poses = np.array(true_poses)
    est_poses = np.array(est_poses)
    error_poses = np.array(error_poses)
    metrics = np.array(metrics)

    num_objects = len(true_poses)
    colors = ['k', 'g', 'r', 'm', 'b', 'c', 'y']

    plt.figure()
    plt.grid()
    plt.plot(true_poses[:, 0], true_poses[:, 1], colors[0] + '.-', lw=1, label='true')
    plt.plot(est_poses[:, 0], est_poses[:, 1], colors[1] + '--', lw=1, label='estimated')

    plt.legend()
    plt.xlabel('x (m)')
    plt.ylabel('y (m)')
    plt.title('true vs. estimated position')
    plt.show()

    plt.figure()
    plt.tight_layout()
    plt.suptitle('error vs. metrics')

    plt.subplot(231)
    plt.grid()
    plt.plot(metrics[:, 0], error_poses[:, 0], colors[0] + '.', lw=1, label='x')
    plt.plot(metrics[:, 0], error_poses[:, 1], colors[1] + '.', lw=1, label='y')
    plt.plot(metrics[:, 0], error_poses[:, 2], colors[2] + '.', lw=1, label='z')
    plt.legend()
    plt.xlabel('true distance (m)')
    plt.ylabel('error (m)')
    plt.title('xyz error vs. true distance')

    plt.subplot(232)
    plt.grid()
    plt.plot(metrics[:, 1], error_poses[:, 0], colors[0] + '.', lw=1, label='x')
    plt.plot(metrics[:, 1], error_poses[:, 1], colors[1] + '.', lw=1, label='y')
    plt.plot(metrics[:, 1], error_poses[:, 2], colors[2] + '.', lw=1, label='z')
    plt.legend()
    plt.xlabel('measured distance (m)')
    plt.ylabel('error (m)')
    plt.title('xyz error vs. measured distance')

    plt.subplot(233)
    plt.grid()
    plt.plot(metrics[:, 2], error_poses[:, 0], colors[0] + '.', lw=1, label='x')
    plt.plot(metrics[:, 2], error_poses[:, 1], colors[1] + '.', lw=1, label='y')
    plt.plot(metrics[:, 2], error_poses[:, 2], colors[2] + '.', lw=1, label='z')
    plt.legend()
    plt.xlabel('measured pixel area (pixels)')
    plt.ylabel('error (m)')
    plt.title('xyz error vs. measured pixel area')

    plt.subplot(234)
    plt.grid()
    plt.plot(metrics[:, 0], error_poses[:, 3], colors[0] + '.', lw=1, label='yaw')
    plt.plot(metrics[:, 0], error_poses[:, 4], colors[1] + '.', lw=1, label='pitch')
    plt.plot(metrics[:, 0], error_poses[:, 5], colors[2] + '.', lw=1, label='roll')
    plt.legend()
    plt.xlabel('true distance (m)')
    plt.ylabel('error (m)')
    plt.title('ypr error vs. true distance')

    plt.subplot(235)
    plt.grid()
    plt.plot(metrics[:, 1], error_poses[:, 3], colors[0] + '.', lw=1, label='yaw')
    plt.plot(metrics[:, 1], error_poses[:, 4], colors[1] + '.', lw=1, label='pitch')
    plt.plot(metrics[:, 1], error_poses[:, 5], colors[2] + '.', lw=1, label='roll')
    plt.legend()
    plt.xlabel('measured distance (m)')
    plt.ylabel('error (m)')
    plt.title('xyz error vs. measured distance')

    plt.subplot(236)
    plt.grid()
    plt.plot(metrics[:, 2], error_poses[:, 3], colors[0] + '.', lw=1, label='yaw')
    plt.plot(metrics[:, 2], error_poses[:, 4], colors[1] + '.', lw=1, label='pitch')
    plt.plot(metrics[:, 2], error_poses[:, 5], colors[2] + '.', lw=1, label='roll')
    plt.legend()
    plt.xlabel('measured pixel area (pixels)')
    plt.ylabel('error (m)')
    plt.title('xyz error vs. measured pixel area')
    plt.show()


if __name__ == '__main__':
    main(sys.argv)
