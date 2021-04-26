#!/usr/bin/env python3
from __future__ import print_function
import os
import roslib
import sys
import rospy
import numpy as np
import cv2
from cv2 import aruco
import tf2_ros
import pickle
import datetime
import time
from sensor_msgs.msg import Image
from tf2_geometry_msgs import PoseStamped
from geometry_msgs.msg import Pose
from geometry_msgs.msg import TransformStamped
from gazebo_msgs.msg import ModelState
from dse_msgs.msg import PoseMarkers
from cv_bridge import CvBridge, CvBridgeError
from scipy.spatial.transform import Rotation as R
import csv
import dse_lib

roslib.load_manifest('dse_simulation')


class aruco_pose:

    def __init__(self):

        # # Get parameters from launch file
        # self.ros_prefix = 'tb3_0'
        # if len(self.ros_prefix) != 0 and self.ros_prefix[0] != '/':
        #     self.ros_prefix = '/' + self.ros_prefix
        # # side length of tag in meters
        # self.markerLength = 0.1
        # self.cal_file = 'calibrationSave_gazebo.p'

        # Get parameters from launch file
        self.ros_prefix = rospy.get_param('~prefix', '')
        if len(self.ros_prefix) != 0 and self.ros_prefix[0] != '/':
            self.ros_prefix = '/' + self.ros_prefix
        # side length of tag in meters
        self.markerLength = rospy.get_param('~marker_length', 0.1)
        self.cal_file = rospy.get_param('~calibration_file', 'calibrationSave_2.p')
        self.data_skip = rospy.get_param('~data_skip', 0)
        self.data_skip_count = 0

        # x is forward, y is left, z is up
        # camera estimated max/mins
        self.max_dist = 8
        self.min_dist = 0.3
        self.max_theta = 0.9
        self.min_theta = -0.9
        self.max_height = 4
        self.min_height = 0.1
        self.max_yaw_pitch = 0.9

        self.target_image_count = 500

        # import saved calibration information
        # calibrationSave.p should be correct for laptop webcam
        cal = pickle.load(open(os.path.join(sys.path[0], self.cal_file), "rb"))
        self.retval, self.cameraMatrix, self.distCoeffs, self.rvecsUnused, self.tvecsUnused = cal
        print(self.distCoeffs)
        para = aruco.DetectorParameters_create()
        para.cornerRefinementMethod = aruco.CORNER_REFINE_SUBPIX

        self.decimator = 0
        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber(self.ros_prefix + "/camera/rgb/image_raw", Image, self.callback)
        self.pose_pub = rospy.Publisher(self.ros_prefix + "/dse/pose_markers", PoseMarkers, queue_size=1)
        self.tfBuffer = tf2_ros.Buffer()
        self.listener = tf2_ros.TransformListener(self.tfBuffer)
        self.br = tf2_ros.TransformBroadcaster()
        self.gazebo_modelstate_pub = rospy.Publisher("/gazebo/set_model_state", ModelState, queue_size=1)

        path = sys.path[0].split('/')
        output_file = os.path.join('/'.join(path[:path.index('dse_simulation')+1]), 'Debugging data/extrinsic_calibration data.csv')
        print(output_file)
        self.all_data_file = open(output_file, mode='w')
        self.all_csv_writer = csv.writer(self.all_data_file, delimiter=',')
        csv_header = []
        for i in ['true', 'est', 'error']:
            for j in ['x', 'y', 'z', 'rz', 'ry', 'rx']:
                csv_header.append(str(j + ' ' + i))
        self.all_csv_writer.writerow(csv_header)
        self.all_csv_writer.writerow(csv_header)

    def send_all_poses_csv(self, true_poses, est_poses):
        true_poses_np = np.array(true_poses)
        true_poses_flat = true_poses_np.flatten()
        est_poses_np = np.array(est_poses)
        est_poses_flat = est_poses_np.flatten()

        all = np.concatenate((true_poses_flat, est_poses_flat))
        self.all_csv_writer.writerow(all)

    def callback(self, data):
        # Lowering the camera image rate
        if self.data_skip_count < self.data_skip:
            self.data_skip_count += 1
            return
        self.data_skip_count = 0

        try:
            frame = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            print(e)
            return

        (rows, cols, channels) = frame.shape
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # This is where you set what type pf tag to use: aruco.DICT_NXN_250
        aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_250)
        parameters = aruco.DetectorParameters_create()

        # detectMarkers(image, dictionary[, corners[, ids[, parameters[, rejectedImgPoints]]]]) -> corners, ids, rejectedImgPoints
        corners, ids, rejectedImgPoints = aruco.detectMarkers(frame, aruco_dict, parameters=parameters)
        frame = aruco.drawDetectedMarkers(frame, corners, ids, (255, 255, 255))

        # estimatePoseSingleMarkers(corners, markerLength, cameraMatrix, distCoeffs[, rvecs[, tvecs[, _objPoints]]]) -> rvecs, tvecs, _objPoints
        rvecs, tvecs, objPoints = aruco.estimatePoseSingleMarkers(corners, self.markerLength, self.cameraMatrix,
                                                                  self.distCoeffs, None,
                                                                  None, None)

        if ids is not None and len(rvecs) == 1:
            from_tf = 'tb3_0/camera_rgb_frame'
            to_tf = 'aruco_marker_0'
            try:
                transform = self.tfBuffer.lookup_transform(from_tf, to_tf, rospy.Time(0))
            except tf2_ros.LookupException as e:
                print(e)
            else:
                x = tvecs[0][0][2]/1.184 + 0.110
                y = -tvecs[0][0][0]/1.032 + 0.243
                z = -tvecs[0][0][1]/1.151 - 0.297
                dist = np.sqrt(x**2 + y**2 + z**2)

                x = x - 0.008*dist + 0.031
                y = y + 0.049*dist - 0.222
                z = z - 0.062*dist + 0.281

                est_xyz = np.array([x, y, z])
                rvecs_reordered = [rvecs[0][0][2], rvecs[0][0][0], rvecs[0][0][1]]
                r = R.from_rotvec(rvecs_reordered)
                est_ypr = r.as_euler('zyx')
                r = R.from_euler('zyx', est_ypr + [np.pi, 0, np.pi])
                est_ypr = r.as_euler('zyx')
                est_xyzypr = np.concatenate((est_xyz, est_ypr))

                trans = transform.transform.translation
                rot = transform.transform.rotation
                true_xyz = [trans.x, trans.y, trans.z]
                true_ypr = dse_lib.quat_from_pose2eul(transform.transform.rotation)
                true_xyzypr = np.concatenate((true_xyz, true_ypr))

                r = R.from_euler('zyx', est_ypr)
                meas_rotm = r.as_matrix()
                r = R.from_euler('zyx', true_ypr)
                true_rotm = r.as_matrix()
                error_rotm = meas_rotm.dot(true_rotm.T)
                r = R.from_matrix(error_rotm)
                error_ypr = r.as_euler('zyx')
                error_xyz = est_xyz - true_xyz
                error_xyzypr = np.concatenate((error_xyz, error_ypr))
                self.all_csv_writer.writerow(np.concatenate((true_xyzypr, est_xyzypr, error_xyzypr)))

                self.decimator += 1
                print(self.target_image_count - self.decimator, " images left")

        # Move the tag to a new position/orientation
        new_state = ModelState()
        new_state.model_name = 'aruco_marker_0'
        dist = np.random.uniform(self.min_dist, self.max_dist)
        theta = np.random.uniform(self.min_theta, self.max_theta)
        height = np.random.uniform(self.min_height, self.max_height)
        roll = np.random.uniform(-np.pi, np.pi)
        yaw = np.random.uniform(-self.max_yaw_pitch, self.max_yaw_pitch)
        pitch = np.random.uniform(-self.max_yaw_pitch, self.max_yaw_pitch)
        new_state.pose.position.x = dist * np.cos(theta)
        new_state.pose.position.y = dist * np.sin(theta)
        new_state.pose.position.z = height
        eul = np.array([yaw, pitch, roll])
        new_state.pose.orientation = dse_lib.euler2quat_from_pose(new_state.pose.orientation, eul[:, None])
        self.gazebo_modelstate_pub.publish(new_state)

        t = TransformStamped()
        t.header.frame_id = "world"
        t.child_frame_id = "aruco_marker_0"
        t.header.stamp = rospy.Time.now()
        t.transform.translation.x = new_state.pose.position.x
        t.transform.translation.y = new_state.pose.position.y
        t.transform.translation.z = new_state.pose.position.z
        t.transform.rotation = new_state.pose.orientation
        self.br.sendTransform(t)

        # Once we have enough images, generate calibration file and end
        if self.decimator >= self.target_image_count:
            self.image_sub.unregister()
            return



def main(args):
    rospy.init_node('aruco_pose_estimation_node', anonymous=True)
    ic = aruco_pose()
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main(sys.argv)
