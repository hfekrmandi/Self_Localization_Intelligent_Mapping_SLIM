#!/usr/bin/env python2
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
from dse_msgs.msg import PoseMarkers
from cv_bridge import CvBridge, CvBridgeError
from scipy.spatial.transform import Rotation as R
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
        self.markerLength = rospy.get_param('~marker_length', 0.22884)
        self.cal_file = rospy.get_param('~calibration_file', 'calibrationSave_2.p')
        self.data_skip = rospy.get_param('~data_skip', 0)
        self.data_skip_count = 0

        # import saved calibration information
        # calibrationSave.p should be correct for laptop webcam
        cal = pickle.load(open(os.path.join(sys.path[0], self.cal_file), "rb"))
        self.retval, self.cameraMatrix, self.distCoeffs, self.rvecsUnused, self.tvecsUnused = cal
        print(self.distCoeffs)
        para = aruco.DetectorParameters_create()
        para.cornerRefinementMethod = aruco.CORNER_REFINE_SUBPIX

        # # import saved calibration information
        # # calibrationSave.p should be correct for laptop webcam
        # cal2 = pickle.load(open(os.path.join(sys.path[0], 'calibration_1080p.p'), "rb"))
        # self.retval2, self.cameraMatrix2, self.distCoeffs2, self.rvecsUnused2, self.tvecsUnused2 = cal2
        # print(self.distCoeffs2)

        # cap = cv2.VideoCapture(0)
        #
        # fourcc = cv2.VideoWriter_fourcc(*'XVID')
        # out = cv2.VideoWriter('output.avi', fourcc, 12.0, (640, 480), False)
        # font = cv2.FONT_HERSHEY_SIMPLEX

        prev_time = datetime.datetime.now()
        times = [0]
        n_stored = 0
        n_stored_max = 10
        n_points_plotting = 100

        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber(self.ros_prefix + "/camera/rgb/image_raw", Image, self.callback)
        self.pose_pub = rospy.Publisher(self.ros_prefix + "/dse/pose_markers", PoseMarkers, queue_size=1)
        self.tfBuffer = tf2_ros.Buffer()
        self.listener = tf2_ros.TransformListener(self.tfBuffer)

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

        if (ids is not None):
            # print('translation vectors')
            # print(tvecs)
            # print('rotation vectors')
            # print(rvecs)
            # print()
            for i in range(len(rvecs)):
                frame = aruco.drawAxis(frame, self.cameraMatrix, self.distCoeffs, rvecs[i], tvecs[i],
                                       self.markerLength / 2)

            marker_pose = PoseMarkers()
            marker_pose.ids = list(ids.flatten())
            pose_array = []
            for i in range(len(rvecs)):
                pose = PoseStamped()
                pose.header.stamp = rospy.Time.now()
                pose.header.frame_id = self.ros_prefix[1:] + '/camera_rgb_frame'

                # Apply linear bias to the translation estimates
                # x = tvecs[0][0][2]/1.184 + 0.110
                # y = -tvecs[0][0][0]/1.032 + 0.243
                # z = -tvecs[0][0][1]/1.151 - 0.297
                # dist = np.sqrt(x**2 + y**2 + z**2)
                # x = x - 0.008*dist + 0.031
                # y = y + 0.049*dist - 0.222
                # z = z - 0.062*dist + 0.281
                pose.pose.position.x = tvecs[0][0][2]
                pose.pose.position.y = -tvecs[0][0][0]
                pose.pose.position.z = -tvecs[0][0][1]

                # Swap the angles around to correctly represent our coordinate system
                # Aruco puts zero at the tag, with z facing out...
                # We want x forward, y left, z up, euler order zyx = ypr
                rvecs_reordered = [rvecs[0][0][2], rvecs[0][0][0], rvecs[0][0][1]]
                r = R.from_rotvec(rvecs_reordered)
                est_ypr = r.as_euler('zyx')
                quat = dse_lib.eul2quat(est_ypr[:, None])
                # r = R.from_euler('zyx', est_ypr + [np.pi, 0, np.pi])
                # quat = r.as_quat
                # print(quat[:])
                # print(pose.pose.orientation)
                pose.pose.orientation.x = quat[0]
                pose.pose.orientation.y = quat[1]
                pose.pose.orientation.z = quat[2]
                pose.pose.orientation.w = quat[3]

                pose = self.tfBuffer.transform(pose, self.ros_prefix[1:] + '/base_link')
                pose_array.append(pose)
            marker_pose.pose_array = pose_array
            self.pose_pub.publish(marker_pose)


        # cv2.imshow("Image window", frame)
        # cv2.waitKey(3)


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
