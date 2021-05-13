#!/usr/bin/env python2
from __future__ import print_function
import os
import roslib
import sys
import rospy
import numpy as np
import cv2
from cv2 import aruco
import pickle
import datetime
import time
from sensor_msgs.msg import Image
from geometry_msgs.msg import Pose
from dse_msgs.msg import PoseMarkers
from cv_bridge import CvBridge, CvBridgeError
from scipy.spatial.transform import Rotation as R

roslib.load_manifest('dse_simulation')


class aruco_pose:

    def __init__(self):

        # # Get parameters from launch file
        # self.ros_prefix = 'tb3_0'
        # if len(self.ros_prefix) != 0 and self.ros_prefix[0] != '/':
        #     self.ros_prefix = '/' + self.ros_prefix
        # # side length of tag in meters
        # self.markerLength = 0.1
        # self.cal_file = 'calibration_1080p.p'

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

        # drawAxis(image, cameraMatrix, distCoeffs, rvec, tvec, length) -> image

        if (ids is not None):
            # print('translation vectors')
            # print(tvecs)
            # print('rotation vectors')
            # print(rvecs)
            # print()
            for i in range(len(rvecs)):
                frame = aruco.drawAxis(frame, self.cameraMatrix, self.distCoeffs, rvecs[i], tvecs[i],
                                       self.markerLength / 2)

            for rot in rvecs:
                rvec_rodr = np.eye(3)
                rvec_rodr = cv2.Rodrigues(rot, rvec_rodr)
                # print(rvec_rodr[0])
                angles = np.zeros([0, 2])
                theta = -np.arcsin(rvec_rodr[0][2][0])
                psi = np.arctan2(rvec_rodr[0][2][1] / np.cos(theta), rvec_rodr[0][2][2] / np.cos(theta))
                phi = np.arctan2(rvec_rodr[0][1][0] / np.cos(theta), rvec_rodr[0][0][0] / np.cos(theta))
                psi *= 180 / np.pi
                theta *= 180 / np.pi
                phi *= 180 / np.pi
                ## [psi, theta, phi] rotate about
                ## [  y,     x,   z] respectively

            marker_pose = PoseMarkers()
            marker_pose.ids = list(ids.flatten())
            for i in range(len(rvecs)):
                pose = Pose()

                pose.position.x = tvecs[i][0][2]
                pose.position.y = -tvecs[i][0][0]
                pose.position.z = tvecs[i][0][1]

                rvecs_reordered = [rvecs[i][0][2], rvecs[i][0][0], rvecs[i][0][1]]
                r = R.from_rotvec(rvecs_reordered)
                quat = r.as_quat()

                pose.orientation.x = quat[0]
                pose.orientation.y = quat[1]
                pose.orientation.z = quat[2]
                pose.orientation.w = quat[3]
                marker_pose.pose_array.poses += [pose]
            marker_pose.pose_array.header.stamp = rospy.Time.now()
            marker_pose.pose_array.header.frame_id = self.ros_prefix + '/camera_rgb_frame'
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
