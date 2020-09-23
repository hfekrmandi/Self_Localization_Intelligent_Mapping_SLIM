#!/usr/bin/env python
from __future__ import print_function
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
from cv_bridge import CvBridge, CvBridgeError

roslib.load_manifest('my_package')


class aruco_pose:

    def __init__(self):
        # side length of tag in meters
        self.markerLength = 0.1
        # import saved calibration information
        # calibrationSave.p should be correct for laptop webcam
        cal = pickle.load(open("/home/alex/simulation_ws/src/dse_simulation/code/calibrationSave_2.p", "rb"))
        self.retval, self.cameraMatrix, self.distCoeffs, self.rvecsUnused, self.tvecsUnused = cal
        para = aruco.DetectorParameters_create()
        para.cornerRefinementMethod = aruco.CORNER_REFINE_SUBPIX

        cap = cv2.VideoCapture(0)

        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter('output.avi', fourcc, 12.0, (640, 480), False)
        font = cv2.FONT_HERSHEY_SIMPLEX

        prev_time = datetime.datetime.now()
        times = [0]
        n_stored = 0
        n_stored_max = 10
        n_points_plotting = 100

        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber("/camera/rgb/image_raw", Image, self.callback)

    def callback(self, data):
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
        rvecs, tvecs, objPoints = aruco.estimatePoseSingleMarkers(corners, self.markerLength, self.cameraMatrix, self.distCoeffs, None,
                                                                  None, None)

        # drawAxis(image, cameraMatrix, distCoeffs, rvec, tvec, length) -> image

        if (ids is not None):
            print('translation vectors')
            print(tvecs)
            print('rotation vectors')
            # print(rvecs)
            # print()
            for i in range(len(rvecs)):
                frame = aruco.drawAxis(frame, self.cameraMatrix, self.distCoeffs, rvecs[i], tvecs[i], self.markerLength / 2)

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
                print([psi, theta, phi])
                ## [psi, theta, phi] rotate about
                ## [  y,     x,   z] respectively

            print()

        (rows, cols, channels) = frame.shape
        if cols > 60 and rows > 60:
            cv2.circle(frame, (50, 50), 10, 255)

        cv2.imshow("Image window", frame)
        cv2.waitKey(3)


def main(args):
    ic = aruco_pose()
    rospy.init_node('aruco_pose_estimate', anonymous=True)
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main(sys.argv)
