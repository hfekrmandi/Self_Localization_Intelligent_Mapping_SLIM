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
from gazebo_msgs.msg import ModelState
import dse_lib

roslib.load_manifest('dse_simulation')


class charuco_calibration:
    def __init__(self):

        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
        # # board = cv2.aruco.CharucoBoard_create(5,8,.025,.0125,dictionary)
        self.board = cv2.aruco.CharucoBoard_create(5, 8, 0.1, 0.075, self.aruco_dict)
        img = self.board.draw((2000, 2000))
        #
        # Dump the calibration board to a file
        cv2.imwrite('charuco.png', img)

        # x is forward, y is left, z is up
        # camera estimated max/mins
        self.max_dist = 5
        self.min_dist = 0.6
        self.max_theta = 0.707
        self.min_theta = -0.707
        self.max_height = 2
        self.min_height = 0.4
        self.max_yaw_pitch = 0.707

        self.target_image_count = 100
        self.data_skip_count = 0
        self.data_skip = 100
        self.allCorners = []
        self.allIds = []
        self.decimator = 0
        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber("camera/rgb/image_raw", Image, self.image_cb)
        self.gazebo_modelstate_pub = rospy.Publisher("/gazebo/set_model_state", ModelState, queue_size=1)

    def image_cb(self, data):
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

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        self.imsize = gray.shape
        res = cv2.aruco.detectMarkers(gray, self.aruco_dict)

        if len(res[0]) > 0:
            res2 = cv2.aruco.interpolateCornersCharuco(res[0], res[1], gray, self.board)
            if res2[1] is not None and res2[2] is not None and len(res2[1]) > 7:
                self.allCorners.append(res2[1])
                self.allIds.append(res2[2])
                self.decimator += 1
                print('remaining: ', self.target_image_count - self.decimator)

            # cv2.aruco.drawDetectedMarkers(gray, res[0], res[1])

        # cv2.imshow('frame', gray)
        # cv2.waitKey(3)

        # Move the tag to a new position/orientation
        new_state = ModelState()
        new_state.model_name = 'charuco_board'
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

        # Once we have enough images, generate calibration file and end
        if self.decimator >= self.target_image_count:
            self.image_sub.unregister()
            self.calibrate()

    def calibrate(self):

        cv2.destroyAllWindows()

        print(self.allIds)

        print("calibrating now")
        startTime = time.time()
        # print(startTime)

        try:
            print("something else")
            cal = cv2.aruco.calibrateCameraCharuco(self.allCorners, self.allIds, self.board, self.imsize, None, None)
            print("something")
        except:
            print("failure")
            raise
        else:
            print(cal)
            print("triumph")  # huge success, hard to overstate my satisfaction
            deltaTime = time.time() - startTime
            print("calibration took " + str(deltaTime) + " seconds")
            outputFile = "calibrationSave_gazebo.p"
            pickle.dump(cal, open(os.path.join(sys.path[0], outputFile), "wb"), protocol=2)
            # retval, cameraMatrix, distCoeffs, rvecs, tvecs = cal


def main(args):
    rospy.init_node('charuco_calibration_node', anonymous=True)
    ic = charuco_calibration()
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main(sys.argv)
