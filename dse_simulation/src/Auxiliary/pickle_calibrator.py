'''
Performs calibration using data pickled by calibrationDataCollector.py
Allows for RPi camera to be calibrated on a more capable computer.
'''

import time
import cv2.aruco as A
import cv2
import numpy as np
import pickle

calibrationData = pickle.load( open( "/home/alex/simulation_ws/src/dse_simulation/code/calibrationData_2.p", "rb" ) )
allCorners = calibrationData[0]
allIds = calibrationData[1]
imsize = calibrationData[2]

dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
board = cv2.aruco.CharucoBoard_create(5,8,.025,.0125,dictionary)

print(allIds)

print("calibrating now")
startTime = time.time()
#print(startTime)


try:
    print("something else")
    cal = cv2.aruco.calibrateCameraCharuco(allCorners,allIds,board,imsize,None,None)
    print("something")
except:
    print("failure")
    raise
else:
    print("triumph") # huge success, hard to overstate my satisfaction
    deltaTime = time.time() - startTime
    print("calibration took " + str(deltaTime) + " seconds")
    pickle.dump(cal, open( "/home/alex/simulation_ws/src/dse_simulation/code/calibrationSave_2.p", "wb" ))
    #retval, cameraMatrix, distCoeffs, rvecs, tvecs = cal
'''
cal = cv2.aruco.calibrateCameraCharuco(allCorners,allIds,board,imsize,None,None)
print("triumph") # huge success, hard to overstate my satisfaction
pickle.dump(cal, open( "calibrationSave.p", "wb" ))
'''
