'''
Performs calibration using data pickled by calibrationDataCollector.py
Allows for RPi camera to be calibrated on a more capable computer.
'''

import time
import cv2.aruco as A
import cv2
import numpy as np
import pickle

dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
board = cv2.aruco.CharucoBoard_create(5,8,.025,.0125,dictionary)
img = board.draw((200*3,200*3))

#Dump the calibration board to a file
cv2.imwrite('charuco.png',img)


#Start capturing images for calibration
cap = cv2.VideoCapture(0)

allCorners = []
allIds = []
decimator = 0
in_between_images = 5
for i in range(400 * in_between_images):

    ret, frame = cap.read()
    if i % in_between_images == 0:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        res = cv2.aruco.detectMarkers(gray, dictionary)

        if len(res[0]) > 0:
            res2 = cv2.aruco.interpolateCornersCharuco(res[0], res[1], gray, board)
            if res2[1] is not None and res2[2] is not None and len(res2[1]) > 3 and decimator % 4 == 0:
                allCorners.append(res2[1])
                allIds.append(res2[2])

            cv2.aruco.drawDetectedMarkers(gray, res[0], res[1])

    cv2.imshow('frame', gray)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    decimator += 1

imsize = gray.shape

cap.release()
cv2.destroyAllWindows()

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
    pickle.dump(cal, open( "calibrationSave_2.p", "wb" ), protocol=2)
    #retval, cameraMatrix, distCoeffs, rvecs, tvecs = cal
'''

cal = cv2.aruco.calibrateCameraCharuco(allCorners,allIds,board,imsize,None,None)
print("triumph") # huge success, hard to overstate my satisfaction
pickle.dump(cal, open( "calibrationSave.p", "wb" ))
'''

