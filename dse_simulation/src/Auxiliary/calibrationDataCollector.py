'''
This software only collects and stores the information needed for the calibration.
Allows for RPi camera to be calibrated on a more capable computer.
'''

import time
import cv2.aruco as A
import cv2
import numpy as np
import pickle

dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
# board = cv2.aruco.CharucoBoard_create(5,8,.025,.0125,dictionary)
board = cv2.aruco.CharucoBoard_create(5, 8, 0.1, 0.075, dictionary)
img = board.draw((2000, 2000))

#Dump the calibration board to a file
cv2.imwrite('charuco.png', img)


#Start capturing images for calibration
cap = cv2.VideoCapture(0)

allCorners = []
allIds = []
decimator = 0
for i in range(400):

    ret,frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    res = cv2.aruco.detectMarkers(gray, dictionary)

    if len(res[0])>0:
        res2 = cv2.aruco.interpolateCornersCharuco(res[0],res[1], gray, board)
        if res2[1] is not None and res2[2] is not None and len(res2[1])>3 and decimator%4==0:
            allCorners.append(res2[1])
            allIds.append(res2[2])

        cv2.aruco.drawDetectedMarkers(gray, res[0], res[1])

    cv2.imshow('frame', gray)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    decimator += 1

imsize = gray.shape

calibrationData = [allCorners, allIds, imsize]
pickle.dump(calibrationData, open( "calibrationData.p", "wb" ))

cap.release()
cv2.destroyAllWindows()
