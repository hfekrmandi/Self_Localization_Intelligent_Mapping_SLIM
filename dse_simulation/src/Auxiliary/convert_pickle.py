import pickle
import numpy as np
import os
import sys

# cal_file = 'calibrationSave_2.p'
# cal = pickle.load(open(os.path.join(sys.path[0], cal_file), "rb"))
# retval, cameraMatrix, distCoeffs, rvecsUnused, tvecsUnused = cal
# print()

retval = 0
cameraMatrix = np.array([[1080, 0, 960], [0, 1080, 540], [0, 0, 1]], dtype=float)
distCoeffs = np.array([[0, 0, 0, 0, 0]], dtype=float)
rvecsUnused = np.array([])
tvecsUnused = np.array([])

outputFile = 'calibration_1080p.p'
outputData = retval, cameraMatrix, distCoeffs, rvecsUnused, tvecsUnused
pickle.dump(outputData, open(os.path.join(sys.path[0], outputFile), 'wb'), protocol=2)

#data = pickle.load( open( "/home/alex/simulation_ws/src/Self_Localization_Intelligent_Mapping_SLIM/dse_simulation/src/calibrationSave_2.p", "rb" ) )
#allCorners = data[0]
#allIds = data[1]
#imsize = data[2]
#outputData = [allCorners, allIds, imsize]
#pickle.dump(outputData, open( "/home/alex/simulation_ws/src/dse_simulation/code/calibrationData_2.p", "wb" ), protocol=2)

