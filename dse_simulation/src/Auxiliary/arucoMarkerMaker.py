#!/usr/bin/env python3
import numpy as np
import cv2
import cv2.aruco as aruco
from dse_simulation.src import dse_constants

'''
    drawMarker(...)
        drawMarker(dictionary, id, sidePixels[, img[, borderBits]]) -> img
'''

# # get input from user, verify value is valid
# validValue = False
# while(validValue != True):
#     try:
#         value=int(input('Please input integer from 0 to 249 for tag value: '))
#     except ValueError:
#         print("Not a number")
#     else:
#         if(value >= 0):
#             validValue = True
#         else:
#             print("Number must be greater or equal to zero")

'''
More tag versions are available and listed below.

DICT_4X4_100
DICT_4X4_1000
DICT_4X4_250
DICT_4X4_50
DICT_5X5_100
DICT_5X5_1000
DICT_5X5_250
DICT_5X5_50
DICT_6X6_100
DICT_6X6_1000
DICT_6X6_250
DICT_6X6_50
DICT_7X7_100
DICT_7X7_1000
DICT_7X7_250
DICT_7X7_50
'''


total_size = dse_constants.ARUCO_TAG_TOTAL_SIZE
aruco_size = dse_constants.ARUCO_TAG_NO_BORDER_SIZE
start = int((total_size - aruco_size)/2)
end = int(total_size - (total_size - aruco_size)/2)

# tag restricted to 6x6 blocks plus 1 block thick black perimeter
aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_250)
image_path = "/home/alex/simulation_ws/src/Self_Localization_Intelligent_Mapping_SLIM/dse_turtlebot_descriptions/Media/materials/textures/fiducials/"
desc = "/home/alex/simulation_ws/src/Self_Localization_Intelligent_Mapping_SLIM/dse_turtlebot_descriptions/Media/materials/scripts/fiducials2.material"

with open(desc, 'w+') as f:
    for value in range(50, 55):
        # second parameter is id number, last parameter is total image size
        img = aruco.drawMarker(aruco_dict, value, aruco_size)
        #cv2.imshow('Generated Aruco Marker (Press any key to quit)',img)
        new_img = 255 * np.ones((total_size, total_size))
        new_img[start:end, start:end] = img
        #cv2.imshow('Generated Aruco Marker (Press any key to quit)', new_img)
        image_name = "aruco_marker_" + str(value) + ".jpg"
        cv2.imwrite(image_path + image_name, new_img)

        text = """material Aruco/Tag""" + str(value) + """
{
  technique
  {
    pass
    {
      ambient 1 1 1 1
      diffuse 1 1 1 1
      specular 0 0 0 0

      texture_unit
      {
        texture ../textures/fiducials/""" + image_name + """
        filtering bilinear
        max_anisotropy 16
      }
    }
  }
}\n\n"""

        f.write(text)

#cv2.waitKey(0)
#cv2.destroyAllWindows()
