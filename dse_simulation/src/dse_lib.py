import roslib
import sys
import rospy
import numpy as np
import datetime
import time
from dse_msgs.msg import PoseMarkers
from std_msgs.msg import Float64MultiArray
from std_msgs.msg import MultiArrayLayout
from std_msgs.msg import MultiArrayDimension
from dse_msgs.msg import InfFilterPartials
from dse_msgs.msg import InfFilterResults
from scipy.spatial.transform import Rotation as R


def svgs_R_from_range_SRT(range):
    # Assuming linear error with a slope of:
    # [x y z phi theta psi]
    # x = [0.0515; 0.0515; 0.018; 0.1324; 0.1324; 0.1324]; # Degrees
    x = np.transpose([0.0515, 0.0515, 0.018, 0.0023, 0.0023, 0.0023]) # Radians
    # x = [0.0075; 0.0075; 0.0075; 0.0075; 0.0075; 0.0075]; # 5% of distance

    # Slope values are for 3-sigma error, so dividing by 3
    range = (range / 3) * np.eye(6)
    r_std = np.multiply(range, x)
    r_var = np.multiply(r_std, r_std)
    # Compute variance from standard deviation
    return r_var


def aruco_R_from_range(range):
    # Assuming linear error with a slope of:
    # [x y z phi theta psi]
    # x = [0.0515; 0.0515; 0.018; 0.1324; 0.1324; 0.1324]; # Degrees
    x = 10*np.transpose([0.01, 0.01, 0.01, 0.01, 0.01, 0.01]) # Radians
    # x = [0.0075; 0.0075; 0.0075; 0.0075; 0.0075; 0.0075]; # 5% of distance

    # Slope values are for 3-sigma error, so dividing by 3
    range = range * np.eye(6)
    r_std = np.multiply(range, x)
    r_var = np.multiply(r_std, r_std)
    # Compute variance from standard deviation
    return r_var


def aruco_R_from_range_3D(range):
    # Assuming linear error with a slope of:
    # [x y z phi theta psi]
    # x = [0.0515; 0.0515; 0.018; 0.1324; 0.1324; 0.1324]; # Degrees
    x = 10*np.transpose([0.01, 0.01, 0.01]) # Radians
    # x = [0.0075; 0.0075; 0.0075; 0.0075; 0.0075; 0.0075]; # 5% of distance

    # Slope values are for 3-sigma error, so dividing by 3
    range = range * np.eye(3)
    r_std = np.multiply(range, x)
    r_var = np.multiply(r_std, r_std)
    # Compute variance from standard deviation
    return r_var


def theta_2_rotm(theta):
    R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    return R


def multi_array_2d_input(mat, multi_arr):
    multi_arr.layout.dim.append(MultiArrayDimension())
    multi_arr.layout.dim.append(MultiArrayDimension())
    multi_arr.layout.dim[0].label = 'rows'
    multi_arr.layout.dim[0].size = np.shape(mat)[0]
    multi_arr.layout.dim[0].stride = np.shape(mat)[0]*np.shape(mat)[1]
    multi_arr.layout.dim[1].label = 'cols'
    multi_arr.layout.dim[1].size = np.shape(mat)[1]
    multi_arr.layout.dim[1].stride = np.shape(mat)[1]
    multi_arr.layout.data_offset = 0

    multi_arr.data = mat.flatten()
    return multi_arr


def multi_array_2d_output(multi_arr):
    arr = np.array(multi_arr.data)
    shape = [multi_arr.layout.dim[0].size, multi_arr.layout.dim[1].size]
    mat = arr.reshape(shape)
    return mat

def dual_relative_obs_jacobian(vector_1, vector_2):

    [x1, y1, z1, p1, t1, s1] = vector_1
    [x2, y2, z2, p2, t2, s2] = vector_2

    Jx = [-np.cos(s1) * np.cos(t1), -np.cos(t1) * np.sin(s1), np.sin(t1), 0, 0, 0,
          np.cos(s1) * np.cos(t1),np.cos(t1) * np.sin(s1), -np.sin(t1), 0, 0, 0]

    Jy = [np.cos(p1) * np.sin(s1) - np.cos(s1) * np.sin(p1) * np.sin(t1),
          - np.cos(p1) * np.cos(s1) - np.sin(p1) * np.sin(s1) * np.sin(t1), -np.cos(t1) * np.sin(p1), 0, 0, 0,
          np.cos(s1) * np.sin(p1) * np.sin(t1) - np.cos(p1) * np.sin(s1),
          np.cos(p1) * np.cos(s1) + np.sin(p1) * np.sin(s1) * np.sin(t1), np.cos(t1) * np.sin(p1), 0, 0, 0]

    Jz = [- np.sin(p1) * np.sin(s1) - np.cos(p1) * np.cos(s1) * np.sin(t1),
          np.cos(s1) * np.sin(p1) - np.cos(p1) * np.sin(s1) * np.sin(t1), -np.cos(p1) * np.cos(t1), 0, 0, 0,
          np.sin(p1) * np.sin(s1) + np.cos(p1) * np.cos(s1) * np.sin(t1),
          np.cos(p1) * np.sin(s1) * np.sin(t1) - np.cos(s1) * np.sin(p1), np.cos(p1) * np.cos(t1), 0, 0, 0]

    Jp = [0, 0, 0, 0, 0, 0,
          0, 0, 0, 0, 0, 0]

    Jt = [0, 0, 0, 0, 0, 0,
          0, 0, 0, 0, 0, 0]

    Js = [0, 0, 0, 0, 0, 0,
          0, 0, 0, 0, 0, 0]

    J = [Jx, Jy, Jz, Jp, Jt, Js]
    return J


def dual_relative_obs_jacobian_3D(vector_1, vector_2):

    [x1, y1, t1] = vector_1
    [x2, y2, t2] = vector_2

    Jx = [-np.cos(t1), -np.sin(t1), 0, np.cos(t1), np.sin(t1), 0]
    Jy = [np.sin(t1), -np.cos(t1), 0, -np.sin(t1), np.cos(t1), 0]
    Jt = [0, 0, 1, 0, 0, -1]

    J = [Jx, Jy, Jt]
    return J


# def dual_relative_obs_jacobian_3D(vector_1, vector_2):
#
#     [x1, y1, t1] = vector_1
#     [x2, y2, t2] = vector_2
#
#     Jx = [-np.cos(t1), -np.sin(t1), np.sin(t1) * (x1 - x2) - np.cos(t1) * (y1 - y2), np.cos(t1), np.sin(t1), 0]
#     Jy = [np.sin(t1), -np.cos(t1), np.cos(t1) * (x1 - x2) - np.sin(t1) * (y1 - y2), -np.sin(t1), np.cos(t1), 0]
#     if t1 == t2:
#         Jt = [0, 0, 0, 0, 0, 0]
#     else:
#         Jt = [0, 0, -(np.cos(t1) * np.sin(t2) - np.cos(t2) * np.sin(t1)) / (
#                     1 - (np.cos(t1) * np.cos(t2) + np.sin(t1) * np.sin(t2)) ** 2) ** 0.5,
#               0, 0, (np.cos(t1) * np.sin(t2) - np.cos(t2) * np.sin(t1)) / (
#                           1 - (np.cos(t1) * np.cos(t2) + np.sin(t1) * np.sin(t2)) ** 2) ** 0.5]
#
#     J = [Jx, Jy, Jt]
#     return J


    # Define measurement jacobian for camera
def h_camera_3D(H, x, agent1, agent2, dim_state, dim_obs):
    agent1_row_min = dim_state * agent1
    agent1_row_max = agent1_row_min + dim_obs
    agent2_row_min = dim_state * agent2
    agent2_row_max = agent2_row_min + dim_obs

    x1 = x[agent1_row_min:agent1_row_max]
    x2 = x[agent2_row_min:agent2_row_max]

    Jacobian = np.array(dual_relative_obs_jacobian_3D(x1, x2))
    H[:, agent1_row_min:agent1_row_max] = Jacobian[:, 0:dim_obs]
    H[:, agent2_row_min:agent2_row_max] = Jacobian[:, dim_obs:2*dim_obs]
    return H


    # Define measurement jacobian for camera
def h_camera(H, x, agent1, agent2, dim_state, dim_obs):
    agent1_row_min = dim_state * agent1
    agent1_row_max = agent1_row_min + dim_obs
    agent2_row_min = dim_state * agent2
    agent2_row_max = agent2_row_min + dim_obs

    x1 = x[agent1_row_min:agent1_row_max]
    x2 = x[agent2_row_min:agent2_row_max]

    Jacobian = np.array(dual_relative_obs_jacobian(x1, x2))
    H[:, agent1_row_min:agent1_row_max] = Jacobian[:, 0:dim_obs]
    H[:, agent2_row_min:agent2_row_max] = Jacobian[:, dim_obs:2*dim_obs]
    return H