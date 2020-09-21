import roslib
import sys
import rospy
import numpy as np
import datetime
import time
from geometry_msgs.msg import Pose
from dse_msgs.msg import PoseMarkers
from std_msgs.msg import Float64MultiArray
from std_msgs.msg import MultiArrayLayout
from std_msgs.msg import MultiArrayDimension
from dse_msgs.msg import InfFilterPartials
from dse_msgs.msg import InfFilterResults
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt

import dse_constants

# Covariance based on distance as estimated for the SVGS short-range system
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


# Covariance based n distance as estimated for the aruco system (Needs more testing)
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


# Covariance based n distance as estimated for the aruco system (Needs more testing)
def aruco_R_from_range_3D(range):
    # Assuming linear error with a slope of:
    # [x y z phi theta psi]
    # x = [0.0515; 0.0515; 0.018; 0.1324; 0.1324; 0.1324]; # Degrees
    x = 10*np.transpose([0.01, 0.01, 0.01]) # Radians
    # x = [0.0075; 0.0075; 0.0075; 0.0075; 0.0075; 0.0075]; # 5% of distance

    # Slope values are for 3-sigma error, so dividing by 3
    range = (range + 0.001) * np.eye(3)
    r_std = np.multiply(range, x)
    r_var = np.multiply(r_std, r_std)
    # Compute variance from standard deviation
    return r_var


# Compute the 2D rotation matrix from the angle theta
def theta_2_rotm(theta):
    R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    return R


# Fill and return a pose array with values from the state variable x
def pose_array_from_state(pose_array, x, dim_state, dim_obs):
    num_objs = len(x) / dim_state
    for i in range(num_objs):
        pose = Pose()

        i_low = dim_state * i

        pose.position.x = x[i_low + 0]
        pose.position.y = x[i_low + 1]
        pose.position.z = x[i_low + 2]
        tmp = x[i_low+3:i_low+6, 0]
        r = R.from_euler('zyx', x[i_low+3:i_low+6, 0])
        quat = r.as_quat()

        pose.orientation.x = quat[0]
        pose.orientation.y = quat[1]
        pose.orientation.z = quat[2]
        pose.orientation.w = quat[3]
        pose_array.poses += [pose]
    return pose_array


# Fill in a multi-array ROS message type with a 2D input array
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


# Grab and return a 2D array from a multi-array ROS message
def multi_array_2d_output(multi_arr):
    arr = np.array(multi_arr.data)
    shape = [multi_arr.layout.dim[0].size, multi_arr.layout.dim[1].size]
    mat = arr.reshape(shape)
    return mat


# Compute the observation jacobian H for a 6D-obs system.
# Currently no functions for the angles, DO NOT USE
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


# Compute the observation jacobian H for a 3D-obs system.
def dual_relative_obs_jacobian_3D(x1, x2):

    [x1, y1, t1] = x1
    [x2, y2, t2] = x2

    Jx = [-np.cos(t1), -np.sin(t1), 0, np.cos(t1), np.sin(t1), 0]
    Jy = [np.sin(t1), -np.cos(t1), 0, -np.sin(t1), np.cos(t1), 0]
    Jt = [0, 0, 1, 0, 0, -1]

    J = [Jx, Jy, Jt]
    return J


# If the agent doesn't know about a newly observed agent, extend all variables to accomodate it
def extend_arrays(observed_ids, id_list, Y_11, y_11, dim_state):
    # Compute x and P so that if there are no new agents it doesn't error out
    x_11 = np.linalg.inv(Y_11).dot(y_11)
    P_11 = np.linalg.inv(Y_11)

    for id in observed_ids:
        # If we found a new agent
        if not np.isin(id, id_list):
            id_list = np.concatenate((id_list, [id]))
            dim = len(id_list) * dim_state

            # Extend the information matrix Y
            Y_11_tmp = dse_constants.INF_MATRIX_INITIAL * np.eye(dim)
            Y_11_tmp[0:np.shape(Y_11)[0], 0:np.shape(Y_11)[0]] = Y_11
            Y_11 = Y_11_tmp

            # Extend the information vector y
            y_11_tmp = dse_constants.INF_VECTOR_INITIAL * np.arange(1, dim+1)[:, None]
            y_11_tmp[0:np.shape(y_11)[0]] = y_11
            y_11 = y_11_tmp

            # re-compute x and P to match
            x_11 = np.linalg.inv(Y_11).dot(y_11)
            P_11 = np.linalg.inv(Y_11)
    
    return id_list, Y_11, y_11, P_11, x_11


# Fill in the matrices F and Q:
# F - Motion Jacobian
# Q - Motion Covariance
def fill_FQ(id_list, dt, x_11, dim_state, dim_obs):
    n_stored = len(id_list)
    F_0 = np.zeros((n_stored * dim_state, n_stored * dim_state))
    Q_0 = np.zeros((n_stored * dim_state, n_stored * dim_state))
    
    # Fill in Q and F (Different for waypoint vs. robot)
    for i in range(len(id_list)):
        i_low = dim_state * i
        i_high = i_low + dim_state

        # Q is a function of distance traveled in the last time step
        Q_0[i_low:i_high, i_low:i_high] = q_distance(dt, x_11, i, dim_state)

        # If we are looking at ID 0, it is a waypoint and as such doesn't move (F is identity matrix)
        if id_list[i] == 0:
            F_0[i_low:i_high, i_low:i_high] = f_eye(dim_state)
        else:
            # Else use the unicycle model
            if dim_obs == 3:
                F_0[i_low:i_high, i_low:i_high] = f_unicycle_3D(dt, x_11, i, dim_state)
            else:
                F_0[i_low:i_high, i_low:i_high] = f_unicycle(dt, x_11, i, dim_state)

    return F_0, Q_0


# Fill in the matrices R and H, as well as the vector z
# R - Measurement Covariance
# H - Measurement Jacobian
# z - The measurement itself
def fill_RHz(id_list, my_id, observed_ids, observed_poses, x_11, euler_order, dim_state, dim_obs, R_var = 0.001):

    # Define the sizes of each variable
    n_stored = len(id_list)
    n_obs = len(observed_ids)
    R_0 = R_var * np.eye(n_obs * dim_obs)
    H_0 = np.zeros((n_obs * dim_obs, n_stored * dim_state))
    z_0 = np.zeros((n_obs * dim_obs, 1))
    
    # Fill in H and Z
    for i in range(len(observed_ids)):
        id = observed_ids[i]
        index = np.where(id_list == id)[0][0]           # Index of observed agent
        obs_index = np.where(id_list == my_id)[0][0]    # Index of observing agent

        i_low = dim_obs * i
        i_high = i_low + dim_obs

        # Compute the euler angles from the quaternion paseed in
        quat = np.zeros(4)
        quat[0] = observed_poses[i].orientation.x
        quat[1] = observed_poses[i].orientation.y
        quat[2] = observed_poses[i].orientation.z
        quat[3] = observed_poses[i].orientation.w
        r = R.from_quat(quat)
        # https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.transform.Rotation.from_quat.html
        z_eul = r.as_euler(euler_order)

        # Different functions for 3D vs. 6D observation
        if dim_obs == 3:
            z_pos = np.array([observed_poses[i].position.x, observed_poses[i].position.y])
            z_eul = [z_eul[0]]
            dist = np.linalg.norm(z_pos)
            R_0[i_low:i_high, i_low:i_high] = 1 * aruco_R_from_range_3D(dist)
            H_0 = h_camera_3D(H_0, x_11, obs_index, index, dim_state, dim_obs)
        else:
            z_pos = np.array([observed_poses[i].position.x, observed_poses[i].position.y, observed_poses[i].position.z])
            dist = np.linalg.norm(z_pos)
            R_0[i_low:i_high, i_low:i_high] = 1 * aruco_R_from_range(dist)
            H_0 = h_camera(H_0, x_11, obs_index, index, dim_state, dim_obs)

        z_0[i_low:i_high] = np.concatenate((z_pos, z_eul))[:, None]

    return R_0, H_0, z_0


# Fill in the matrix B and the vector u
# B - Control matrix
# u - Control signals
# This function is not ready and has not been tested
def fill_Bu(id_list, my_id, ctrl_ids, x, ctrl, dim_state, dim_obs):

    # Define the sizes of each variable
    n_stored = len(id_list)
    B = np.zeros((n_stored * dim_state, n_stored * dim_state))
    u = np.zeros((n_stored * dim_state, 1))

    # For each agent that we have a control signal:
    for i in range(len(ctrl_ids)):
        id = ctrl_ids[i]
        index = np.where(id_list == id)[0][0]           # Index of observed agent

        i_low = dim_state * index
        i_high = i_low + dim_obs

        B[i_low:i_high, i_low:i_high] = B_eye(dim_obs)
        if dim_obs == 3:
            u[i_low:i_high] = np.array([ctrl[i][0], ctrl[i][1], ctrl[i][5]])[:, None]
        else:
            u[i_low:i_high] = ctrl[i]

    return B, u


# Define the measurement jacobian for a camera (3D-observation)
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


# Define the measurement jacobian for a camera
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


# Helper functions
def state_from_id(x, id_list, id, dim_state):
    index = np.where(id_list == id)[0][0]
    i_low = dim_state * index
    i_high = i_low + dim_state
    return x[i_low:i_high]


def cov_from_id(P, id_list, id, dim_state):
    index = np.where(id_list == id)[0][0]
    i_low = dim_state * index
    i_high = i_low + dim_state
    return P[i_low:i_high, i_low:i_high]


# Define motion jacobian for unicycle robot (3D-observation)
def f_unicycle_3D(dt, x, agent1, dim_state):
    agent1_row_min = dim_state * agent1
    agent1_row_max = agent1_row_min + dim_state

    x1 = x[agent1_row_min:agent1_row_max]

    F = np.eye(dim_state)
    w = x1[5]

    # Using the fundamental theorem of engineering, sin(x) = x,
    # sin(a*x)/x = a (Really only when x is 0)
    if w == 0:
        F[0, 3] = dt
        F[0, 4] = 0
        F[1, 3] = 0
        F[1, 4] = dt

        F[3, 3] = 1
        F[3, 4] = 0
        F[4, 3] = 0
        F[4, 4] = 1
    else:
        F[0, 3] = np.sin(w*dt) / w
        F[0, 4] = -(1 - np.cos(w*dt)) / w
        F[1, 3] = (1 - np.cos(w*dt)) / w
        F[1, 4] = np.sin(w*dt) / w

        F[3, 3] = np.cos(w*dt)
        F[3, 4] = -np.sin(w*dt)
        F[4, 3] = np.sin(w*dt)
        F[4, 4] = np.cos(w*dt)

    F[2, 5] = dt
    return F


# Define motion jacobian for unicycle robot
def f_unicycle(dt, x, agent1, dim_state):
    agent1_row_min = dim_state * agent1
    agent1_row_max = agent1_row_min + dim_state

    x1 = x[agent1_row_min:agent1_row_max]

    F = np.eye(dim_state)
    w = x1[9]

    # Using the fundamental theorem of engineering, sin(x) = x,
    # sin(a*x)/x = a (Really only when x is 0)
    if w == 0:
        F[0, 6] = dt
        F[0, 7] = 0
        F[1, 6] = 0
        F[1, 7] = dt

        F[6, 6] = 1
        F[6, 7] = 0
        F[7, 6] = 0
        F[7, 7] = 1
    else:
        F[0, 6] = np.sin(w*dt) / w
        F[0, 7] = -(1 - np.cos(w*dt)) / w
        F[1, 6] = (1 - np.cos(w*dt)) / w
        F[1, 7] = np.sin(w*dt) / w

        F[6, 6] = np.cos(w*dt)
        F[6, 7] = -np.sin(w*dt)
        F[7, 6] = np.sin(w*dt)
        F[7, 7] = np.cos(w*dt)

    block = dt * np.eye(3)
    F[3:6, 9:12] = block
    return F


# Define stationary jacobian for waypoints
def f_eye(dim_state):
    F = np.eye(dim_state)
    return F


# Define motion model covariance (distance-based)
def q_distance(dt, x, agent1, dim_state):
    i_low = dim_state * agent1
    i_high = i_low + dim_state

    # Q is (dt * (x_dot + 0.001) * 5%) ^ 2
    Q_pos = (dt * (np.linalg.norm(x[i_low+3:i_low+5]) + 0.001) * 0.05) ** 2
    Q_theta = (dt * (np.linalg.norm(x[i_low+5]) + 0.001) * 0.05) ** 2
    print('Q_pos: ' + str(Q_pos) + ' and Q_theta: ' + str(Q_theta))

    # Define the velocity covariance
    Q = 1 * np.eye(dim_state)
    Q[3:5, 3:5] = dse_constants.MOTION_BASE_COVARIANCE / (dt ** 2) * np.eye(2)
    Q[5, 5] = dse_constants.MOTION_BASE_COVARIANCE / (dt ** 2)

    # if Q_pos or Q_theta is <= 0, problems occur
    if Q_pos > 0:
        Q[0:2, 0:2] = Q_pos * np.eye(2)
    if Q_theta > 0:
        Q[2, 2] = Q_theta
    return Q


# Define motion model covariance (static)
def q_const(dim_state, var=0.000001):
    Q = var * np.eye(dim_state)
    return Q


# Direct control matrix
def B_eye(dim_state):
    B = np.eye(dim_state)
    return B


# Unknown control matrix
def B_zeros(dim_state):
    B = np.zeros((dim_state, dim_state))
    return B
