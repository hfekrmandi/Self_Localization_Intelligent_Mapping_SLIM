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
    x = 2*np.transpose([0.01, 0.01, 0.01, 0.01, 0.01, 0.01]) # Radians
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
    x = 2*np.transpose([0.01, 0.01, 0.01]) # Radians
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


# Converts a quaternion into euler angles, using the euler order described in dse_constants.py
def quat2eul(quat):
    r = R.from_quat(quat)
    eul = r.as_euler(dse_constants.EULER_ORDER)
    return eul


# Converts euler angles into a quaternion, using the euler order described in dse_constants.py
def eul2quat(eul):
    r = R.from_euler(dse_constants.EULER_ORDER, eul[:, 0])
    quat = r.as_quat()
    return quat


# Expects a quaternion in the form: orientation.x,y,z,w
def quat_from_pose2eul(orientation):
    quat = [0, 0, 0, 0]
    quat[0] = orientation.x
    quat[1] = orientation.y
    quat[2] = orientation.z
    quat[3] = orientation.w
    eul = quat2eul(quat)
    return eul


# Expects a quaternion in the form: orientation.x,y,z,w
def euler2quat_from_pose(orientation, euler):
    quat = eul2quat(euler)
    orientation.x = quat[0]
    orientation.y = quat[1]
    orientation.z = quat[2]
    orientation.w = quat[3]
    return orientation


def state_12D_to_6D(x_12D):
    num_objs = int(len(x_12D) / 12)
    x_6D = np.zeros((num_objs * 6, 1))
    for i in range(num_objs):
        i_6D_low = 6 * i
        i_12D_low = 12 * i
        x_6D[i_6D_low + 0] = x_12D[i_12D_low + 0]
        x_6D[i_6D_low + 1] = x_12D[i_12D_low + 1]
        x_6D[i_6D_low + 2] = x_12D[i_12D_low + 3]
        x_6D[i_6D_low + 3] = x_12D[i_12D_low + 6]
        x_6D[i_6D_low + 4] = x_12D[i_12D_low + 7]
        x_6D[i_6D_low + 5] = x_12D[i_12D_low + 9]
    return x_6D


# Expects a pose in the form: x, y, z, w
def state_from_pose(pose):
    euler_orientation = quat_from_pose2eul(pose.orientation)
    x = np.array([pose.position.x, pose.position.y, pose.position.z, euler_orientation])[:, None]
    return x


# Expects a pose in the form: x, y, z, w
def state_from_pose_3D(pose):
    euler_orientation = quat_from_pose2eul(pose.orientation)
    x = np.array([pose.position.x, pose.position.y, euler_orientation[0]])[:, None]
    return x


# Expects a state in the form: x, y, z, eul_z, eul_y, eul_x
def pose_from_state(x):
    pose = Pose()
    pose.position.x = x[0, 0]
    pose.position.y = x[1, 0]
    pose.position.z = x[2, 0]
    pose.orientation = euler2quat_from_pose(pose.orientation, x[3:6])
    return pose


# Expects a state in the form: x, y, eul_z
def pose_from_state_3D(x):
    pose = Pose()
    pose.position.x = x[0, 0]
    pose.position.y = x[1, 0]
    pose.position.z = 0
    euler_angles = np.array([x[2, 0], 0, 0])[:, None]
    pose.orientation = euler2quat_from_pose(pose.orientation, euler_angles)
    return pose


# Fill and return a pose array with values from the state variable x
def pose_array_from_state(pose_array, x, dim_state, dim_obs):
    num_objs = int(len(x) / dim_state)
    for i in range(num_objs):

        i_low = dim_state * i
        i_high = i_low + dim_obs
        x_i = x[i_low:i_high]

        if dim_state == 6:
            pose_array.poses.append(pose_from_state_3D(x_i))
        else:
            pose_array.poses.append(pose_from_state(x_i))

    return pose_array


# Fill and return a pose array with values from the state variable x
def state_from_pose_array(pose_array, dim_state, dim_obs):
    num_objs = np.shape(pose_array.poses)[0]
    x = np.zeros((num_objs * dim_state, 1))

    for i in range(num_objs):

        i_low = dim_state * i
        i_high = i_low + dim_obs

        if dim_state == 6:
            x[i_low:i_high] = state_from_pose_3D(pose_array.poses[i])
        else:
            x[i_low:i_high] = state_from_pose(pose_array.poses[i])

    return x


# Expects a pose in the form: x, y, z, w
def measurement_from_pose(pose):
    euler_orientation = quat_from_pose2eul(pose.orientation)
    x = np.array([pose.position.x, pose.position.y, pose.position.z, euler_orientation])[:, None]
    return x


# Expects a pose in the form: x, y, z, w
def measurement_from_pose_3D(pose):
    euler_orientation = quat_from_pose2eul(pose.orientation)
    x = np.array([pose.position.x, pose.position.y, euler_orientation[0]])[:, None]
    return x


# Expects a state in the form: x, y, z, eul_z, eul_y, eul_x
def pose_from_measurement(x):
    pose = Pose()
    pose.position.x = x[0, 0]
    pose.position.y = x[1, 0]
    pose.position.z = x[2, 0]
    pose.orientation = euler2quat_from_pose(pose.orientation, x[3:6])
    return pose


# Expects a state in the form: x, y, eul_z
def pose_from_measurement_3D(x):
    pose = Pose()
    pose.position.x = x[0, 0]
    pose.position.y = x[1, 0]
    pose.position.z = 0
    euler_angles = np.array([x[2, 0], 0, 0])[:, None]
    pose.orientation = euler2quat_from_pose(pose.orientation, euler_angles)
    return pose


# Fill and return a pose array with values from the measurement z
def pose_array_from_measurement(pose_array, z, dim_obs):
    num_objs = int(len(z) / dim_obs)
    for i in range(num_objs):

        i_low = dim_obs * i
        i_high = i_low + dim_obs
        x_i = z[i_low:i_high]

        if dim_obs == 3:
            pose_array.poses.append(pose_from_measurement_3D(x_i))
        else:
            pose_array.poses.append(pose_from_measurement(x_i))

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


# def observe_agent2_from_agent1_Hz(agent1_global, agent2_global):
#     H = dual_relative_obs_jacobian(agent1_global, agent2_global)
#     z = H.dot(np.concatenate(agent1_global, agent2_global))
#     return z
#
#
# def observe_agent2_from_agent1_Hz_3D(agent1_global, agent2_global):
#     H = dual_relative_obs_jacobian_3D(agent1_global, agent2_global)
#     z = H.dot(np.concatenate(agent1_global, agent2_global))
#     return z


def agent2_to_frame_agent1(agent1_global, agent2_global):
    t1 = agent1_global[0:3]
    r1 = R.from_euler(dse_constants.EULER_ORDER, agent1_global[3:6, 0])
    R1 = r1.as_dcm()

    t2 = agent2_global[0:3]
    r2 = R.from_euler(dse_constants.EULER_ORDER, agent2_global[3:6, 0])
    R2 = r2.as_dcm()

    tz = (np.transpose(R1).dot(t2) - np.transpose(R1).dot(t1))[:, 0]
    Rz = np.transpose(R1).dot(R2)
    rz = R.from_dcm(Rz)
    rz = rz.as_euler(dse_constants.EULER_ORDER)
    z = np.concatenate((tz, rz))[:, None]
    return z


def agent2_to_frame_agent1_3D(agent1_global, agent2_global):
    t1 = agent1_global[0:2, 0]
    R1 = theta_2_rotm(agent1_global[2, 0])

    t2 = agent2_global[0:2, 0]
    R2 = theta_2_rotm(agent2_global[2, 0])

    zt = np.transpose(R1).dot(t2) - np.transpose(R1).dot(t1)
    zR = np.transpose(R1).dot(R2)
    zr = [-np.arctan2(zR[0, 1], zR[0, 0])]
    z = np.concatenate((zt, zr))[:, None]
    return z


def agent2_from_frame_agent1(agent1_in_agent2, agent2_global):
    t1 = agent2_global[0:3]
    r1 = R.from_euler(dse_constants.EULER_ORDER, agent2_global[3:6, 0])
    R1 = r1.as_dcm()

    t2 = agent1_in_agent2[0:3]
    r2 = R.from_euler(dse_constants.EULER_ORDER, agent1_in_agent2[3:6, 0])
    R2 = r2.as_dcm()

    tz = (R1.dot(t2) + t1)[:, 0]
    Rz = R1.dot(R2)
    rz = R.from_dcm(Rz)
    rz = rz.as_euler(dse_constants.EULER_ORDER)
    z = np.concatenate((tz, rz))[:, None]
    return z


def agent2_from_frame_agent1_3D(agent2_global, agent1_in_agent2):
    t1 = agent2_global[0:2, 0]
    R1 = theta_2_rotm(agent2_global[2, 0])

    t2 = agent1_in_agent2[0:2, 0]
    R2 = theta_2_rotm(agent1_in_agent2[2, 0])

    tz = (R1.dot(t2) + t1)
    Rz = R1.dot(R2)
    rz = [np.arctan2(Rz[0, 1], Rz[0, 0])]
    z = np.concatenate((tz, rz))[:, None]
    return z


def relative_states_from_global_3D(rel_id, ids, states, dim_state, dim_obs):
    rel_index = np.where(ids == rel_id)[0][0]
    obj_ids = ids[np.where(ids != rel_id)]

    min_index = rel_index * dim_state
    max_index = min_index + dim_state

    rel_state = states[min_index:max_index]
    indices = np.ones(np.shape(states)[0], dtype=bool)
    indices[min_index:max_index] = np.zeros((dim_state))
    obj_states = states[indices, :]

    # print('agent id: ' + str(rel_id) + ' other ids: ' + str(obj_ids))
    # print('agent state: ' + str(rel_state) + ' other states: ' + str(obj_states))

    transformed_states = np.zeros(np.shape(obj_states))
    for i in range(len(obj_ids)):
        min_index = i * dim_state
        max_index = min_index + dim_obs

        obj_state = obj_states[min_index:max_index]
        transformed_state = agent2_to_frame_agent1_3D(rel_state[0:dim_obs, :], obj_state)
        transformed_states[min_index:max_index] = transformed_state

    return obj_ids, transformed_states


# Compute the observation jacobian H for a 6D-obs system.
# Currently no functions for the angles, DO NOT USE
def dual_relative_obs_jacobian(state1, state2):

    [x1, y1, z1, p1, t1, s1] = state1
    [x2, y2, z2, p2, t2, s2] = state2

    Jx = [-np.cos(s1) * np.cos(t1), -np.cos(t1) * np.sin(s1), np.sin(t1), 0, 0, 0,
          np.cos(s1) * np.cos(t1), np.cos(t1) * np.sin(s1), -np.sin(t1), 0, 0, 0]

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


# Compute the observation jacobian H for a 3D-observation system
# Given two state vectors in the global coordinate system, x1 and x2
# What is the jacobian of the local observation of x2 from x1
def dual_relative_obs_jacobian_3D(state1, state2):

    [x1, y1, t1] = state1
    [x2, y2, t2] = state2

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

        # If we are looking at ID 0, it is a waypoint and as such doesn't move (F is identity matrix)
        if id_list[i] == -1:
            Q_0[i_low:i_high, i_low:i_high] = q_distance_3D(dt, x_11, i, dim_state)
            F_0[i_low:i_high, i_low:i_high] = f_eye(dim_state)
        else:
            # Else use the unicycle model
            if dim_obs == 3:

                # Q is a function of distance traveled in the last time step
                Q_0[i_low:i_high, i_low:i_high] = q_distance_3D(dt, x_11, i, dim_state)
                F_0[i_low:i_high, i_low:i_high] = f_unicycle_3D(dt, x_11, i, dim_state)
            else:
                # Q is a function of distance traveled in the last time step
                Q_0[i_low:i_high, i_low:i_high] = q_distance(dt, x_11, i, dim_state)
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

        # Compute the euler angles from the quaternion passed in
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
            H_0 = h_camera_3D(H_0, x_11, i, obs_index, index, dim_state, dim_obs)
        else:
            z_pos = np.array(observed_poses[i].position.x, observed_poses[i].position.y, observed_poses[i].position.z)
            dist = np.linalg.norm(z_pos)
            R_0[i_low:i_high, i_low:i_high] = 1 * aruco_R_from_range(dist)
            H_0 = h_camera(H_0, x_11, i, obs_index, index, dim_state, dim_obs)

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
            u[i_low+3:i_high+3] = np.array([ctrl[i][0], ctrl[i][1], ctrl[i][4]])[:, None]
        else:
            u[i_low+6:i_high+6] = ctrl[i]

    return B, u


# Define the measurement jacobian for a camera (3D-observation)
def h_camera_3D(H, x, meas_index, agent1, agent2, dim_state, dim_obs):
    agent1_row_min = dim_state * agent1
    agent1_row_max = agent1_row_min + dim_obs
    agent2_row_min = dim_state * agent2
    agent2_row_max = agent2_row_min + dim_obs
    meas_row_min = dim_obs * meas_index
    meas_row_max = meas_row_min + dim_obs

    x1 = x[agent1_row_min:agent1_row_max]
    x2 = x[agent2_row_min:agent2_row_max]

    Jacobian = np.array(dual_relative_obs_jacobian_3D(x1, x2))
    H[meas_row_min:meas_row_max, agent1_row_min:agent1_row_max] = Jacobian[:, 0:dim_obs]
    H[meas_row_min:meas_row_max, agent2_row_min:agent2_row_max] = Jacobian[:, dim_obs:2*dim_obs]
    return H


# Define the measurement jacobian for a camera
def h_camera(H, x, meas_index, agent1, agent2, dim_state, dim_obs):
    agent1_row_min = dim_state * agent1
    agent1_row_max = agent1_row_min + dim_obs
    agent2_row_min = dim_state * agent2
    agent2_row_max = agent2_row_min + dim_obs
    meas_row_min = dim_obs * meas_index
    meas_row_max = meas_row_min + dim_obs

    x1 = x[agent1_row_min:agent1_row_max]
    x2 = x[agent2_row_min:agent2_row_max]

    Jacobian = np.array(dual_relative_obs_jacobian(x1, x2))
    H[meas_row_min:meas_row_max, agent1_row_min:agent1_row_max] = Jacobian[:, 0:dim_obs]
    H[meas_row_min:meas_row_max, agent2_row_min:agent2_row_max] = Jacobian[:, dim_obs:2*dim_obs]
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


# Define motion model covariance (3D-observation, distance-based)
def q_distance_3D(dt, x, agent1, dim_state):
    i_low = dim_state * agent1
    i_high = i_low + dim_state

    # Q is (dt * (x_dot + 0.001) * 5%) ^ 2
    Q_pos = (dt * (np.linalg.norm(x[i_low+3:i_low+5]) + 0.1) * 0.05) ** 2
    Q_theta = (dt * (np.linalg.norm(x[i_low+5]) + 0.1) * 0.05) ** 2

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


# Define motion model covariance (distance-based)
def q_distance(dt, x, agent1, dim_state):
    i_low = dim_state * agent1
    i_high = i_low + dim_state

    # Q is (dt * (x_dot + 0.001) * 5%) ^ 2
    Q_pos = (dt * (np.linalg.norm(x[i_low+6:i_low+9]) + 0.001) * 0.05) ** 2
    Q_theta = (dt * (np.linalg.norm(x[i_low+9:i_low+12]) + 0.001) * 0.05) ** 2

    # Define the velocity covariance
    Q = 1 * np.eye(dim_state)
    Q[6:12, 6:9] = dse_constants.MOTION_BASE_COVARIANCE / (dt ** 2) * np.eye(6)

    # if Q_pos or Q_theta is <= 0, problems occur
    if Q_pos > 0:
        Q[0:3, 0:3] = Q_pos * np.eye(3)
    if Q_theta > 0:
        Q[3:6, 3:6] = Q_theta * np.eye(3)
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


# # Ensures that every agent has the same state variables in the same order
# def get_sorted_agent_states(SIM, objectIndex):
#
#     # Build combined list of ids
#     id_list = []
#     for agent_index in range(np.shape(SIM.OBJECTS)[0]):
#         if SIM.OBJECTS(agent_index).type == OMAS_objectType.agent:
#             for i in range(np.shape(SIM.OBJECTS)[0]):
#                 if objectIndex[agent_index].objectID == SIM.OBJECTS(agent_index).objectID:
#                     agents[objectIndex[agent_index].objectID] = objectIndex[agent_index]
#                     id_list = [id_list, objectIndex[agent_index].memory_id_list]
#
#     # Ensure that the list is sorted, so it is the same on sequential runs
#     id_list = sort(unique(id_list))
#     dim_state = agents[1].dim_state
#     dim_obs = agents[1].dim_obs
#     n_agents = numel(id_list)
#
#     # Ensure all agents' state variables match the master list
#     for agent_index in range(np.shape(agents)[0]):
#         agent = agents[agent_index]
#
#         # If the state variables don't match, add them in
#         if not isequal(agent.memory_id_list, id_list):
#             Y = 0.01 * np.eye(n_agents * dim_state)
#             y = zeros(n_agents * dim_state, 1)
#             I = zeros(n_agents * dim_state)
#             i = zeros(n_agents * dim_state, 1)
#
#             # Move the agents' values to the location specified in the master list
#             for agent_index_1 in range(np.shape(agent.memory_id_list)[0]):
#                 for agent_index_2 in range(np.shape(agent.memory_id_list)[0]):
#
#                     group_index_1 = find(id_list == agent.memory_id_list[agent_index_1])
#                     group_index_2 = find(id_list == agent.memory_id_list[agent_index_2])
#
#                     # Generate indices (to make the assignment setp shorter)
#                     g_row_lo = dim_state * (group_index_1 - 1) + 1
#                     g_row_hi = dim_state * group_index_1
#                     g_col_lo = dim_state * (group_index_2 - 1) + 1
#                     g_col_hi = dim_state * group_index_2
#                     a_row_lo = dim_state * (agent_index_1 - 1) + 1
#                     a_row_hi = dim_state * agent_index_1
#                     a_col_lo = dim_state * (agent_index_2 - 1) + 1
#                     a_col_hi = dim_state * agent_index_2
#
#                     Y[g_row_lo:g_row_hi, g_col_lo:g_col_hi] = agent.memory_Y[a_row_lo:a_row_hi, a_col_lo:a_col_hi]
#                     I[g_row_lo:g_row_hi, g_col_lo:g_col_hi] = agent.memory_I[a_row_lo:a_row_hi, a_col_lo:a_col_hi]
#
#                 y[g_row_lo:g_row_hi] = agent.memory_y[a_row_lo:a_row_hi]
#                 i[g_row_lo:g_row_hi] = agent.memory_i[a_row_lo:a_row_hi]
#
#             agent.memory_id_list = id_list
#             agent.memory_Y = Y
#             agent.memory_y = y
#             agent.memory_I = I
#             agent.memory_i = i


# Ensures that every agent has the same state variables in the same order
def get_sorted_agent_states(array_ids, array_Y, array_y, array_I, array_i, dim_state):

    # Build combined list of ids
        # Still trying to come up with a way to take in data of any form and return vector of ids
    flat_list = [item for sublist in array_ids for item in sublist]
    id_list = np.unique(flat_list)
    id_list = np.sort(id_list)
    n_agents = len(id_list)

    # Ensure all agents' state variables match the master list
    # For each agent that sent in data
    for i in range(len(array_ids)):

        # If the state variable isn't correct, re-order/extend it
        if not np.array_equal(id_list, array_ids[i]):

            # Build an empty set of variables
            # Full-size, ready for data to be inserted
            # Potentially change the initialization?
            master_Y = 0.01 * np.eye(n_agents * dim_state)
            master_y = np.zeros((n_agents * dim_state, 1))
            master_I = np.zeros((n_agents * dim_state, n_agents * dim_state))
            master_i = np.zeros((n_agents * dim_state, 1))

            # Move the agents' values to the location specified in the master list
            # Loop through the input data in chunks of (state_dim x state_dim)
                # Take each block and move it to the correct location in the master arrays

            for agent_row_index in range(len(array_ids[i])):
                for agent_column_index in range(len(array_ids[i])):

                    # Given a chunk of data and a row and column index,
                    # grab the row and column ids of the input data
                    # Find the location of those ids in the master arrays
                    group_row_index = np.where(id_list == array_ids[i][agent_row_index])[0][0]
                    group_column_index = np.where(id_list == array_ids[i][agent_column_index])[0][0]

                    # Generate indices (to make the assignment step shorter)
                    g_row_lo = dim_state * group_row_index
                    g_row_hi = g_row_lo + dim_state
                    g_col_lo = dim_state * group_column_index
                    g_col_hi = g_col_lo + dim_state
                    a_row_lo = dim_state * agent_row_index
                    a_row_hi = a_row_lo + dim_state
                    a_col_lo = dim_state * agent_column_index
                    a_col_hi = a_col_lo + dim_state

                    # Move this chunk of data to the master arrays
                    master_Y[g_row_lo:g_row_hi, g_col_lo:g_col_hi] = array_Y[i][a_row_lo:a_row_hi, a_col_lo:a_col_hi]
                    master_I[g_row_lo:g_row_hi, g_col_lo:g_col_hi] = array_I[i][a_row_lo:a_row_hi, a_col_lo:a_col_hi]

                master_y[g_row_lo:g_row_hi] = array_y[i][a_row_lo:a_row_hi]
                master_i[g_row_lo:g_row_hi] = array_i[i][a_row_lo:a_row_hi]

            array_ids[i] = id_list
            array_Y[i] = master_Y
            array_y[i] = master_y
            array_I[i] = master_I
            array_i[i] = master_i

    return array_ids, array_Y, array_y, array_I, array_i
