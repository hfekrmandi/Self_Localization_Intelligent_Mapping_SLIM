import rospy
from std_msgs.msg import String
from gazebo_msgs.msg import ModelStates
import os
import roslib
import sys
import rospy
import numpy as np
import cv2
import tf2_ros
import datetime
import time
from geometry_msgs.msg import PoseStamped
from geometry_msgs.msg import PoseWithCovarianceStamped
import tf2_geometry_msgs
from dse_msgs.msg import PoseMarkers
from scipy.spatial.transform import Rotation as R
import dse_lib
import copy

class GazeboModel(object):
    def __init__(self, robots_name_list = ['mobile_base_2', 'mobile_base_1']):

        # We wait for the topic to be available and when it is then retrive the index of each model
        # This was separated from callbal to avoid doing this in each callback
        self._robots_models_dict = {}
        self._robots_pose_list = []
        self._robots_index_dict = {}
        self._robots_name_list = robots_name_list

        self.rate_get_robot_index_object = rospy.Rate(1) # 1Hz

        self.get_robot_index()

        # We now start the suscriber once we have the indexes of each model
        rospy.Subscriber("/gazebo/model_states", ModelStates, self.callback)

    def get_robot_index(self):


        data = None
        found_all_robot_names = False
        while not found_all_robot_names:
            rospy.loginfo("Retrieveing Model indexes ")
            try:
                data = rospy.wait_for_message("/gazebo/model_states", ModelStates, timeout=5)
                # Save it in the format {"robot1":4,"robot2":2}
                if data:
                    # Here we have model_states data, but not guarantee that the models we want are there

                    for robot_name in self._robots_name_list:
                        robot_name_found = self.update_robot_index(data, robot_name)
                        if robot_name_found:
                            pass
                        else:
                            break

                    found_all_robot_names = len(self._robots_index_dict) == len(self._robots_name_list)
                else:
                    rospy.loginfo("Topic /gazebo/model_states NOT Ready yet, trying again ")

            except Exception as e:
                s = str(e)
                rospy.loginfo("Error in get_robot_index = "+ s)

            self.rate_get_robot_index_object.sleep()

        assert found_all_robot_names, "NOT all the robot names were found"
        rospy.loginfo("Final robots_index_dict =  %s ", str(self._robots_index_dict))

    def update_robot_index(self,data, robot_name):
        try:
            index = data.name.index(robot_name)
            self._robots_index_dict[robot_name] = index
            found = True
        except ValueError:
            rospy.loginfo("Robot Name="+str(robot_name)+", is NOT in model_state, trying again")
            found = False

        return found

    def callback(self,data):

        for robot_name in self._robots_name_list:
            # Retrieve the corresponding index
            robot_name_found = self.update_robot_index(data, robot_name)
            if robot_name_found:
                data_index = self._robots_index_dict[robot_name]
                # Get the pose data from theat index
                try:
                    data_pose = data.pose[data_index]
                except IndexError:
                    rospy.logwarn("The model with data index "+str(data_index)+", something went wrong.")
                    data_pose = None
            else:
                data_pose = None
            # Save the pose inside the dict {"robot1":pose1,"robot2":pose2}
            self._robots_models_dict[robot_name] = data_pose


    def get_model_pose(self,robot_name):

        pose_now = None

        try:
            pose_now = self._robots_models_dict[robot_name]
        except Exception as e:
            s = str(e)
            rospy.loginfo("Error, The _robots_models_dict is not ready = "+ s)

        return pose_now


def listener():
    rospy.init_node('listener', anonymous=True)
    robots_name_list = ['turtle1', 'turtle2']
    gz_model = GazeboModel(robots_name_list)
    rate = rospy.Rate(1) # 10hz
    while not rospy.is_shutdown():
        for robot_name in robots_name_list:
            pose_now = gz_model.get_model_pose(robot_name)
            print("POSE NOW ROBOT ="+robot_name+"==>"+str(pose_now))
        rate.sleep()
    #rospy.spin()


def noisy_transform(transform):
    true_position = transform.transform.translation
    true_orientation = transform.transform.rotation
    true_xyz = np.array([true_position.x, true_position.y, true_position.z])
    true_eul = dse_lib.quat_from_pose2eul(true_orientation)
    true_state = np.concatenate((true_xyz, true_eul))
    true_distance = np.linalg.norm(true_xyz)

    # add = [0, 0, 0, 0, 0, 0]
    # mult = [1, 1, 1, 1, 1, 1]
    noise = np.random.multivariate_normal([0, 0, 0, 0, 0, 0], dse_lib.R_from_range(true_distance))
    sim_state = true_state + noise

    [true_position.x, true_position.y, true_position.z] = sim_state[0:3]
    true_orientation = dse_lib.euler2quat_from_pose(true_orientation, sim_state[3:6, None])
    covariance = dse_lib.covariance_to_ros_covariance(dse_lib.R_from_range(true_distance))
    return transform, covariance


def transform_to_pose_stamped_covariance(transform, covariance):
    pose_stamped = PoseWithCovarianceStamped()
    pose_stamped.header = transform.header
    pose_stamped.pose.pose.position = transform.transform.translation
    pose_stamped.pose.pose.orientation = transform.transform.rotation
    pose_stamped.pose.covariance = covariance
    return pose_stamped


def transform_to_pose_stamped(transform):
    pose_stamped = PoseStamped()
    pose_stamped.header = transform.header
    pose_stamped.pose.position = transform.transform.translation
    pose_stamped.pose.orientation = transform.transform.rotation
    return pose_stamped


def pose_stamped_to_pose_stamped_covariance(pose, covariance):
    pose_covar = PoseWithCovarianceStamped()
    pose_covar.header = pose.header
    pose_covar.pose.pose.position = pose.pose.position
    pose_covar.pose.pose.orientation = pose.pose.orientation
    pose_covar.pose.covariance = covariance
    return pose_covar


def object_pose_in_world(object_tf, tfBuffer, world_tf='world'):
    object_in_world_xfm = tfBuffer.lookup_transform(world_tf, object_tf, rospy.Time(0))
    pose_stamped = transform_to_pose_stamped(object_in_world_xfm)
    return pose_stamped
