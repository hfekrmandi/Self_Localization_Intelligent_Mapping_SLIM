#!/usr/bin/env python2
import roslib
roslib.load_manifest('dse_simulation')
import rospy

import tf
from gazebo_msgs.msg import ModelStates
from gazebo_msgs.msg import LinkStates
from geometry_msgs.msg import PoseArray
import dse_lib
import dse_constants

class turtlebot_tf_transforms:

    # Define initial/setup values
    def __init__(self):

        # Get parameters from launch file
        self.gzbo_name = rospy.get_param('~gazebo_name')
        self.tf_name = rospy.get_param('~tf_name')
        self.dim_state = rospy.get_param('~dim_state')
        # self.gzbo_name = 'tb3_0'
        # self.tf_name = 'tb3_0/odom'
        # self.dim_state = 6

        self.ros_prefix = self.gzbo_name
        if len(self.ros_prefix) != 0 and self.ros_prefix[0] != '/':
            self.ros_prefix = '/' + self.ros_prefix

        self.gzbo_link_sub = rospy.Subscriber('/gazebo/link_states', LinkStates, self.gzbo_true_callback)
        self.gzbo_model_sub = rospy.Subscriber('/gazebo/model_states', ModelStates, self.handle_turtle_pose, queue_size=10)
        self.gzbo_vis_pub = rospy.Publisher(self.ros_prefix + '/dse/vis/gazebo_true', PoseArray, queue_size=10)

        if self.dim_state == 6:
            self.dim_obs = 3
        elif self.dim_state == 12:
            self.dim_obs = 6
        else:
            rospy.signal_shutdown('invalid state dimension passed in')

        # Define static variables
        self.dt = 0.1
        self.t_last = rospy.get_time()
        self.br = tf.TransformBroadcaster()
        self.zero_pos = (0, 0, 0)
        self.zero_ori = tf.transformations.quaternion_from_euler(0, 0, 0)

    def handle_turtle_pose(self, msg):
        n = len(msg.name)
        for i in range(n):
            if msg.name[i] == self.gzbo_name:
                self.br.sendTransform(self.zero_pos, self.zero_ori, rospy.Time.now(), self.tf_name, "world")
                break

    # Create pose_array for the information results
    def gzbo_true_callback(self, data):
        n = len(data.name)
        got_tag = False
        got_robot = False
        if self.ros_prefix == '':
            turtlebot_name = 'turtlebot3_waffle_pi::base_footprint'
        else:
            turtlebot_name = self.ros_prefix + '::base_footprint'

        for i in range(n):
            if data.name[i] == 'aruco_marker_0::link':
                tag_state = dse_lib.state_from_pose_3D(data.pose[i])
                got_tag = True
            elif data.name[i] == turtlebot_name:
                robot_state = dse_lib.state_from_pose_3D(data.pose[i])
                got_robot = True

        if got_tag and got_robot:
            diff = dse_lib.agent2_to_frame_agent1_3D(robot_state, tag_state)
            poses = PoseArray()
            poses.header.stamp = rospy.Time.now()
            if self.ros_prefix == '':
                poses.header.frame_id = 'base_footprint'
            else:
                poses.header.frame_id = self.ros_prefix + '/base_footprint'
            poses = dse_lib.pose_array_from_measurement(poses, diff, self.dim_obs)
            self.gzbo_vis_pub.publish(poses)


if __name__ == '__main__':
    rospy.init_node('world_tf_broadcaster')
    tf_gen = turtlebot_tf_transforms()
    rospy.spin()
