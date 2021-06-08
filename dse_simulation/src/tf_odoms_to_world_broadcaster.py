#! /usr/bin/env python3
import rospy
import time
import tf
import gazebo_lib


def handle_turtle_pose(pose_msg, robot_name):
    br = tf.TransformBroadcaster()

    br.sendTransform((pose_msg.position.x, pose_msg.position.y, pose_msg.position.z),
                     (pose_msg.orientation.x, pose_msg.orientation.y, pose_msg.orientation.z, pose_msg.orientation.w),
                     rospy.Time.now(),
                     robot_name,
                     "/world")


def publisher_of_tf():
    rospy.init_node('publisher_of_tf_node', anonymous=True)

    # Get parameters from launch file
    #robot_name_list = rospy.get_param('~robot_names')
    robot_name_list = ['tb3_0', 'tb3_1', 'tb3_2']
    tf_name_list = []
    for robot in robot_name_list:
       tf_name_list.append(robot + '/odom')
    gazebo_model_object = gazebo_lib.GazeboModel(robot_name_list)

    # Leave enough time to be sure the Gazebo Model logs have finished
    time.sleep(1)
    rospy.loginfo("Ready..Starting to Publish TF data now...")

    rate = rospy.Rate(5)  # 5hz
    while not rospy.is_shutdown():
        for i in range(len(robot_name_list)):
            robot_name = robot_name_list[i]
            tf_name = tf_name_list[i]

            pose_now = gazebo_model_object.get_model_pose(robot_name)
            if not pose_now:
                print("The " + str(robot_name) + "'s Pose is not yet available...Please try again later")
            else:
                handle_turtle_pose(pose_now, tf_name)
        rate.sleep()


if __name__ == '__main__':
    try:
        publisher_of_tf()
    except rospy.ROSInterruptException:
        pass