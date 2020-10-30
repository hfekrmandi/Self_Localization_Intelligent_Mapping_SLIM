#!/usr/bin/env python
import roslib
roslib.load_manifest('dse_simulation')
import rospy

import tf
from gazebo_msgs.msg import ModelStates
import dse_lib
import dse_constants


def handle_turtle_pose(msg):
    gzbo_name = rospy.get_param('~gazebo_name')
    tf_name = rospy.get_param('~tf_name')
    #gzbo_name = 'tb3_0'
    #tf_name = 'tb3_0/odom'
    n = len(msg.name)
    br = tf.TransformBroadcaster()
    for i in range(n):
        if msg.name[i] == gzbo_name:
            position = (0, 0, 0)
            orientation = tf.transformations.quaternion_from_euler(0, 0, 0)
            br.sendTransform(position, orientation, rospy.Time.now(), tf_name, "world")
            break


if __name__ == '__main__':
    rospy.init_node('world_tf_broadcaster')
    rospy.Subscriber('/gazebo/model_states', ModelStates, handle_turtle_pose, queue_size=10)
    rospy.spin()
