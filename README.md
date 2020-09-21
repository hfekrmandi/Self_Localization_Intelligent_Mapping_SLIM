# Autonomous-GNC-MAS

To run this package: 
You will also need to pull the following ROS packages:

1. https://github.com/ROBOTIS-GIT/turtlebot3
2. https://github.com/ROBOTIS-GIT/turtlebot3_msgs
3. https://github.com/ROBOTIS-GIT/turtlebot3_simulations

And you will need to specify a type of turtlebot. The type with a camera is the waffle_pi, so run this in the terminal.

`export TURTLEBOT3_MODEL=waffle_pi`

Then you can run the launch file aruco_test.launch,

`roslaunch  aruco_test.launch`

and you should see the robot's camera view and any identified tags in said images. This launch file will run an aruco image processing node to detect and estimate the pose of tags, outputting that pose over the /dse/pose_markers topic. 
