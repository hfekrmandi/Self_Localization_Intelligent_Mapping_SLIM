# Autonomous-GNC-MAS

To run this package: 
You will also need to pull the following ROS packages:
https://github.com/ROBOTIS-GIT/turtlebot3
https://github.com/ROBOTIS-GIT/turtlebot3_msgs
https://github.com/ROBOTIS-GIT/turtlebot3_simulations

And you will need to specify a type of turtlebot. The type with a camera is the waffle_pi, so:
export TURTLEBOT3_MODEL=waffle_pi

Then you can run the launch file aruco_test.launch, and you should see the robot's camera view and any identified tags in said images. This launch file will run an aruco image processing node to detect and estimate the pose of tags, outputting that pose over the /dse/pose_markers topic. 

Copied MATLAB code from TRIM repository. To test, run the dse_sim_3_robots.launch file. This will launch 3 nodes:

- aruco_pose_estimate in arucoPoseEstimation
  - 
- information_filter_node in information_filter
- direct_estimator_node in direct_estimator
