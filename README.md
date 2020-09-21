# Autonomous-GNC-MAS

Installing this code - 
Option A: Use your catkin_ws in ~/
	Henceforth this will be referred to as simulation_ws, just keep that in mind
Option B: Create a new ROS workspace for this code:
	Create a folder such as simulation_ws in ~/
	Cd into it and create a src folder (mkdir src)
	run catkin_make

Then, cd into ~/simulation_ws/src/ and clone this github repo. Also clone these 3 turtlebot ROS packages:
https://github.com/ROBOTIS-GIT/turtlebot3
https://github.com/ROBOTIS-GIT/turtlebot3_msgs
https://github.com/ROBOTIS-GIT/turtlebot3_simulations

Then run cd .. (you should be in ~/simulation_ws), and run catkin_make again
Your directories should look like:
~/simulation_ws
	- build
	- devel
		- setup.bash
	- src
		- Autonomous-GNC-MAS
			- Autonomous-GNC-MAS
			- dse_msgs
			- dse_simulation
			- README.md
		- CMakeLists.txt
		- tureltbot3
		- turtlebot3_msgs
		- turtlebot3_simulations

Now, each time you open a new terminal make sure to run:
export TURTLEBOT3_MODEL=waffle_pi
source ~/simulation_ws/devel/setup.bash

Now you should be able to run a launch file, ex:
roslaunch dse_simulation aruco_test.launch

This launch file starts a gazebo world, spawns in a turtlebot and a cube with aruco tags on each face. It also starts a camera node, aruco_pose_estimation, which will use the turtlebot's camera to generate pose estimates of any observed tags. 

Other launch files:
dse_sim_3_robots.launch - Only 1 robot, but runs an information filter in the background.
dse_sim_3_robots_test_plot.launch - Instead of running a gazebo, this file uses a simulation node to create pose estimates for a robot, useful for testing/debugging.
dse_sim_3_robots_world_only.launch - Launches the gazebo world and robot, but no other nodes. Useful for testing/debugging individual nodes. 


Node descriptions - 

- aruco_pose_estimation_node in aruco_pose_estimation
- information_filter_node in information_filter
- direct_estimator_node in direct_estimator

