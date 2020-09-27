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
- dse_sim_3_robots.launch - Only 1 robot, but runs an information filter in the background.
- dse_sim_3_robots_test_plot.launch - Instead of running a gazebo, this file uses a simulation node to create pose estimates for a robot, useful for testing/debugging.
- dse_sim_3_robots_world_only.launch - Launches the gazebo world and robot, but no other nodes. Useful for testing/debugging individual nodes. 


Node descriptions - 
- aruco_pose_estimation_node in aruco_pose_estimation
	- Inputs: Camera images
	- Outputs: Poses and IDs of all observed tags
	- What it does: uses the aruco library to find the pose of each tag, and publishes that pose in the robot's coordinate frame
- information_filter_node in information_filter
	- Inputs: Poses and IDs of all observed tags
	- Outputs: Information filter prior and measurement (Y_01, y_01, delta_I and delta_i)
	- What it does: Computes a step of the information filter
	- Reference information filter here
- direct_estimator_node in direct_estimator
	- Inputs: Information filter prior and measurement (Y_01, y_01, delta_I and delta_i)
	- Outputs: Information filter full estimate (Y_00 and y_00)
	- What it does: Adds the prior and measurement to create the result, and publishes that. Replaces the consensus since it isn't implemented yet. 
- dse_plotting_node in rqt_plot_output.py
	- Inputs: Poses and IDs of all information filter estimates, measurements, and true states
	- Outputs: Individual poses of all information filter estimates, measurements, and true states
	- What it does: Outputs a Pose for each estimate, measurement, and true state (rqt_plot won't seem to display PoseArrays)
- dse_visualization_node in visualize_pose
	- Inputs: Poses and IDs of all information filter estimates and measurements
	- Outputs: Poses of all information filter estimates and measurements
	- What it does: Outputs a PoseArray for all estimates and measurements (rviz can't display the custom pose array with an ID)


Debugging the code - 
- In Pycharm (or any debugging python IDE)
	- In a terminal, source ~/simulation_ws/devel/setup.bash
	- Then start the IDE from that terminal (Ensures that it knows about ROS libraries/messages
	- Ensure that you are using Python 2.7
	- Degug any node file as you would a normal Python file. 
	- You can customize launch files to launch all other nodes, or start others individually through terminals
- In RVIZ
	- Make sure that the visualize_pose source file is running (publishes RVIZ-displayable messages)
	- In a terminal, source ~/simulation_ws/devel/setup.bash
	- Start RVIZ with the command rviz
	- On the left side, under Displays -> Global Options, type in odom as the Fixed Frame
	- Then in the bottom left click Add -> By Topic -> /dse/vis/[estimates, measurement, or true] whatever you want to look at. 
	- You can also go into each PoseArray and change the color so the pose arrows are individually identifiable
- in rqt_plot
	- Make sure that the rqt_plot_output.py source file is running (publishes rqt_plot-displayable messages)
	- In a terminal, source ~/simulation_ws/devel/setup.bash
	- Start rqt_plot with the command rqt_plot
	- Under topic, type in the topic name you want to see. 
	- /dse/plt/measurements/[position/[x, y, z], orientation/[x, y, z, w]] are the position and quaternion orientation of the measurement
	- /dse/plt/estimates/[robot, tag]/[position/[x, y, z], orientation/[x, y, z, w]] are the position and quaternion orientation estimates for the robot or tag
	- /dse/plt/true/[robot, tag]/[position/[x, y, z], orientation/[x, y, z, w]] are the position and quaternion orientation true values for the robot or tag
- By adding print statements
	- use print() normally in any node file
	- make sure that in the launch file for each node you have output="screen"

Other files - 
- src/dse_leb.py
	- Library of functions for computing R, H, z, F, Q...
