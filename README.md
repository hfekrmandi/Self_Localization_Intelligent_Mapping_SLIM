# Autonomous-GNC-MAS

If you do not have ROS installed, follow the tutorial here for Ubuntu 20.04: http://wiki.ros.org/noetic/Installation/Ubuntu (Note- install ros-noetic-desktop-full, as this code currently requires the gazebo simulator to function). Follow all steps on that page and then continue with this document. 
If you are planning to use a different ROS version than noetic, you will have to do some work to get python3 working with ROS. It was originally runnning in ROS Melodic, so there's a decent chance. 

Installing this code - 
Create a new ROS workspace for this code:
...
mkdir ~/simulation_ws
cd ~/simulation_ws
mkdir src
catkin_make
...

Then, cd into ~/simulation_ws/src/ and clone this github repo. Also clone 3 turtlebot ROS packages:
...
cd ~/simulation_ws/src
git clone -b dev_inf_filter https://github.com/hfekrmandi/Self_Localization_Intelligent_Mapping_SLIM
git clone -b noetic-devel https://github.com/ROBOTIS-GIT/turtlebot3
git clone -b noetic-devel https://github.com/ROBOTIS-GIT/turtlebot3_msgs
git clone -b noetic-devel https://github.com/ROBOTIS-GIT/turtlebot3_simulations
git clone -b melodic-devel https://github.com/ros/robot_state_publisher
cd ..
catkin_make

sudo apt update
sudo apt install python3 pip-python3
pip3 install scipy numpy opencv-python opencv-contrib-python
...

The robot_state_publisher in noetic doesn't have a tf_prefix option, which is a death sentence for multiple identical agents. To fix this, I have included pulling the melodic version which doesn't have this issue. At this time there is an open PR so maybe it will be fixed later? https://github.com/ros/robot_state_publisher/pull/139

Replace (USER) with your username (you can see it with echo $USER).
...
echo "source /home/(USER)/simulation_ws/devel/setup.bash" >> ~/.bashrc
source ~/.bashrc
...

Your directories should look like:
~/simulation_ws
	- build
	- devel
		- setup.bash
	- src
		- Self_Localization_Intelligent_Mapping_SLIM
			- Autonomous-GNC-MAS
			- dse_msgs
			- dse_simulation
			- dse_simulation_gazebo
			- dse_simulation_python
			- dse_simulation_real
			- dse_turtlebot_descriptions
			- README.md
		- tureltbot3
		- turtlebot3_msgs
		- turtlebot3_simulations

I also highly recommend installing terminator. It allows you to easily run multiple terminals simultaneously.
With it open, you can use Ctrl-Shift-E to split a terminal horizontally, and Ctrl-Shift-O to split vertically.
...
sudo apt update
sudo apt install terminator
...

And for writing code, I recommend pycharm community. It can be installed through the ubuntu software center. Then, launch it from a terminal
...
pycharm-community
...
and open a project -> select the Self_Localization_Intelligent_Mapping_SLIM folder. Set up the interpreter to be your system's python3 executable, File -> Settings -> Project -> Python Interpereter -> (click on the gear on the right) -> Add... -> Existing Environment -> Add /usr/bin/python3. This will ensure that pycharm has your installed python libraries and all of the ROS libraries. 

To test this code: 
...
roslaunch dse_simulation demo_1_agent.launch
...
This will start a gazebo simulation with a single agent and 3 tags lined up in front of it. Each of the 3 tags is seen and estimated by the agent. The estimates are visualized as a sample of 50 vectors from the mean and covariance of the estimate(Will be improved later to a covariance ellipse).

Visualizing in RVIZ
- Run the launch file dse_sim_3_robots.launch
	- This file runs the gazebo base_link true/estimate publisher node. 
	- In a terminal, source ~/simulation_ws/devel/setup.bash
	- Run the command roslaunch turtlebot3_gazebo turtlebot3_gazebo_rviz.launch
	- On the left side, under Displays -> Global Options, change the frame from odom to base_link
		- base_link is the robot's reference frame, so everything will be displayed in reference to it
	- Then in the bottom left click Add -> By Topic -> /dse/vis/[estimates or gazebo_true] whatever you want to look at. 
	- You can also go into each PoseArray and change the color so the pose arrows are individually identifiable
		- It's recommended to change the true pose to green

Node descriptions - 
- aruco_pose_estimation_node in aruco_pose_estimation.py
	- Inputs: Camera images
	- Outputs: Poses and IDs of all observed tags
	- What it does: uses the aruco library to find the pose of each tag, and publishes that pose in the robot's coordinate frame
- information_filter_node in information_filter.py
	- Inputs: Poses and IDs of all observed tags
	- Outputs: Information filter prior and measurement (Y_01, y_01, delta_I and delta_i)
	- What it does: Computes a step of the information filter
	- Reference information filter here
- direct_estimator_node in direct_estimator.py
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
	- Ensure that you are using your python 3 interpreter (should be at /usr/bin/python3)
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
- src/dse_lib.py
	- Library of functions mainly for the information filter
- src/consensus_lib.py
	- Library of functions for the consensus

Folder structure for this package:
- Autonomous-GNC-MAS
	ROS Meta-package folder, nothing important
- dse_msgs
	custom ROS messages
- dse_simulation
	Most everything
	- documentation
		rosgraphs and other documents showing the code flow
	- launch
		files to run worlds or simulations
	- media
		custom materials and models used in simulations
	- rviz
		rviz configs, usually one per simulation that preselects topics of interest
	- src
		All of the python code files. Each ROS node is its own file here
    - test
        Files for unit testing
	- world
		Simulation world files. Defines what is in the environment
- dse_simulation_gazebo
	Nothing yet, the idea is this holds gazebo-specific files and simulations
- dse_simulation_python
	Nothing yet, the idea is this holds lower-level python only simulations
- dse_simulation_real
	Nothing yet, the idea is this holds code for running on a real turtlebot
- dse_turtlebot_descriptions
	files for custom turtlebots with Aruco tags mounted on top of them. Also contains edited turtlebot3 files for changing the camera. 
- README.md
