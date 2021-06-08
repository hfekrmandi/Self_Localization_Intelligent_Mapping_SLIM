# Code files
gazebo_sim_visualize_pose_world.py
In
 - /gazebo/model_states
Out
 - TFs for all gazebo objects in the world coordinate system
Takes gazebo links and generates a TF for each one. Primarily of interest are the base frame of each agent, and each vision target. 

centralized_estimator_10hz_2.py
In
 - /tb3_*/dse/inf/partial
Out
 - /tb3_*/dse/inf/results
Computes the information filter results as agent 0's prior and the sum of all agents' measurements. 

aruco_pose_estimation
In
 - /tb3_X/camera/rgb/image_raw
Out
 - /tb3_X/dse/pose_markers
Uses the aruco code to estimate the pose of markers detected in camera images

information_filter
In
 - /tb3_X/cmd_vel
 - /tb3_X/dse/inf/results
 - /tb3_X/dse/pose_markers
Out
 - /tb3_X/dse/inf/partial
Computes the information filter prior and measurement

rqt_plot_output.py
In
 - /tb3_X/dse/pose_markers
 - /tb3_X/dse/pose_true
 - /tb3_X/dse/inf/results
Out
 - /tb3_X/dse/plt/measurement
 - /tb3_X/dse/plt/true/robot
 - /tb3_X/dse/plt/true/tag
 - /tb3_X/dse/plt/estimates/robot
 - /tb3_X/dse/plt/estimates/tag
Converts inputs to pose and publishes them

gazebo_sim_visualize_pose_base.py
In
 - /tb3_X/dse/pose_markers
 - /tb3_X/dse/python_pose_true
 - /tb3_X/dse/inf/results
Out
 - /tb3_X/dse/vis/measurement
 - /tb3_X/dse/vis/python_true
 - /tb3_X/dse/vis/estimates
Converts inputs to pose_array and publishes them

world_tf_broadcaster.py
In
 - /gazebo/link_states
 - /gazebo/model_states
Out
 - /tb3_X/dse/vis/gazebo_true
 - TF transform for each robot's base frame to the world frame

circle_controller.py - 
In
 - 
Out
 - 

