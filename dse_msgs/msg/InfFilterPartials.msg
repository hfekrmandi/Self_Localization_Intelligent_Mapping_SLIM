########################################
# Messages
########################################
int32 sender_id # ID of the agent that sent this message
int32[] ids # Array of known IDs
std_msgs/Float64MultiArray inf_matrix_prior
std_msgs/Float64MultiArray inf_vector_prior
std_msgs/Float64MultiArray obs_matrix
std_msgs/Float64MultiArray obs_vector
