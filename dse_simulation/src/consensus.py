import roslib
import sys
import rospy
import numpy as np
import datetime
import time
from geometry_msgs.msg import Pose
from dse_msgs.msg import PoseMarkers
from std_msgs.msg import Float64MultiArray
from std_msgs.msg import MultiArrayLayout
from std_msgs.msg import MultiArrayDimension
from dse_msgs.msg import InfFilterPartials
from dse_msgs.msg import InfFilterResults
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt

import dse_constants
import dse_lib
import consensus_lib

#  5. ////////////// UPDATE Communication matrix (@t=k) ///////////////
#     if step < 20 || step > 40
#         comm_list = [[2, 3] [1, 3] [1, 2]]
#     else
#         comm_list = [[3] [3] [1, 2]]
#     end
#     comm_list = [[] [] []]
objectIndex = apply_obs_model_distance(META, objectIndex)
objectIndex = apply_comm_model_distance(META, objectIndex)


# 6. //////// UPDATE SIMULATION/ENVIRONMENTAL META DATA (@t=k) ////////
[META, metaEVENTS] = UpdateSimulationMeta(META, objectIndex)  # Update META snapshot with equivilant objectIndex.state
# LOG THE META EVENTS
if ~isempty(metaEVENTS)  # If META events occurred this timestep
    EVENTS = vertcat(EVENTS, metaEVENTS)  # Append to global EVENTS
end
# /////////////////////////////////////////////////////////////////////


# 7. ////////// COMPUTE INFORMATION FILTER ESTIMATION (@t=k) //////////
if META.threadPool ~= 0
objectSnapshot = objectIndex  # Make a temporary record of the object set
parfor(ID1=1:META.totalObjects)
# MOVE THROUGH OBJECT INDEX AND UPDATE EACH AGENT
[detection[ID1], objectIndex[ID1], objectEVENTS] = UpdateInformationFilter(META, objectSnapshot, objectIndex
[ID1])
# LOG THE OBJECT EVENTS
if ~isempty(objectEVENTS)
    EVENTS = vertcat(EVENTS, objectEVENTS)
end
end
else
for ID1 = 1:META.totalObjects
# MOVE THROUGH OBJECT INDEX AND UPDATE EACH AGENT
[detection[ID1], objectIndex[ID1], objectEVENTS] = UpdateInformationFilter(META, objectIndex, objectIndex
[ID1])  # Update objectIndex snapshot with new META data
# LOG THE OBJECT EVENTS
if ~isempty(objectEVENTS)  # If objectEVENTS occur in this timestep
    EVENTS = vertcat(EVENTS, objectEVENTS)  # Append to global EVENTS
end
end
end
# /////////////////////////////////////////////////////////////////////


# 8. ///////////////// COMPUTE CONSENSUS STEPS (@t=k) /////////////////
agent_data = get_sorted_agent_states(META, objectIndex)
agent_groups = break_agents_into_groups(META, agent_data)
consensus(agent_groups)
# /////////////////////////////////////////////////////////////////////


# 9. ///////////////// RECORD ESTIMATED STATE (@t=k) /////////////////////
for index = 1:numel(agent_data)
# Collect the META.OBJECTS.state data (as the simulations understanding of the global states)
# and save to the output DATA.globalTrajectories set, must be done synchronously).
agent = agent_data
[index]
X_agent = agent.memory_x
P_agent = agent.memory_P
IDs = agent.memory_id_list
Obs = agent.memory_id_obs
Comm = agent.memory_id_comm
DATA.ids(index, 1:numel(IDs), META.TIME.currentStep) = IDs
'
DATA.Observations(index, 1:numel(Obs), META.TIME.currentStep) = Obs
'
DATA.Connections(index, 1:numel(Comm), META.TIME.currentStep) = Comm
'
for ID = IDs
P = agent.cov_from_id(P_agent, IDs, agent.objectID)
#             P_zero = agent.cov_from_id(P_agent, IDs, agent.objectID)
#             P_estimate = agent.cov_from_id(P_agent, IDs, ID)
#             P_relative = P_estimate - P_zero
DATA.estimate_covariance(index, ID,:,:, META.TIME.currentStep) = P

X_zero = agent.state_from_id(X_agent, IDs, agent.objectID)
X_estimate = agent.state_from_id(X_agent, IDs, ID)
X_relative = X_estimate - X_zero
DATA.estimates_rel(index, ID,:, META.TIME.currentStep) = X_relative

X_gt_agent = META.OBJECTS(agent.objectID).X
X_gt_object = META.OBJECTS(ID).X
X_gt_relative = X_gt_object - X_gt_agent
DATA.estimate_errors(index, ID,:, META.TIME.currentStep) = X_relative - X_gt_relative
end
end
# /////////////////////////////////////////////////////////////////////


# 10. //////// UPDATE AGENT ESTIMATE FROM CONSENSUS DATA (@t=k) ////////
if META.threadPool ~= 0
objectSnapshot = objectIndex  # Make a temporary record of the object set
parfor(ID1=1:META.totalObjects)
# MOVE THROUGH OBJECT INDEX AND UPDATE EACH AGENT
[objectIndex[ID1], objectEVENTS] = UpdateObject(META, objectSnapshot, objectIndex
[ID1])
# LOG THE OBJECT EVENTS
if ~isempty(objectEVENTS)
    EVENTS = vertcat(EVENTS, objectEVENTS)
end
end
else
for ID1 = 1:META.totalObjects
# MOVE THROUGH OBJECT INDEX AND UPDATE EACH AGENT
[objectIndex[ID1], objectEVENTS] = UpdateObject(META, objectIndex, objectIndex
[ID1])  # Update objectIndex snapshot with new META data
# LOG THE OBJECT EVENTS
if ~isempty(objectEVENTS)  # If objectEVENTS occur in this timestep
    EVENTS = vertcat(EVENTS, objectEVENTS)  # Append to global EVENTS
end
end
end
# /////////////////////////////////////////////////////////////////////
