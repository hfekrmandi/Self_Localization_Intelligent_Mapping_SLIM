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


# Grab the agent with a specific ID
# Done by looping through the list of agents and checking the ID
def agent_with_id(agents, ID):
    for i in range(np.shape(agents)[0]):
        if agents[i].objectID == ID:
            agent = agents[i]
            return
    agent = []


# Ensures that every agent has the same state variables in the same order
def get_sorted_agent_states(SIM, objectIndex):

    # Build combined list of ids
    id_list = []
    for agent_index in range(np.shape(SIM.OBJECTS)[0]):
        if SIM.OBJECTS(agent_index).type == OMAS_objectType.agent:
            for i in range(np.shape(SIM.OBJECTS)[0]):
                if objectIndex[agent_index].objectID == SIM.OBJECTS(agent_index).objectID:
                    agents[objectIndex[agent_index].objectID] = objectIndex[agent_index]
                    id_list = [id_list, objectIndex[agent_index].memory_id_list]

    # Ensure that the list is sorted, so it is the same on sequential runs
    id_list = sort(unique(id_list))
    dim_state = agents[1].dim_state
    dim_obs = agents[1].dim_obs
    n_agents = numel(id_list)

    # Ensure all agents' state variables match the master list
    for agent_index in range(np.shape(agents)[0]):
        agent = agents[agent_index]

        # If the state variables don't match, add them in
        if not isequal(agent.memory_id_list, id_list):
            Y = 0.01 * np.eye(n_agents * dim_state)
            y = zeros(n_agents * dim_state, 1)
            I = zeros(n_agents * dim_state)
            i = zeros(n_agents * dim_state, 1)

            # Move the agents' values to the location specified in the master list
            for agent_index_1 in range(np.shape(agent.memory_id_list)[0]):
                for agent_index_2 in range(np.shape(agent.memory_id_list)[0]):

                    group_index_1 = find(id_list == agent.memory_id_list[agent_index_1])
                    group_index_2 = find(id_list == agent.memory_id_list[agent_index_2])

                    # Generate indices (to make the assignment setp shorter)
                    g_row_lo = dim_state * (group_index_1 - 1) + 1
                    g_row_hi = dim_state * group_index_1
                    g_col_lo = dim_state * (group_index_2 - 1) + 1
                    g_col_hi = dim_state * group_index_2
                    a_row_lo = dim_state * (agent_index_1 - 1) + 1
                    a_row_hi = dim_state * agent_index_1
                    a_col_lo = dim_state * (agent_index_2 - 1) + 1
                    a_col_hi = dim_state * agent_index_2

                    Y[g_row_lo:g_row_hi, g_col_lo:g_col_hi] = agent.memory_Y[a_row_lo:a_row_hi, a_col_lo:a_col_hi]
                    I[g_row_lo:g_row_hi, g_col_lo:g_col_hi] = agent.memory_I[a_row_lo:a_row_hi, a_col_lo:a_col_hi]

                y[g_row_lo:g_row_hi] = agent.memory_y[a_row_lo:a_row_hi]
                i[g_row_lo:g_row_hi] = agent.memory_i[a_row_lo:a_row_hi]

            agent.memory_id_list = id_list
            agent.memory_Y = Y
            agent.memory_y = y
            agent.memory_I = I
            agent.memory_i = i


# Update the momory_id_comm list in each agent to include each agent they
# can communicate with (based on a communication model).
def apply_comm_model_obs(agents):

    # Apply communication model and create list of agents each agent can communicate with
    # Current communication model is the same as the observation model
    agents_arr = []
    for agent = agents:
        agents_arr = [agents_arr, agent[1]]

    for agent = agents_arr:
        agent.memory_id_comm = []
        for i in range(np.shape(agent.memory_id_obs)[0]):
            if not isempty(agent_with_id(agents_arr, agent.memory_id_obs[i])):
                agent.memory_id_comm = [agent.memory_id_comm, agent.memory_id_obs[i]]


# Update the momory_id_comm list in each agent to include each agent they
# can communicate with (based on a communication model).
def apply_comm_model(SIM, agents, comm_list):

    # Apply communication model and create list of agents each agent can communicate with
    # Current communication model is the same as the observation model
    agents_arr = []
    for id in range(np.shape(agents)[0]):
        agent = agents[id]
        if SIM.OBJECTS(agent.objectID).type == 1:
            agents_arr = [agents_arr, agent]

    for index in range(np.shape(agents_arr)[0]):
        agent = agents_arr(index)
        agent_comm_list = cell2mat(comm_list[index])
        agent.memory_id_comm = agent_comm_list


# Update the momory_id_comm list in each agent to include each agent they
# can communicate with.
def apply_comm_model_distance(SIM, agents, threshold):

    agents_arr = []
    for id in range(np.shape(agents)[0]):
        agent = agents[id]
        if SIM.OBJECTS(agent.objectID).type == 1:
            agents_arr = [agents_arr, agent]

    for i in range(np.shape(agents_arr)[0]):
        agents_arr[i].memory_id_comm = []
        for j in range(np.shape(agents_arr)[0]):
            if i != j:
                detect = agents_arr[i].commRadius
                Xi = SIM.OBJECTS[agents_arr[i].objectID].X
                Xj = SIM.OBJECTS[agents_arr[j].objectID].X
                if norm(Xj[1: 3] - Xi[1:3]) <= detect:
                    agents_arr[i].memory_id_comm = [agents_arr[i].memory_id_comm, j]


# Update the momory_id_obs list in each agent to include each agent they
# can observe.
def apply_obs_model_distance(SIM, agents, threshold):

    agents_arr = []
    for id in range(np.shape(agents)[0]):
        agent = agents[id]
        if SIM.OBJECTS(agent.objectID).type == 1:
            agents_arr = [agents_arr, agent]

    for i in range(np.shape(agents_arr)[0]):
        agents_arr[i].memory_id_obs = []
        for j in range(np.shape(agents_arr)[0]):
            if i != j:
                detect = agents_arr[i].obsRadius
                Xi = SIM.OBJECTS[agents_arr[i].objectID].X
                Xj = SIM.OBJECTS[agents_arr[j].objectID].X
                if norm(Xj[1: 3] - Xi[1:3]) <= detect:
                    agents_arr[i].memory_id_obs = [agents_arr[i].memory_id_obs, j]


# Break agents up into groups based on communication graph
def break_agents_into_groups(SIM, agent_data):
    # Split into groups of agents that can communicate with eachother,
    # and puts those agents in a group. Continues this along the chain
    # until there are no more agents in this group, then finds the other
    # isolated groups.

    agent_groups = []
    num_groups = 0
    # While there are still agents to group up
    while numel(agent_data) > 0:

        # Start with the first remaining agent
        group = agent_data[1]
        id_obs = group(1).memory_id_comm
        new_group = 1
        agent_data[1] = []

        # while there are new agents in the group
        while new_group > 0:

            # Get a list of the newly-observed IDs
            len = numel(group)
            tmp = group[1, len - new_group + 1:len]
            id_obs_new = []
            for m in range(np.shape(tmp)[0]):
                id_obs_new = [id_obs_new, tmp(m).memory_id_comm]

            id_obs_new = sort(unique(id_obs_new))
            id_obs_new = setdiff(id_obs_new, id_obs)
            id_obs = sort([id_obs, id_obs_new])
            new_group = 0
            indices = []

            # Get the agents with ids matching the observed list
            for i in id_obs:
                for j in range(np.shape(agent_data)[0]):
                    if agent_data[j].objectID == i:
                        group = [group, agent_data[j]]
                        new_group = new_group + 1
                        indices = [indices, j]

            # Remove grouped agents from the general pool
            for i in sort(indices, 'descend'):
                agent_data[i] = []

        agent_groups[num_groups + 1] = group
        num_groups = num_groups + 1


# Create a graph from a group of agents
def create_graph(agents):
    adj = np.eye(numel(agents))

    id_to_index = []
    for i in range(np.shape(agents)[0]):
        id_to_index[i] = agents[i].objectID

    for i in range(np.shape(agents)[0]):
        for j in agents[i].memory_id_comm:
            adj[i, find(j == id_to_index)] = 1

    if size(adj, 1) == 1 & size(adj, 2) == 1:
        tmp = 0

    graph = generate_graph(adj)

    return graph, id_to_index



# Perform one consensus step
def consensus_group(agents, step, num_steps):
    # Perform consensus computations

    ## Initialize Graph
    # Generate graph
    # Create adjacency matrix
    # use generate_graph function
    [graph, id_to_index] = create_graph(agents)
    size_comp = networkComponents(graph.p)

    ## Compute and store consensus variables
    for i in range(np.shape(agents)[0]):

        # Grab variables from neighboring agents
        Y_local = []
        y_local = []
        idx_neighbors = agents[i].memory_id_comm

        for j in range(np.shape(idx_neighbors)[0]):
            agent = agent_with_id(agents, idx_neighbors[j])
            Y_local[:,:, j] = (agent.memory_Y)
            y_local[:,:, j] = (agent.memory_y)

        # Compute and apply CI weights
        [weights_ci, Y_prior, y_prior] = calc_ci_weights_ver3(Y_local, y_local, 'det')

        delta_I = zeros(size(agents[1].memory_I))
        delta_i = zeros(size(agents[1].memory_i))

        for j in range(np.shape(agents)[0]):
            p_jk = graph.p[i, j]

            delta_I = delta_I + p_jk * agents[j].memory_I
            delta_i = delta_i + p_jk * agents[j].memory_i

        ratio = step / num_steps
        Y = Y_prior + ratio * size_comp[i] * delta_I
        y = y_prior + ratio * size_comp[i] * delta_i

        consensus_data[i].Y = Y
        consensus_data[i].y = y
        consensus_data[i].Y_prior = Y_prior
        consensus_data[i].y_prior = y_prior
        consensus_data[i].delta_I = delta_I
        consensus_data[i].delta_i = delta_i

    return consensus_data


def consensus(agent_groups):
    num_steps = 20
    for group_num in range(np.shape(agent_groups)[0]):
        if numel(agent_groups[group_num]) == 1:
            agent_groups[group_num].memory_Y = agent_groups[group_num].memory_Y + agent_groups[group_num].memory_I
            agent_groups[group_num].memory_y = agent_groups[group_num].memory_y + agent_groups[group_num].memory_i
            agent_groups[group_num].memory_P = np.linalg.inv(agent_groups[group_num].memory_Y)
            agent_groups[group_num].memory_x = np.linalg.inv(agent_groups[group_num].memory_Y) *agent_groups[group_num].memory_y
        else:
            # Compute first consensus step
            step = 1
            for i in range(np.shape(agent_groups[group_num])[0]):
                consensus_data[step, group_num][i].Y_prior = agent_groups[group_num][i].memory_Y
                consensus_data[step, group_num][i].y_prior = agent_groups[group_num][i].memory_y
                consensus_data[step, group_num][i].delta_I = agent_groups[group_num][i].memory_I
                consensus_data[step, group_num][i].delta_i = agent_groups[group_num][i].memory_i

            # Compute the remaining consensus steps
            for step in range(2, num_steps):
                consensus_data[step, group_num] = consensus_group(agent_groups[group_num], step, num_steps)

            # After all agents' variables have been computed, store them
            for i in range(np.shape(consensus_data[step, group_num])[0]):
                agent_groups[group_num][i].memory_Y = consensus_data[step, group_num][i].Y_prior
                agent_groups[group_num][i].memory_y = consensus_data[step, group_num][i].y_prior
                agent_groups[group_num][i].memory_I = consensus_data[step, group_num][i].delta_I
                agent_groups[group_num][i].memory_i = consensus_data[step, group_num][i].delta_i

            # Store final consensus in each agent
            for i in range(np.shape(consensus_data[step, group_num])[0]):
                agent_groups[group_num][i].memory_Y = consensus_data[step, group_num][i].Y
                agent_groups[group_num][i].memory_y = consensus_data[step, group_num][i].y
                agent_groups[group_num][i].memory_P = np.linalg.inv(consensus_data[step, group_num][i].Y)
                agent_groups[group_num][i].memory_x = np.linalg.inv(consensus_data[step, group_num][i].Y) * consensus_data[step, group_num][i].y

    return agent_groups


# function [gt_estimation] = gt(agent_data)
#     y = y_1 + i_1 + i_2
# 
def position_from_id(agent, id):

    x = agent.memory_x
    index_id = find(id == agent.memory_id_list)
    index_agent = find(agent.objectID == agent.memory_id_list)
    dim_state = agent.dim_state

    # Generate indices (to make the assignment setp shorter)
    meas_low = dim_state * (index_id - 1) + 1
    meas_high = dim_state * index_id
    agent_low = dim_state * (index_agent - 1) + 1
    agent_high = dim_state * index_agent

    rel_pos = x[meas_low:meas_high] - x[agent_low:agent_high]
    rel_pos = rel_pos[1:2]

    return rel_pos

# [nComponents,sizes,members] = networkComponents(A)
#
# Daniel Larremore
# April 24, 2014
# larremor@hsph.harvard.edu
# http://danlarremore.com
# Comments and suggestions always welcome.
#
# INPUTS:
# A                     Matrix. This function takes as an input a
# network adjacency matrix A, for a network that is undirected. If you
# provide a network that is directed, this code is going to make it
# undirected before continuing. Since link weights will not affect
# component sizes, weighted and unweighted networks work equally well. You
# may provide a "full" or a "sparse" matrix.
#
# OUTPUTS:
# nComponents             INT - The number of components in the network.
# sizes                 vector<INT> - a vector of component sizes, sorted,
#   descending.
# members               cell<vector<INT>> a cell array of vectors, each
#   entry of which is a membership list for that component, sorted,
#   descending by component size.
#
# Example: (uncomment and copy and paste into MATLAB command window)
# # Generate a 1000 node network adjacency matrix, A
# A = floor(1.0015*rand(1000,1000)) A=A+A' A(A==2)=1 A(1:1001:end) = 0
# # Call networkComponents function
# [nComponents,sizes,members] = networkComponents(A)
# # get the size of the largest component
# sizeLC = sizes(1)
# # get a network adjacency matrix for ONLY the largest component
# LC = A(members[1],members[1])

def networkComponents(A):
    # Number of nodes
    N = size(A,1)
    # Remove diagonals
    A[1:N+1:end] = 0
    # make symmetric, just in case it isn't
    A=A+A'
    # Have we visited a particular node yet?
    isDiscovered = zeros(N,1)
    # Empty members cell
    members = []
    # check every node
    for n in range(N):
        if not isDiscovered[n]:
            # started a new group so add it to members
            members[end+1] = n
            # account for discovering n
            isDiscovered[n] = 1
            # set the ptr to 1
            ptr = 1
            while (ptr <= length(members[end])):
                # find neighbors
                nbrs = find(A[:, members[end](ptr)])
                # here are the neighbors that are undiscovered
                newNbrs = nbrs[isDiscovered[nbrs]==0]
                # we can now mark them as discovered
                isDiscovered[newNbrs] = 1
                # add them to member list
                members[end][end+1:end+length(newNbrs)] = newNbrs
                # increment ptr so we check the next member of this component
                ptr = ptr+1

    # number of components
    nComponents = length(members)
    for n in range(nComponents):
        # compute sizes of components
        group_n = members[n]
        for j in range(length(group_n)):
            size_group[group_n[j]] = numel(group_n)
        #     sizes(n) = length(members[n])

    return size_group, nComponents, members


def generate_graph(Adj):
    # This function accepts an adjecancy matrix where degree of each
    # node is equal to 1 + number of its neighbours. That is, all agents
    # are connected to themselves as well.

    #  number of nodes ( = |v|)
    nv = size(Adj, 2)

    # Assign the graph adjecancy matrix
    G.Adj = Adj

    # Calculate inclusive node degrees
    for i in range(nv)
        G.d[i] = sum(G.Adj[i, :]) + 1

    # Calculate weights for MHMC distributed averaging
    # This is slightly different from the formula used in the paper
    # http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.161.3893&rep=rep1&type=pdf
    for i in range(nv):
        for j in range(nv):
            if G.Adj[i, j] != 0:
                if i != j:
                    G.p[i, j] = min(1 / G.d[i], 1 / G.d[j])
            else:
                G.p[i, j] = 0

    for i in range(nv):
        try:
            G.p[i, i] = 1 - sum(G.p[i, :])
        except:
            tmp = 0

    return G

def calc_ci_weights_ver3(S1, local_inf_vec, method_):
    # Number of covariance matrices 
    nCovSamples = size(S1, 3)

    # Generate a random initialize weight and normalize it so 
    # it sums up to 1.
    x0 = rand(nCovSamples, 1)
    x0 = x0 ./ sum(x0)

    # Thos constraint ensures that the sun of the weights is 1
    Aeq = ones(size(x0))'
    beq = 1

    # Weights belong to the interval [0,1]
    lb = zeros(size(x0))'
    ub = ones(size(x0))'
    A = []
    b = []
    nonlcon = []

    if verLessThan('matlab', '8.4'):
        options = optimset('Algorithm', 'sqp')

        if strcmp(method_, 'tr'):
            x = fmincon(@cost_ci_tr, x0, A, b, Aeq, beq, lb, ub, nonlcon, options)
        elif strcmp(method_, 'det'):
            x = fmincon(@cost_ci_det, x0, A, b, Aeq, beq, lb, ub, nonlcon, options)

        else:
        options = optimoptions('fmincon', 'Display', 'none', 'Algorithm', 'sqp')

        if strcmp(method_, 'tr'):
            x = fmincon(@cost_ci_tr, x0, A, b, Aeq, beq, lb, ub, nonlcon, options)
        elif strcmp(method_, 'det'):
            x = fmincon(@cost_ci_det, x0, A, b, Aeq, beq, lb, ub, nonlcon, options)

    # CI weghts
    weights_ci = x

    # Normalize just in case
    if sum(weights_ci) > 1:
        weights_ci = weights_ci ./ sum(weights_ci)

    # Now that we have the weights, calculate w1*I1+...+wn*In
    inf_vect = special_dot_sum(weights_ci, local_inf_vec, 0)
    inf_mat = calc_inf_ci(x)

    return weights_ci, inf_mat, inf_vect


# Trace cost function as the objective function
def cost_ci_tr(x):
    information_matrix = zeros(size(S1[:, :, 1]))

    for i_tr in range(length(x)):
        information_matrix = information_matrix + x[i_tr, 1] * (S1[:, :, i_tr])

    # Make the information matrix symetric in case numerical errors during the summation calculation
    information_matrix = 0.5 * (information_matrix + information_matrix')

    cost_tr = trace(np.linalg.inv(information_matrix))
    return cost_tr


# Determinant cost function
def cost_ci_det(x):
    information_matrix = zeros(size(S1[:, :, 1]))

    for i_det in range(length(x)):
        information_matrix = information_matrix + x[i_det, 1] * (S1[:, :, i_det])

    # Make the information matrix symetric in case numerical errors during the summation calculation
    information_matrix = 0.5 * (information_matrix + information_matrix')

    cost_det = -log(det(information_matrix))

    # cost calculation near the singularity.
    if isinf(cost_det):
        cost_det = log(det(np.linalg.inv(information_matrix)))

    return cost_det


def calc_inf_ci(x):
    information_matrix = zeros(size(S1[:, :, 1]))

    for i_det in range(length(x)):
        information_matrix = information_matrix  +x[i_det, 1] * (S1[:, :, i_det])

    # Make the information matrix symetric in case numerical errors during the summation calculation
    information_matrix = 0.5 * (information_matrix + information_matrix')

    return information_matrix

