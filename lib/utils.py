import numpy as np
import lib.parameters as P
from lib.cvxapp.amap import AMAPEstimator
import copy

########################################################################
# USEFUL FUNCTIONS
########################################################################


def calc_reward(current_states, desired_states, agents, fully_connected = False):
    """
    :param  current_states: an numpy array of size 2 x num_robots containing the current robot positions
            desired_states: an numpy array of size 2 x num_robots containing one of the desired formations
            fully_connected: if False it calculates the reward based on the distances among edges from the true graph G for agents in agents
                             if True it calculates the reward based on all the distances among the agents in agents
    :return: a dictionary of dictionaries (one per agent) with pairs of agents as keys containing as value the corresponding inter-distance
            dists[aa] contains the inter-distances of all agents in the neighborhood of aa that are in agents
    """
    reward = 0.0

    if not fully_connected:
        for edge in list(P.edges):
            if edge[0] in agents and edge[1] in agents:
                current_dist = (
                                current_states[:, edge[0]] - current_states[:, edge[1]])
                goal_dist = (
                                desired_states[:, edge[0]] - desired_states[:, edge[1]])
                reward += -np.linalg.norm(current_dist - goal_dist)
    else:
        n = len(agents)
        for n1 in range(n):
            for n2 in range(n1+1, n):
                    current_dist = (
                        current_states[:, agents[n1]] - current_states[:, agents[n2]])
                    goal_dist = (
                        desired_states[:, agents[n1]] - desired_states[:, agents[n2]])
                    reward += -np.linalg.norm(current_dist - goal_dist)
    return reward


def distr_optimization(function_list):
    """
    Performs the distributed optimization for the min-max problem, given a list of functions that represent the neighborhood cost
    :param function_list: list of functions (CVX_Approximator instances)
            (one for each agent) that express the agent neighborhood costs - element i corresponds to P.nodes[i] (agent i)
           graph: the graph topology
    :return: action_plan: the optimal actions for each agent as a list of lists
    """
    num_agents = len(P.nodes)

    # proposed_action_and_values is a (num_agents x num_agents x 2 + num_agents, 1) array
    # that has each agent's action tuple value for all agents and the min-max problem's estimated
    # value by each agent: it first has all agents' proposed yaws for each agent (estimates by agent 1, then estimates by agent 2, etc),
    # then all agents' proposed accs for each agent
    # and finally each agent's min-max value estimate

    proposed_action_and_values = np.random.uniform(P.yaw_lb, P.yaw_ub, (num_agents*num_agents, 1))
    proposed_action_and_values = np.concatenate([proposed_action_and_values, np.random.uniform(P.acc_lb, P.acc_ub, (num_agents*num_agents, 1))], axis=0)
    proposed_action_and_values = np.concatenate([proposed_action_and_values, np.random.uniform(-5., 0, (num_agents, 1))], axis=0)

    cc = 1
    while cc < P.NUM_ITERATIONS_PER_STEP+1:
        # print(cc)
        # STEP 1: Update the agent estimates
        prev_proposed_action_and_values = copy.deepcopy(proposed_action_and_values)
        for i in range(num_agents+1): # for every decision variable dimension
            if i < num_agents:
                idcs_1 = [i +_*num_agents for _ in range(num_agents)]
                idcs_2 = [num_agents*num_agents + i + _*num_agents for _ in range(num_agents)]
                proposed_action_and_values[idcs_1, 0] = P.W @ prev_proposed_action_and_values[idcs_1, 0]
                proposed_action_and_values[idcs_2, 0] = P.W @ prev_proposed_action_and_values[idcs_2, 0]

            else:
                idcs = [num_agents*num_agents*2+_ for _ in range(num_agents)]
                proposed_action_and_values[idcs, 0] = P.W @ prev_proposed_action_and_values[idcs, 0]

        # STEP 2: Step toward minimizing own penalty function through a two-step adjustment
        j = 2*num_agents
        proposed_action_and_values[j * num_agents:, 0] = proposed_action_and_values[j * num_agents:, 0] - (1/cc)/num_agents

        for i in range(num_agents):
            idcs = [_ for _ in range(i*num_agents, (i+1)*num_agents)]
            idcs = idcs + [_ for _ in range(i*num_agents+num_agents*num_agents, (i+1)*num_agents+num_agents*num_agents)] + [num_agents*num_agents*2+i]
            proposed_action_and_values[idcs] = project(proposed_action_and_values[idcs, 0].reshape(-1,1), function_list[i], cc, i, num_agents)

        cc += 1
    # print(proposed_action_and_values.shape)
    # print(proposed_action_and_values)
    action_plan = [[0.0, 0.0] for _ in range(num_agents)]
    for i in range(num_agents):
        idcs = [i * num_agents+i, i * num_agents+i+num_agents*num_agents]
        action_plan[i][0] = proposed_action_and_values[idcs[0], 0] + np.random.normal(0.0, P.yaw_std_dev)
        action_plan[i][1] = proposed_action_and_values[idcs[1], 0] + np.random.normal(0.0, P.acc_std_dev)

    return action_plan

def project(variable, func, iter, id, num_total_agents):
    """
    Updates variable according to (10) from paper and computes the projection
    :param variable: the variable value
    :param func: function for which subgradient will be calculated, CVX_Approximator instant
    :param iter: iteration step
    :param id: agent id that the variable and local func belongs to
    :param num__total_agents: total number of agents
    :return: updated_variable
    """
    inter_variable = variable - (1/iter)*P.r[id]*calc_subgradient(func, variable, num_total_agents) # assume a_k=1/k
    updated_variable = np.zeros(inter_variable.shape)
    updated_variable[:num_total_agents, 0] = np.clip(inter_variable[:num_total_agents, 0], P.yaw_lb, P.yaw_ub)
    updated_variable[num_total_agents:2*num_total_agents, 0] = np.clip(inter_variable[num_total_agents:2*num_total_agents, 0], P.acc_lb, P.acc_ub)
    updated_variable[2*num_total_agents, 0] = np.clip(inter_variable[2*num_total_agents:2*num_total_agents+1, 0], 0, None)

    return updated_variable

def calc_subgradient(func, variable, num_total_agents):
    """
    Calculates the subgradient according to 11, given f_i and v_k^i
    :param func: f_i
    :param variable: v_k^i
    :param num_total_agents: total number of agents
    :return: subgradient_value
    """

    local_variable = transform_global_to_local(variable, func.agents, num_total_agents)
    activated = 0
    if func.amap.predict(local_variable.T) >= variable[-1, 0]:
        activated = 1
    subgradient_value = np.transpose(func.amap.subgradient(np.transpose(local_variable)))
    subgradient_value = subgradient_value[1:, ]
    subgradient_value = transform_local_to_global(subgradient_value, func.agents, num_total_agents)
    subgradient_value = np.concatenate([subgradient_value, np.array([-1]).reshape(-1, 1)], axis=0)*activated
    return subgradient_value


def transform_global_to_local(variable, agents, num_total_agents):
    """
    To be used in calc_subgradient: transforms the global variable copy of an agent by removing the agents not in its neighborhood
    :param variable: global variable including the yaws and accelerations of all robots
    :param agents: agents in the local neighborhood
    :param num_total_agents: total number of agents in system
    :return: local variable
    """
    local_variable = np.zeros((2*len(agents), variable.shape[1]))
    i = 0
    for aa in agents:
        local_variable[i*2, 0] = variable[aa, 0]
        local_variable[i*2+1, 0] = variable[aa+num_total_agents, 0]
        i += 1

    return local_variable


def transform_local_to_global(variable, agents, num_total_agents):
    """
    To be used in calc_subgradient: transforms the subgradient w.r.t the variables present in the neighborhood
    to the subgradient w.r.t. all variables for agent i.

    :param variable: local variable including the yaws and accelerations of the neighborhood's robots
    :param agents: agents in the local neighborhood
    :param num_total_agents: total number of agents in system
    :return: global variable
    """
    global_variable = np.zeros((2 * num_total_agents, variable.shape[1]))
    i = 0
    for aa in agents:
        global_variable[aa, 0] = variable[i*2, 0]
        global_variable[aa+num_total_agents, 0] = variable[i*2+1, 0]
        i += 1

    return global_variable


class CVX_Approximator():
    def __init__(self, agents):
        self.agents = agents
        self.amap = AMAPEstimator()
