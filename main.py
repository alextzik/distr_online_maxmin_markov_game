"""
    Main of project: Runs decentralized MCTS for Max-Min or Max-Max centralized problems
"""

__authors__ = "A. Tzikas"
__date__ = "August 2022"

########################################################################
# NECESSARY PACKAGES
########################################################################
import copy
import os
import sys
import time
import pickle
from importlib import reload
import random
import time

import numpy as np
import cvxpy as cp
import multiprocessing

import lib.parameters as P
from lib.utils import calc_reward
from lib.tools import mkdir
from lib.pomcpow import POMCPOW
from lib.dynamics import Dynamics

from agent_plotter import plot_agents

if P.visualize:
    from lib.mapper import WorldMapper
from lib.rollout_planner import GoToGoalv2
import lib.utils as utils

import matplotlib.pyplot as plt

########################################################################
# SIMULATION PARAMETERS
########################################################################
RUNS = 1  # number of experiment runs
STEPS = 150 # number of simulation steps per experiment run
METHODS = ["dec_baseline_cvxpy"]
# METHODS = ["dec_baseline_only_POMDP", "dec_baseline_liujia", "dec_baseline_cvxpy", "dec_baseline_roll", "dec_mcts_maxmin"]
# methods to be independently deployed - choices: dec_mcts_maxmin, dec_mcts_maxmax, dec_baseline_roll, dec_baseline_liujia, dec_baseline_cvxpy

PLOT_COMP_TIME = True

########################################################################
# MAIN FUNCTIONS
########################################################################


def simulation_dec_mcts(planners, which):
    """
        Simulator for the decentralized system

        Parameters
        -----------------------
        planners: a list of MCTS objects, each of which corresponds to one of the agents and involves only the agents in its neighborhood
        which: string input that determines whether a max-min or a min-max problem will be solved
    """
    graph = P.G
    # initialize agent states and state history
    agents_states = P.agent_states
    num_agents = agents_states.shape[1]

    # initialize dynamics for the whole system
    dynamics = Dynamics(graph.nodes)

    # initialize visualization
    if P.visualize:
        world = WorldMapper()

    cc = 0

    # initialize reward keeping for each agent's neighborhood
    rewards_per_neighb = [[] for aa in range(num_agents)]

    while cc < STEPS:
        print("step",cc)
        current_states = agents_states.copy()

        plot_agents(current_states, cc, "proposed")

        neighborhoods_states = []  # list of numpy arrays, element i represents the current state of the agents in the
        # neighborhood of agent i in the order list(graph.adj[aa]) + [aa]
        for aa in range(num_agents):
            neighborhoods_states.append(agents_states[:, list(graph.adj[aa]) + [aa]])

        # Deployment of the planners
        action_plans = []
        Q_values = []  # only to be used in the max-max case: stores the Q value of the proposed action tuples by each agent's MCTS
        for aa in range(num_agents):
            neighborhood = list(graph.adj[aa]) + [aa]

            reward = calc_reward(agents_states[:2, :], P.goal_states[:2, :], neighborhood, True)
            rewards_per_neighb[aa].append(reward)
            planners[aa].cum_cost = -sum(rewards_per_neighb[aa])

            neighborhood_states = tuple([neighborhoods_states[aa][:, _] for _ in range(len(neighborhood))])
            action_plan, Q = planners[aa].plan(neighborhood_states)  # the planner of agent [aa] returns an action plan
            # that is a list of action tuples for the agents in
            # the neighborhood of [aa] with the order list(graph.adj[aa]) + [aa]
            # Q is the Q-value of this action tuple.
            # The planner aims at maximizing the reward

            # TO DO: If we implement an actual POMDP we will need to have the argument of plan be a belief of some form

            Q_values.append(Q)
            action_plans.append(action_plan)

        # Determine actual action for each agent
        action_plan = [[np.random.normal(0.0, P.yaw_std_dev), np.random.normal(0.0, P.acc_std_dev)] for _ in
                       range(num_agents)]

        if which == 'max-max':
            for aa in range(num_agents):
                neighbors = list(graph.adj[aa]) + [aa]
                Qs_neighborhood = [-Q_values[i] for i in neighbors]
                max_neighbor = neighbors[Qs_neighborhood.index(max(Qs_neighborhood))]
                max_neighbor_neighborhood = list(graph.adj[max_neighbor]) + [max_neighbor]
                idx = max_neighbor_neighborhood.index(aa)
                action_plan[aa][0] += action_plans[max_neighbor][idx][0]
                action_plan[aa][1] += action_plans[max_neighbor][idx][1]

        elif which == 'max-min':
            ##########################################
            # FUNCTION THAT PERFORMS THE DISTRIBUTED ACTION SELECTION
            action_plan = utils.distr_optimization([planners[aa].cvx_approximator for aa in range(num_agents)])
            ##########################################

        for aa in range(num_agents):
            planners[aa].reset()

        agents_states = dynamics.update_states(current_states.copy(), action_plan)

        if P.visualize:
            world.update(agents_states, agents_states)

        cc += 1

    return rewards_per_neighb


def simulation_baseline_only_POMDP(planners):
    """
        Simulator for the decentralized system

        Parameters
        -----------------------
        planners: a list of MCTS objects, each of which corresponds to one of the agents and involves only the agents in its neighborhood
        which: string input that determines whether a max-min or a min-max problem will be solved
    """
    graph = P.G
    # initialize agent states and state history
    agents_states = P.agent_states
    num_agents = agents_states.shape[1]

    # initialize dynamics for the whole system
    dynamics = Dynamics(graph.nodes)

    # initialize visualization
    if P.visualize:
        world = WorldMapper()

    cc = 0

    # initialize reward keeping for each agent's neighborhood
    rewards_per_neighb = [[] for aa in range(num_agents)]

    while cc < STEPS:
        print("step",cc)
        current_states = agents_states.copy()

        plot_agents(current_states, cc, "dec_baseline_only_POMDP")

        neighborhoods_states = []  # list of numpy arrays, element i represents the current state of the agents in the
        # neighborhood of agent i in the order list(graph.adj[aa]) + [aa]
        for aa in range(num_agents):
            neighborhoods_states.append(agents_states[:, list(graph.adj[aa]) + [aa]])

        # Deployment of the planners
        action_plans = []
        Q_values = []  # only to be used in the max-max case: stores the Q value of the proposed action tuples by each agent's MCTS
        for aa in range(num_agents):
            neighborhood = list(graph.adj[aa]) + [aa]

            reward = calc_reward(agents_states[:2, :], P.goal_states[:2, :], neighborhood, True)
            rewards_per_neighb[aa].append(reward)
            planners[aa].cum_cost = -sum(rewards_per_neighb[aa])

            neighborhood_states = tuple([neighborhoods_states[aa][:, _] for _ in range(len(neighborhood))])
            action_plan, Q = planners[aa].plan(neighborhood_states)  # the planner of agent [aa] returns an action plan
            # that is a list of action tuples for the agents in
            # the neighborhood of [aa] with the order list(graph.adj[aa]) + [aa]
            # Q is the Q-value of this action tuple.
            # The planner aims at maximizing the reward

            # TO DO: If we implement an actual POMDP we will need to have the argument of plan be a belief of some form

            Q_values.append(Q)
            action_plans.append(action_plan)

        # Determine actual action for each agent
        action_plan = [[np.random.normal(0.0, P.yaw_std_dev), np.random.normal(0.0, P.acc_std_dev)] for _ in
                       range(num_agents)]

        ##########################################
        # FUNCTION THAT PERFORMS THE DISTRIBUTED ACTION SELECTION
        for aa in range(num_agents):
            action_plan[aa][0] = action_plans[aa][-1][0]
            action_plan[aa][0] = action_plans[aa][-1][1]

        ##########################################

        for aa in range(num_agents):
            planners[aa].reset()

        agents_states = dynamics.update_states(current_states.copy(), action_plan)

        if P.visualize:
            world.update(agents_states, agents_states)

        cc += 1

    return rewards_per_neighb


def simulation_dec_baseline_roll():
    """
        Simulator for the decentralized baseline formation control method which
        uses exclusively the rollout planner.
        It deploys a rollout_planner for the entire system at every timestep.

        Parameters
        -----------------------
    """
    graph = P.G
    # initialize agent states and state history
    agents_states = P.agent_states

    num_agents = agents_states.shape[1]

    # initialize dynamics
    dynamics = Dynamics(graph.nodes)

    # initialize rollout planner
    planner = GoToGoalv2(P.nodes, P.main_rover_id, graph)

    # initialize visualization
    if P.visualize:
        world = WorldMapper()

    cc = 0

    # initialize reward keeping for each agent's neighborhood
    rewards_per_neighb = [[] for aa in range(num_agents)]

    while cc < STEPS:
        print("step",cc)
        current_states = agents_states.copy()

        plot_agents(current_states, cc, "baseline_newrollout")

        planned_action = planner.plan(current_states)
        action = [[0., 0.] for aa in range(num_agents)]

        for aa in range(num_agents):
            action[aa][0] = planned_action[aa][0] + np.random.normal(0.0, P.yaw_std_dev)
            action[aa][1] = planned_action[aa][1] + np.random.normal(0.0, P.acc_std_dev)

        agents_states = dynamics.update_states(current_states, action, False)

        for aa in range(num_agents):
            neighborhood = list(graph.adj[aa]) + [aa]

            reward = calc_reward(agents_states[:2, :], P.goal_states[:2, :], neighborhood, fully_connected=True)

            rewards_per_neighb[aa].append(reward)

        if P.visualize:
            world.update(agents_states, agents_states)

        planner.reset()

        cc += 1

    return rewards_per_neighb


def simulation_dec_baseline_liujia():
    """
        Simulator for the decentralized baseline formation control method, based
        on Liu-Jia paper "Robust formation control of discrete-time multi-agent
        systems by iterative learning approach".

        Parameters
        -----------------------
    """
    graph = P.G
    laplacian = P.laplacian

    # initialize agent states and state history
    agents_states = P.agent_states

    num_agents = agents_states.shape[1]

    # initialize dynamics
    dynamics = Dynamics(graph.nodes)

    # Find Gamma
    Gamma = cp.Variable((2,2))
    E = np.hstack((np.ones((num_agents-1, 1)), -np.eye(num_agents-1, num_agents-1)))
    G = np.vstack((np.zeros((1, num_agents-1)), -np.eye(num_agents-1, num_agents-1)))

    constraints = [cp.norm(np.eye(2*(num_agents-1), 2*(num_agents-1)) - cp.kron(E@laplacian@G, dynamics.B@Gamma), 'inf') <= 0.9]
    objective = cp.Minimize(0)
    prob = cp.Problem(objective, constraints)
    prob.solve()
    Gamma = np.asarray(Gamma.value)
    print(Gamma)

    # initialize visualization
    if P.visualize:
        world = WorldMapper()

    cc = 0

    # initialize reward keeping for each agent's neighborhood
    rewards_per_neighb = [[] for aa in range(num_agents)]

    # initialize actions
    planned_action = [[0.0, 0.0] for _ in range(num_agents)]

    while cc < STEPS:
        print("step",cc)
        current_states = agents_states.copy()

        plot_agents(current_states, cc, "baseline_liujia")

        # HERE: Apply eq. (14)
        planned_action_copy = copy.deepcopy(planned_action)
        for aa in range(num_agents):
            prev_action_aa = np.asarray(planned_action_copy[aa]).reshape(2,1)
            term = np.zeros((2,1))
            for jj in range(num_agents):
                if jj != aa:
                    prev_action_jj = np.asarray(planned_action_copy[jj]).reshape(2, 1)
                    term += -laplacian[aa, jj] *\
                            ((dynamics.f(current_states[0, jj], current_states[1, jj])+dynamics.B@prev_action_jj - P.goal_states[:, jj].reshape(-1,1)) -
                            (dynamics.f(current_states[0, aa], current_states[1, aa])+dynamics.B@prev_action_aa - P.goal_states[:, aa].reshape(-1,1)))
            # print(term)
            next_action = prev_action_aa + Gamma@term

            planned_action[aa] = [np.clip(next_action[0, 0], P.yaw_lb, P.yaw_ub), np.clip(next_action[1,0], P.acc_lb, P.acc_ub)]


        for aa in range(num_agents):
            planned_action[aa][0] = planned_action[aa][0] + np.random.normal(0.0, P.yaw_std_dev)
            planned_action[aa][1] = planned_action[aa][1] + np.random.normal(0.0, P.acc_std_dev)

        agents_states = dynamics.update_states(current_states, planned_action, False)

        for aa in range(num_agents):
            neighborhood = list(graph.adj[aa]) + [aa]

            reward = calc_reward(agents_states[:2, :], P.goal_states[:2, :], neighborhood, fully_connected=True)

            rewards_per_neighb[aa].append(reward)

        if P.visualize:
            world.update(agents_states, agents_states)

        cc += 1

    return rewards_per_neighb


def simulation_dec_baseline_cvxpy():
    """
        Simulator for the centralized formation control, based on solving the min max problem explicitly.

        Parameters
        -----------------------
    """
    graph = P.G
    laplacian = P.laplacian

    # initialize agent states and state history
    agents_states = P.agent_states

    num_agents = agents_states.shape[1]

    neighborhoods = []
    for aa in range(num_agents):
        neighborhood = list(graph.adj[aa])+[aa]
        neighborhoods.append(neighborhood)

    # initialize dynamics
    dynamics = Dynamics(graph.nodes)

    # Find U*
    x = {}
    u = {}
    for aa in range(num_agents):
        x[aa] = cp.Variable((2, STEPS + 1))
        u[aa] = cp.Variable((2, STEPS))

    obj = cp.max(cp.vstack([cp.sum([cp.sum([cp.norm(x[neighborhoods[aa][ii]][:, t] - x[neighborhoods[aa][jj]][:, t]
                                                    - (P.goal_states[:, neighborhoods[aa][ii]]-P.goal_states[:, neighborhoods[aa][jj]]), 2)
                                            for ii in range(len(neighborhoods[aa])) for jj in range(ii+1, len(neighborhoods[aa])) ])
                               for t in range(STEPS)]) for aa in range(num_agents)]))


    constraints = []
    for aa in range(num_agents):
        constraints += [x[aa][:, t+1] == dynamics.A@x[aa][:, t] + dynamics.B@u[aa][:, t] for t in range(STEPS)]
        constraints += [x[aa][0, 0] == P.agent_states[0, aa], x[aa][1, 0] == P.agent_states[1, aa]]
        constraints += [u[aa][0, t] <= P.yaw_ub for t in range(STEPS)]
        constraints += [u[aa][0, t] >= P.yaw_lb for t in range(STEPS)]
        constraints += [u[aa][1, t] <= P.acc_ub for t in range(STEPS)]
        constraints += [u[aa][1, t] >= P.acc_lb for t in range(STEPS)]

    prob = cp.Problem(cp.Minimize(obj), constraints)
    prob.solve()

    # initialize visualization
    if P.visualize:
        world = WorldMapper()

    cc = 0

    # initialize reward keeping for each agent's neighborhood
    rewards_per_neighb = [[] for aa in range(num_agents)]

    # initialize actions
    planned_action = [[0.0, 0.0] for _ in range(num_agents)]

    while cc < STEPS:
        print("step",cc)
        current_states = agents_states.copy()

        plot_agents(current_states, cc, "baseline_cvxpy")

        for aa in range(num_agents):
            planned_action[aa] = [u[aa].value[0, cc], u[aa].value[1, cc]]
        for aa in range(num_agents):
            planned_action[aa][0] = planned_action[aa][0] + np.random.normal(0.0, P.yaw_std_dev)
            planned_action[aa][1] = planned_action[aa][1] + np.random.normal(0.0, P.acc_std_dev)

        agents_states = dynamics.update_states(current_states, planned_action, False)

        for aa in range(num_agents):
            neighborhood = list(graph.adj[aa]) + [aa]

            reward = calc_reward(agents_states[:2, :], P.goal_states[:2, :], neighborhood, fully_connected=True)

            rewards_per_neighb[aa].append(reward)

        if P.visualize:
            world.update(agents_states, agents_states)

        cc += 1

    return rewards_per_neighb


def main(method):
    """
        Main function of the code: runs the simulation of each method RUNS times
    """
    file_dir = os.path.dirname(os.path.realpath(__file__))
    log_dir = os.path.join(file_dir, "log")
    mkdir(log_dir)

    timestamp = time.strftime("%Y%m%d%H%M%S")
    data_log_dir = os.path.join(log_dir, timestamp)
    mkdir(data_log_dir)

    num_agents = P.agent_states.shape[1]
    graph = P.G

    # Create result gathering structure
    data = {}
    data[method] = {}
    data[method]["rewards_per_neighb"] = [[0 for i in range(STEPS)] for aa in range(num_agents)]
    data[method]["log_path"] = os.path.join(data_log_dir, method)
    mkdir(data[method]["log_path"])

    for i in range(RUNS):
        print(f"Run {i}")

        # specify random seeds
        seed = random.randrange(2 ** 32 - 1)
        np.random.seed(seed)

        if method == "dec_mcts_maxmin":
            planners = []
            for aa in range(num_agents):
                planner = POMCPOW(Dynamics(list(graph.adj[aa]) + [aa]), list(graph.adj[aa]) + [aa], True, aa, False)
                planners.append(planner)
            rewards_per_neighb = simulation_dec_mcts(planners, 'max-min')

        elif method == "dec_mcts_maxmax":
            planners = []
            for aa in range(num_agents):
                planner = POMCPOW(Dynamics(list(graph.adj[aa]) + [aa]), list(graph.adj[aa]) + [aa], True, aa, False)
                planners.append(planner)
            rewards_per_neighb = simulation_dec_mcts(planners, 'max-max')

        elif method == "dec_baseline_roll":
            rewards_per_neighb = simulation_dec_baseline_roll()

        elif method == "dec_baseline_liujia":
            rewards_per_neighb = simulation_dec_baseline_liujia()

        elif method == "dec_baseline_cvxpy":
            rewards_per_neighb = simulation_dec_baseline_cvxpy()

        elif method == "dec_baseline_only_POMDP":
            planners = []
            for aa in range(num_agents):
                planner = POMCPOW(Dynamics(list(graph.adj[aa]) + [aa]), list(graph.adj[aa]) + [aa], True, aa, False)
                planners.append(planner)
            rewards_per_neighb = simulation_baseline_only_POMDP(planners)

        # save results to file
        np.savetxt(os.path.join(data[method]["log_path"], "rewards_per_neighb" + str(i)) + ".csv",
                   np.array(rewards_per_neighb), delimiter=",")

        # combine results for runs so far
        # divide if the last run
        for aa in range(num_agents):
            data[method]["rewards_per_neighb"][aa] = [a + b for a, b in zip(data[method]["rewards_per_neighb"][aa],
                                                                            rewards_per_neighb[aa])]
            # divide if the last run
            if i == RUNS - 1:
                data[method]["rewards_per_neighb"][aa] = [a / RUNS for a in data[method]["rewards_per_neighb"][aa]]


    np.savetxt(os.path.join(data[method]["log_path"], "avg_rewards_per_neighb.csv"),
                   np.array(data[method]["rewards_per_neighb"]), delimiter=",")


if __name__ == "__main__":
    pool = multiprocessing.Pool()
    pool.map(main, iter(METHODS))
    pool.close()
    pool.join()
