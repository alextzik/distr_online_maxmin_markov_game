"""
    Rollout policy planner
"""

__authors__ = "D. Knowles"
__date__ = "02 Apr 2022"

import os
import sys
import math
sys.path.append("..")

import numpy as np
import networkx as nx
from numpy import linalg
from lib.dynamics import Dynamics
import matplotlib.pyplot as plt
from scipy.integrate import cumtrapz, odeint
from copy import deepcopy

import lib.parameters as P
from lib.tools import wrap0to2pi
import cvxpy as cp


class GoToGoalv2():
    def __init__ (self, agents, id, G, debug=False):
        """
            Simple planner that will always tell the agent to go to goal.
        """
        self.id = id
        self.debug = debug
        self.agents = agents
        self.num_agents = len(self.agents)
        self.dynamics = Dynamics(agents)

        """
           The constructor determines the nonlinear formation control system of differential equations, given a graph topology G
        """
        self.G = G
        self.laplacian = nx.laplacian_matrix(G).toarray()

    def linear_formation_control_system(self, s, t):
        """
            It builds the dynamics of the nonlinear formation control system
        :param s: a 1D array of the agent positions listed as [x[self.agents[0], ..., y[self.agents[0],...]
        :param t: time
        :return: ds/dt
        """
        L = np.zeros((self.num_agents, self.num_agents))
        B = np.zeros((2 * self.num_agents, 1))
        for i in range(self.num_agents):
            for j in range(self.num_agents):
                if i != j:
                    L[i, j] = -self.laplacian[i, j]

            L[i, i] = -np.sum(L[i, :])

        for i in range(self.num_agents):
            for j in range(self.num_agents):
                B[i, 0] += -1 * (-self.laplacian[i, j]) * (
                            P.goal_states[0, self.agents[j]] - P.goal_states[0, self.agents[i]])
                B[i + self.num_agents, 0] += -1 * (-self.laplacian[i, j]) * (
                            P.goal_states[1, self.agents[j]] - P.goal_states[1, self.agents[i]])

        L_2D = np.zeros((2 * self.num_agents, 2 * self.num_agents))
        j = 0
        for i in range(self.num_agents):
            L_2D[i, 0: self.num_agents] = L[j, :]
            L_2D[self.num_agents + i, self.num_agents: 2 * self.num_agents] = L[j, :]
            j = j + 1

        return np.matmul(L_2D, s.reshape(-1, 1)).reshape(-1, ) + B.reshape(-1, )

    def run_linear_formation_control(self, s):
        """Determine goal positions that satisfy the distance requirements of the formation.
                Parameters
                ----------
                s : current state of robots - column of size (2 x num_robots) with x's first
                Returns
                -------
                goal_states: 2 x num_robots with the goal states
        """
        result = odeint(self.linear_formation_control_system, s, np.linspace(0, 10))
        goal_states = result[-1, :]
        goal_states = goal_states.reshape(2, self.num_agents)

        return goal_states


    def plan(self, state):
        """Chose new action plan.
        Parameters
        ----------
        state : current state of robots - _ x num_robots array
        Returns
        -------
        a : tuple of action tuples
            Tuple of length num_agents of actions that should be taken
            for each agent.
        """
        goal_states = self.run_linear_formation_control(
                np.hstack(state[:2, :]).reshape(-1, ))  # goal_states size (2 x num_robots)
        action = []
        for aa in range(self.num_agents):
            # Define and solve a convex optimization problem to compute next action
            x = cp.Variable((2,1))
            objective = cp.norm(goal_states[:, aa]-
                                            self.dynamics.f(state[0, aa], state[1, aa])-self.dynamics.B@x)
            constraints = [P.yaw_lb <= x[0], x[0] <= P.yaw_ub] +\
                          [P.acc_lb <= x[1], x[1] <= P.acc_ub]
            prob = cp.Problem(cp.Minimize(cp.square(objective)), constraints)
            try:
                prob.solve()
                action.append((x.value[0], x.value[1]))
            except cp.error.SolverError:
                action.append((np.random.uniform(P.yaw_lb, P.yaw_ub), np.random.uniform(P.acc_lb, P.acc_ub)))
            # if x.value is None:
            #     action.append((np.random.uniform(P.yaw_lb, P.yaw_ub), np.random.uniform(P.acc_lb, P.acc_ub)))
            # else:
            #   action.append((x.value[0], x.value[1]))

        return tuple(action)

    def reset(self):
        """Any reset necessary between timesteps
        """
        pass

