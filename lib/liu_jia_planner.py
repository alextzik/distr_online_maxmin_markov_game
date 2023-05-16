"""
    This file contains the implementation of the formation planner proposed in the paper
    "Robust formation control of discrete-time multi-agent systems by iterative learning approach"
"""

__authors__ = "A. Tzikas"

# imports
# standard library imports
from copy import deepcopy
import random
import time
from operator import itemgetter

# related third party imports
import networkx as nx
import numpy as np

# local library specific imports
import lib.parameters as P


class planner_liu_jia():

    def __init__(self, B_matrix, desired_dists_to_virt, T):
        """
        :param B_matrix: contains a numpy 2d-array of the B matrix in the dynamics of the agents (constant B assumed)
        :param desired_dists_to_virt: contains the desired distance to virtual leader of each agent
                              numpy array (num_agents, num_state_dim, num_timesteps)
                              if None: assume constant derived from goal states
        :param T: number of timesteps to run for
        """
        self.num_agents = P.goal_states.shape[1]
        self.state_dims = P.goal_states.shape[0]
        self.B = B_matrix

        self.desired_dists_to_virt = np.array((self.num_agents, self.state_dims, T))
        if desired_dists_to_virt is None:
            for _ in range(T):
                self.desired_dists_to_virt[:, :, _] = P.goal_states.T
        else:
            self.desired_dists_to_virt = desired_dists_to_virt

        self.T = T

        self.last_control_input = np.zeros((2, self.num_agents))

        self.Gammas = [np.zeros((2,1)) for _ in range(self.num_agents)]

