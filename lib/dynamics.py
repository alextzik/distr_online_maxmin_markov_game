"""
    Dynamics for the simulation.
    AA229: Sequential Decision Making final project code.
"""

__authors__ = "D. Knowles"
__date__ = "06 Feb 2022"

import numpy as np

import lib.parameters as P
from lib.tools import wrap0to2pi

class Dynamics():

    def __init__(self, agents):
        """
            Dynamics for each agent
        """
        self.num_agents = len(agents)
        self.agents = agents

        # Per-agent dynamics
        # self.f = lambda s1, s2: np.array([0.2*np.cos(s1)-0.1*s2, -0.4*s2]).reshape(-1,1)
        self.f = lambda s1, s2: np.array([s1, s2]).reshape(-1, 1)
        self.A = np.eye(2, 2)
        self.B = np.array([[1, 0], [-1, 2]])
        # self.B = np.array([[1, 0], [0, 1]])

    def update_states(self, agent_states, action_plan, verbose=False):
        """
            Updates agent states based on dynamics

        Parameters
        ----------
        agent_states : np.ndarray
            States of all agents of size (#states x #agents).
        action_plan : list
            A list of tuples for the action tuples for each agent.
            (yaw change, acceleration)
        verbose : bool
            If true, prints debug statements.
        """
        if verbose:
            print("update states:")
            print("actions:", action_plan[0])
            print("dynamics before state:\n", agent_states)
        before = agent_states.copy()
        for aa in range(self.num_agents):
            new_x = self.f(before[0, aa], before[1, aa]) + self.B@np.asarray(action_plan[aa]).reshape(-1,1)
            agent_states[:, aa] = [new_x[0,0], new_x[1,0]]
        if verbose:
            print("dynamics state change\n",agent_states - before)
            print("dynamics absolute state:\n",agent_states)

        return agent_states
