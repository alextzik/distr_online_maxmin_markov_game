"""
This file contains the basic POMCPOW implementation. POMCPOW is implemented as a class and uses a dictionary notion. Based on Sunberg et al.,
the class contains the following:
    - function plan(b): applies the simulate() function n times (part of class)
        - function simulate(s,h,d): builds the POMCPOW tree starting with root node state s, history h and maximum depth d and evaluates Q (part of class)
            - function actionProgWiden(h): performs widening in the action space - also takes care of initializing keys in dictionaries (part of class)
                - function nextAction(h): calculates a new action based on the history h (not explicitly different function)
            - function gen_model(s,a): from state s takes action a, finds next state s' and returns next observation o, next state s' and reward r
            - function gen_observ(s): takes in a state and outputs an observation (just a (different) state)
            - function gen_reward(s, a, s'): takes in current state s, next state s', action a and outputs reward.
            - function rollout(s,h,d): returns a reward by performing a rollout policy (part of this class)
"""

__authors__ = "D. Knowles, A. Tzikas"
__date__ = "06 Feb 2022"

################################################################################################
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
from lib.utils import calc_reward, CVX_Approximator
from lib.cvxapp.amap import AMAPEstimator
from lib.rollout_planner import GoToGoalv2


################################################################################################


class POMCPOW():

    def __init__(self, dynamics, agents, is_mdp, id=None, debug=False):

        self.id = id

        self.num_agents = len(agents)  # Number of agents
        self.agents = agents

        self.debug = debug

        self.is_mdp = is_mdp

        # Observation widening parameters
        self.ko = 0.0 # If we actually have an MDP this must be 0 - one observation possible from each state
        self.ao = 0.5

        # Action widening parameters
        self.ka = 2
        self.aa = 0.5

        self.c = 0.1 # c for UCB

        # Action space parameters (for each agent)
        self.yaw_lb = P.yaw_lb
        self.yaw_ub = P.yaw_ub
        self.acc_lb = P.acc_lb
        self.acc_ub = P.acc_ub

        # Dictionaries
        self.C = {} # Observation/action widening dictionary: maps history (as tuple) to list of observations/action
        self.N = {} # Count dictionary: maps history (as tuple) to integer
        self.Q = {} # Q dictionary: maps history (as tuple) to q value
        self.B = {} # Set of states in each history tuple
        self.W = {} # Maps history (as tuple) to list of tuples (state, probability)

        # Other parametes
        self.n = 100 # Number of simulate() runs in planner
        self.dmax = 5 # Maximum POMCPOW tree depth
        self.historiesTested = [] # Histories tested from initial history [list]- used in planner to get max_a Q(a)
        self.actionsTested = []  # Actions tested from initial history [list]- used in planner to get max_a Q(a)
        self.actionsTested_list = [] # Actions (as 1D arrays with the agents ordering) tested from initial history [list]
        self.Q_values_actions_tested = [] # Q-values of actions tested in initial history
        self.gamma = 1.0 # discount factor

        # Transition dynamics
        self.dynamics = dynamics

        # rollout planner
        self.rollout_planner = GoToGoalv2(agents, self.id, nx.complete_graph(self.num_agents))

        # convex function approximator
        self.cvx_approximator = CVX_Approximator(self.agents)
        self.cum_cost = 0

    """
        The following is the function plan(b): takes as argument a state (if MDP) or belief (if POMDP) and performs the main POMCPOW simulation n times
    """
    def plan(self, b):
        for i in range(0, self.n):
            if self.is_mdp:
                self.simulate(b, (), self.dmax)
            else:
                s = b.sample_state() # TO DO: if we want a POMDP implementation
                self.simulate(s, (), self.dmax)

        if len(self.historiesTested) == 1:
            Qa = [self.Q[self.historiesTested[0]]]
        else:
            Qa = list(itemgetter(*self.historiesTested)(self.Q)) # Q values for all actions tested at root node


        self.Q_values_actions_tested = -np.asarray(Qa) + self.cum_cost

        # train amap estimator
        self.cvx_approximator.amap.fit(np.asarray(self.actionsTested_list), self.Q_values_actions_tested.reshape(-1, 1))
        # return best action from root node as tuple
        best_action = self.actionsTested[Qa.index(max(Qa))]
        return best_action, max(Qa)

    """
        The following is the simulate(s,h,d) function:
            - Takes as arguments a state, a history (tuple) and a maxiumum depth d
            - Performs one MCTS
        The function follows exactly the structure of "Algorithm 2" of Sunberg et al.
    """
    def simulate(self, s, h, d):
        if not(h in self.B.keys()):
            self.B[h] = []
        self.B[h].append(s)

        if d == 0:
            R = 0
            return R

        action = self.action_prog_widen(h, s)

        if not h and not any(np.linalg.norm(np.asarray(action).flatten()-np.asarray(_).flatten()) <= 0.001 for _ in self.actionsTested):
            self.historiesTested.append((*h, action))
            self.actionsTested.append(action)
            self.actionsTested_list.append(np.asarray(action).reshape(-1,))

        next_state, observation, r = self.gen_model(s, action)

        if len(self.C[(*h, action)]) <= self.ko*self.N[(*h, action)]**self.ao:
            pass
            # no observation can be seen more than one time, because of continuity
            # if self.debug:
            #     print("new observation") #,observation)
        else:
            # if self.debug:
            #     print("old observation")
            choices = list(range(len(self.C[(*h, action)])))
            distribution = [1/len(self.C[(*h, action)]) for _ in self.C[(*h, action)]]
            choice = list(np.random.choice(choices,
                          size=1, p = distribution))[0]
            observation = self.C[(*h, action)][choice]

        # if self.debug:
        #     self.bnew_x.append(observation.mus[0][0])
        #     self.bnew_y.append(observation.mus[0][1])

        # Add new belief to its history tuple in Belief dictionary
        if not((*h, action, observation) in self.B.keys()):
            self.B[(*h, action, observation)] = []
        self.B[(*h, action, observation)].append(next_state)

        if not((*h, action, observation) in self.W.keys()):
            self.W[(*h, action, observation)] = []
        self.W[(*h, action, observation)].append(self.get_prob(observation, s, action, next_state))

        # since continuous state space always chooses first option
        if not(observation in self.C[(*h, action)]):
            self.C[(*h, action)].append(observation)
            rollout_reward = self.gamma*self.rollout(next_state, (*h, action, observation), d-1)
            # if self.debug:
            #     print("rollout")
            #     self.R[action] = [r,rollout_reward]
            R = r + rollout_reward
        else:
            # if self.debug:
            #     print("simulating")
            W_sum = sum(self.W[(*h, action, observation)])
            distribution = [self.W[(*h, action, observation)][_]/W_sum for _ in range(len(self.W[(*h, action, observation)]))]
            W_choice = list(np.random.choice([_ for _ in range(len(self.W[(*h, action, observation)]))],
                                                size=1, p=distribution))[0]
            next_state = self.B[(*h, action, observation)][W_choice]
            r = self.gen_reward(s, action, next_state) # this is a deterministic reward: if you are at state s', the reward is the distance to goals
            R = r + self.gamma*self.simulate(next_state, (*h, action, observation), d-1)

        self.N[h] = self.N[h] + 1

        self.N[(*h, action)] = self.N[(*h, action)] + 1

        self.Q[(*h, action)] = self.Q[(*h, action)] + (R - self.Q[(*h, action)])/self.N[(*h, action)]

        return R

    """
        The following is the actionProgWiden(h) function; it performs progressive widening in the action level
        It is based on the algorithm in "Listing 1" from Sunberg, Kochenderfer.
        It also takes care of key initializations in dictionaries.
    """
    def action_prog_widen(self, h, s):
        # Initialize key in dictionary if not present
        if not(h in self.C.keys()):
            self.C[h] = []
            # add action that reduces
            action = tuple([
                (0, 0)\
                for _ in range(0, self.num_agents)
                ])
            self.C[h].append(action)
        if not(h in self.N.keys()):
            self.N[h] = 1
        else:
            # if self.debug:
            #     print("PW",len(self.C[h]),self.ka*self.N[h]**self.aa)
            if len(self.C[h]) <= self.ka*self.N[h]**self.aa:
                action = tuple([
                    (list(np.random.uniform(self.yaw_lb, self.yaw_ub, 1) )[0], list( np.random.uniform(self.acc_lb, self.acc_ub, 1) )[0])\
                                    for _ in range(0, self.num_agents)
                ]) # nextAction(h) assumed to sample one action for each agent uniformly
                self.C[h].append(action)
                # if self.debug:
                #     print("new action,",action)
            else:
                action = random.choice(self.C[h])
                # if self.debug:
                #     print("chose old action!",action)

        # Initialize key in dictionary if not present
        if not((*h, action) in self.Q.keys()):
            self.Q[(*h, action)] = 0
        if not((*h, action) in self.N.keys()):
            self.N[(*h, action)] = 0.01
        if not((*h, action) in self.C.keys()):
            self.C[(*h, action)] = []

        historiesTestedh = tuple([(*h, a) for a in self.C[h]])
        actionsTestedh = [a for a in self.C[h]]

        if len(actionsTestedh) == 1:
            Q_ha =[self.Q[(historiesTestedh[0])]]
        else:
            Q_ha = list(itemgetter(*historiesTestedh)(self.Q)) # Q values for all actions tested at history h node


        explorVal = [self.c*np.sqrt(np.log(self.N[h])/self.N[(*h, a)]) for a in self.C[h]]
        UCB_score = [sum(x) for x in zip(Q_ha, explorVal)]

        # return best action from root node
        return actionsTestedh[UCB_score.index(max(UCB_score))]

    def gen_model(self, s, action):
        state = np.zeros((P.agent_states.shape[0], len(action)))
        next_state = []
        for _ in range(len(action)):
            state[0, _] = s[_][0]
            state[1, _] = s[_][1]

        # Add noise to the action to model uncertainty in dynamics
        true_action = [[0.0, 0.0] for _ in range(len(action))]
        for _ in range(len(action)):
            true_action[_][0] = action[_][0] + np.random.normal(0.0, P.yaw_std_dev)
            true_action[_][1] += action[_][1] + np.random.normal(0.0, P.acc_std_dev)
        newState = self.dynamics.update_states(state.copy(), true_action, False)
        for _ in range(len(action)):
            next_state.append(tuple(newState[:, _]))
        next_state = tuple(next_state)
        observation = self.gen_observ(next_state)
        reward = self.gen_reward(s, action, next_state)

        return next_state, observation, reward

    def gen_reward(self, s, action, next_s):
        next_state = np.zeros((P.agent_states.shape[0], len(self.agents)))
        for _ in range(len(self.agents)):
            next_state[0, _] = next_s[_][0]
            next_state[1, _] = next_s[_][1]

        reward = calc_reward(next_state[0:2, :], P.goal_states[:2, self.agents], [_ for _ in range(len(action))], fully_connected=True)

        return reward

    def gen_observ(self, next_state):
        """Returns tuple of observations

        Parameters
        ----------
        next_state : tuple
            tuple of state tuples for each agent ((stateofagent1),
            (stateofagent2),...)

        Returns
        -------
        observation : tuple
            Concatenated IMU and UWB measurements

        """

        if self.is_mdp:
            observation = deepcopy(next_state)
        else:
            raise NotImplementedError
            # TO DO: if we want a POMDP implementation
        return observation


    def get_prob(self, observation, s, action, next_state):
        """Probability of observation

        Parameters
        ----------
        observation: tuple
            Observation received.
        s : tuple of tuples
            state for each agent
        action : tuple of tuples
            actions for each agent
        next_state : tuple
            tuple of state tuples for each agent ((stateofagent1),
            (stateofagent2),...)

        Returns
        -------
        prob : float
            Probability of receiving observation at next_state

        """

        prob = 1.0

        if not self.is_mdp:
            raise NotImplementedError
            # TO DO: if we want a POMDP implementation
        return prob

    def rollout(self, s, h, d):
        reward = 0.0
        if d == 0:
            return reward

        state = np.zeros((P.agent_states.shape[0], len(self.agents)))
        for _ in range(len(self.agents)):
            state[0, _] = s[_][0]
            state[1, _] = s[_][1]

        action = self.rollout_planner.plan(state)

        next_state = self.dynamics.update_states(state, action, False)

        reward = calc_reward(next_state[:2, :], P.goal_states[:2, self.agents], [_ for _ in range(len(self.agents))], fully_connected=True)

        next_s = []
        for _ in range(len(action)):
            next_s.append(tuple(next_state[:, _]))
        next_s = tuple(next_s)

        return reward + self.gamma*self.rollout(next_s, h, d-1)

    """
        The following function resets the dictionaries
    """
    def reset(self):
        self.C={}
        self.N={}
        self.Q={}
        self.B={}
        self.W={}


        self.historiesTested = []
        self.actionsTested_list = []
        self.actionsTested = []
        self.Q_values_actions_tested = []  # Q-values of actions tested in initial history

        # convex function approximator
        self.cvx_approximator = CVX_Approximator(self.agents)

        if self.debug:
            self.R={}
            self.bnew_x = []
            self.bnew_y = []

