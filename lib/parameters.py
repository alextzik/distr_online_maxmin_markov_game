"""
    Parameters for simulation.
"""

__authors__ = "A. Tzikas"
__date__ = "August 2022"

import numpy as np
import networkx as nx


########################################################################
# GRAPH STRUCTURE
########################################################################
# Dictates the communication topology and the agents each agent's MCTS involves
nodes = [0, 1, 2, 3, 4] # 0 to n-1
edges = [(0, 1), (1, 3), (3, 4), (0, 2), (2, 4), (3, 2),  (0, 4), (1, 2), (0, 3)]
G = nx.Graph()
G.add_nodes_from(nodes)
G.add_edges_from(edges)
laplacian = nx.laplacian_matrix(G).toarray()

########################################################################
# STATE SPACE
# ########################################################################
# initial agent states
agent_states = np.array([[2., -1.,  4.,  5., 17.,],# 5, -8], # x position [m]
                         [  8., 1.,  3., -3.,  1.,],#  5, -4], # y position [m]
                         # [  0.,  0.,  0.,  0.,  0.,],#  0., 0.], # z position [m]
                         # [np.pi/2., np.pi/2., np.pi/2., np.pi/2., np.pi/2.,],#  np.pi/2., np.pi/2.], # yaw angle [rad]
                         # # [ 0.0, 0.0, 0.0, 0.0, 0.0], # velocity [m/s]
                         # [ 0.1, 0.1, 0.1, 0.1, 0.1,],# 0.1, 0.1], # velocity [m/s]
                         ])

# desired formation (inter-agent distance set) as a position set
goal_states = np.array([[-40., -20.,  0., 20., 40.,],#  10, 15], # x position [m]
                        [ 0., 40., 40., 40., 0.,],#  10, 20], # y position [m]
                        # [  0.,  0.,  0.,  0.,  0.,],#  0., 0.,], # z position [m]
                        # [np.pi, np.pi/2., np.pi/2., np.pi/2., -np.pi,],#  -np.pi, -np.pi], # yaw angle [rad]
                        # [ 0.0, 0.0, 0.0, 0.0, 0.0,],#  0.0, 0.0], # velocity [m/s]
                        ])

# choose which agent is the main rover
main_rover_id = 3

########################################################################
# ACTION SPACE
########################################################################
# Action space parameters (for each agent)
yaw_lb = -1
yaw_ub = 1
acc_lb = -1
acc_ub = 1
vel_lb = -1
vel_ub = 1

########################################################################
# OBSERVATION SPACE
########################################################################
yaw_std_dev = 0.0  # [m]
acc_std_dev = 0.0  # [m/0]
    # Assume 0 noise in the action.

########################################################################
# SIMULATION
########################################################################
visualize = False
sim_hz = 20  # desired simulation update frequency

########################################################################
# MAP
########################################################################
# grid borders
# x min, x max, dx
grid_x = (-50, 50, 10)
grid_y = (0, 40, 10)

# length of each axis for the xyz -> rgb reference frame
ref_frame_length = 100

########################################################################
# MIN-MAX DISTRIBUTED OPTIMIZATION FRAMEWORK PARAMETERS
########################################################################

# Doubly stochastic matrix for distributed optimization that satisfies with 0 value for non-neighboring agents
# https://people.duke.edu/~ccc14/bios-821-2017/scratch/Python07A.html
# W = np.random.uniform(0.05, 1.0, (len(nodes), len(nodes)))
# W = np.multiply(W, nx.adjacency_matrix(G).toarray() + np.diag([1.0 for _ in range(len(nodes))]))
#
# rsum = None
# csum = None
#
# while (np.any(rsum != 1)) | (np.any(csum != 1)):
#     W /= W.sum(0)
#     W = W / W.sum(1)[:, np.newaxis]
#     rsum = W.sum(1)
#     csum = W.sum(0)
#
# print(W)
NUM_ITERATIONS_PER_STEP = 1000

W = np.array([[0.21292963, 0.3203916,  0.1658977,  0.1291315,  0.17164957],
 [0.31445902, 0.20252932, 0.21113819, 0.27187347, 0.        ],
 [0.16527396, 0.262745,   0.03941108, 0.20046218, 0.33210778],
 [0.11098306, 0.21433408, 0.28928935, 0.15917713, 0.22621638],
 [0.19635433, 0.,         0.29426367, 0.23935572, 0.27002627]])


r = [2.0 for _ in range(len(nodes))]
########################################################################
# END
########################################################################
