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
edges_full = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (1, 4), (2, 3), (2, 4), (3, 4)]
edges_cycl = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 0)]

G_full = nx.Graph()
G_full.add_nodes_from(nodes)
G_full.add_edges_from(edges_full)
laplacian_full = nx.laplacian_matrix(G_full).toarray()

G_cycl = nx.Graph()
G_cycl.add_nodes_from(nodes)
G_cycl.add_edges_from(edges_cycl)
laplacian_cycl = nx.laplacian_matrix(G_cycl).toarray()
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
# W = np.multiply(W, nx.adjacency_matrix(G_cycl).toarray() + np.diag([1.0 for _ in range(len(nodes))]))

# rsum = None
# csum = None

# while (np.any(rsum != 1)) | (np.any(csum != 1)):
#     W /= W.sum(0)
#     W = W / W.sum(1)[:, np.newaxis]
#     rsum = W.sum(1)
#     csum = W.sum(0)

# print(W)
NUM_ITERATIONS_PER_STEP = 1000

W_full = np.array([[0.55901448, 0.0877849,  0.14932374, 0.20387688, 0.        ],
 [0.24487056, 0.08614041, 0.18714822, 0.10801737, 0.37382344],
 [0.15184044, 0.20077562, 0.13462882, 0.17334824, 0.33940688],
 [0.04427453, 0.45354986, 0.04244017, 0.25934088, 0.20039456],
 [0.,         0.17174921, 0.48645904, 0.25541663, 0.08637511]])

W_cycl = np.array([[0.45607027, 0.27197703, 0.,         0.,         0.2719527],
 [0.18496217, 0.47759681, 0.33744102, 0.,         0.        ],
 [0.,         0.25042616, 0.26403104, 0.4855428,  0.        ],
 [0.,         0.,         0.39852794, 0.06273178, 0.53874029],
 [0.35896756, 0.,         0.,         0.45172542, 0.18930702]])

r = [2.0 for _ in range(len(nodes))]
########################################################################
# END
########################################################################
