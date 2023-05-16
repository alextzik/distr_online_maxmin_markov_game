# Distributed Online Planning for Min-Max Problems in Networked Markov Games

This repository contains the developed code for the paper submitted to NeurIPS 2023

## Overview of Proposed Method
![](https://github.com/alextzik/distr_online_maxmin_markov_game/blob/main/GIFs/OverviewProposedMethod.png)

## Code Structure

The structure of the code is as follows:
* ```main.py```
  * Runs the main simulation loop
    * The user should specify the algorithms to be run in line 44. The available options are:
      * "dec_baseline_only_POMDP": POMCPOW baseline
      * "dec_baseline_liujia": implementation of the algorithm found in the paper "Robust formation control of discrete-time multi-agent
        systems by iterative learning approach"
      * "dec_baseline_cvxpy": termed optimal in the paper
      * "dec_baseline_roll": termed rollout baseline in the paper
      * "dec_mcts_maxmin": proposed algorithm
    * The user should specify the number of runs for each method in line 42 and the number of steps per run in line 43
    * The output results for the runs are stored in the log directory

* ```parameters.py```
  * Contains the parameters for the simulation runs, such as the agents' initial state, goal state, the network topology, noise characterisitcs (currently noise is assumed zero), the doubly stochastic matrix for the distributed optimization (needs to be recalculated if number of agents changes, by uncommenting lines 85-99)
* ```result_plotting.py```
  * Used to plot the results obtained after running ```main.py```
  * The user needs to specify the results directory in line 26 and the methods to be plotted in line 25

## Results

Below we provide trajectory animations for the results pertaining to Figure 1 of the paper. Also available in the folder /GIFs.

In the following figures 'C' denotes the current relative state and 'G' the goal relative state in the desired formation between the agents that are connected by the edge. 
### Proposed algorithm 
![h](https://github.com/alextzik/decentralized_mcts/blob/comp_with_liu_jia/GIFs/proposed.gif)
### Optimal action sequence 
![h](https://github.com/alextzik/decentralized_mcts/blob/comp_with_liu_jia/GIFs/cvxpy.gif)










