import lib.parameters as Params
from matplotlib import pyplot as plt
import numpy as np
import os



"""
    This function saves an image of the current agent locations and their desired and goal distances
"""
def plot_agents(agent_states, step, algo):
    eps = 0.1
    plt.figure()
    for aa in range(agent_states.shape[1]):
        plt.plot(agent_states[0, aa], agent_states[1, aa], marker='o', markersize=20)
        plt.text(agent_states[0, aa]+eps, agent_states[1, aa]+eps, f"{aa}")
        for jj in [k for k in range(agent_states.shape[1]) if k != aa and k<=aa]:
            plt.plot(agent_states[0, [aa, jj]], agent_states[1, [aa, jj]])

            current_dist = np.round((agent_states[:2, aa] - agent_states[:2, jj]), 1)
            goal_dist = np.round((Params.goal_states[:2, aa]-Params.goal_states[:2, jj]), 1)

            plt.text((agent_states[0, aa] + agent_states[0, jj])/2 + eps, (agent_states[1, aa]+agent_states[1, jj])/2 + eps, \
                        f"C: {current_dist}, G: {goal_dist}",)

    file_dir = os.path.dirname(os.path.realpath(__file__))
    log_dir = os.path.join(file_dir, "log")
    photos_path = os.path.join(log_dir, algo)
    if step == 0:
        os.mkdir(photos_path)

    plt.savefig(os.path.join(photos_path, algo)+f"step_{step}.png")
    plt.close()

    return 0







