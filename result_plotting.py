"""
    Result plotting
"""

__authors__ = "D. Knowles, A. Tzikas"
__date__ = "10 Apr 2022"

import os
import pickle

import numpy as np
import matplotlib.pyplot as plt

import lib.parameters as P

AGENT_COLORS = ["#820000", # digital red dark
                "#006cb8", # digital blue
                "#008566", # digital green
                # "#620059",   # plum
                "#E98300",   # poppy
                "#FEDD5C",   # illuminating
                # "#E04F39",   # spirited
                ]

METHODS = ["dec_baseline_only_POMDP", "dec_baseline_cvxpy", "dec_baseline_roll", "dec_mcts_maxmin"]
LOG = "20230427154214" # Change to desired log file

plot_format = {}
for method in METHODS:
    plot_format[method] = {}
    if method == "dec_mcts_maxmin":
        plot_format[method]["label"] = "proposed"
        plot_format[method]["color"] = "blue"
        plot_format[method]["color_std"] = "lightblue"
        plot_format[method]["ls"] = "-"
    elif method == "dec_baseline_cvxpy":
        plot_format[method]["label"] = "optimal"
        plot_format[method]["color"] = "darkorchid"
        plot_format[method]["color_std"] = "plum"
        plot_format[method]["ls"] = "-"
    elif method == "dec_baseline_roll":
        plot_format[method]["label"] = "rollout baseline"
        plot_format[method]["color"] = "darkgreen"
        plot_format[method]["color_std"] = "lightgreen"
        plot_format[method]["ls"] = "-"
    elif method == "dec_baseline_liujia":
        plot_format[method]["label"] = "dec_baseline_liujia"
        plot_format[method]["color"] = "red"
        plot_format[method]["color_std"] = "lightgreen"
        plot_format[method]["ls"] = "-"
    elif method == "dec_baseline_only_POMDP":
        plot_format[method]["label"] = "POMCPOW baseline"
        plot_format[method]["color"] = "orange"
        plot_format[method]["color_std"] = "lightgreen"
        plot_format[method]["ls"] = "-"


def main():
    file_dir = os.path.dirname(os.path.realpath(__file__))

    log_dir = os.path.join(file_dir,"log",LOG)

    plot_per_neighborhood(log_dir, "rewards_per_neighb", ylim=None)

    plot_regret(log_dir, "rewards_per_neighb", ylim=None)

    plt.show()


def plot_per_neighborhood(log_dir, which, ylim=None):
    graph = P.G
    fig = plt.figure(figsize=(11,6))

    titles = {}
    titles["rewards_per_neighb"] = "Agent Instantaneous Rewards"

    for method in METHODS:
        filename = os.path.join(log_dir,method,"avg_"+which+".csv")
        data_avg = np.loadtxt(filename, delimiter=",")

        if len(data_avg) <= 5:
            num_figrows = 3
        else:
            num_figrows = 4

        STEPS = len(data_avg[0])
        for aa in range(len(data_avg)):
            data_agent = np.loadtxt(os.path.join(log_dir,method,which+"0"+".csv"), delimiter=",")[aa,:].reshape(1,-1)
            for _ in range(1, len(os.listdir(os.path.join(log_dir,method)))-1):
                data_agent = np.concatenate([data_agent, np.loadtxt(os.path.join(log_dir,method,which+str(_)+".csv"), delimiter=",")[aa,:].reshape(1,-1)], axis=0)
            std = np.std(data_agent, axis=0)

            print("Neighborhood of " + str(aa + 1) + " Reward - " + method + " : ", np.sum(data_avg[aa][:STEPS]))
            plt.subplot(2, num_figrows, int(aa + 1))
            plt.plot(range(STEPS), data_avg[aa][:STEPS],
                     label=plot_format[method]["label"],
                     color=plot_format[method]["color"],
                     linestyle=plot_format[method]["ls"],
                     )
            plt.fill_between(range(STEPS), data_avg[aa][:STEPS]-std[:STEPS], data_avg[aa][:STEPS]+std[:STEPS], color=plot_format[method]["color_std"])

    for aa in range(data_avg.shape[0]):
        plt.subplot(2,num_figrows,int(aa+1))
        plt.title("Agent " + str(aa + 1))
        plt.xlabel('Timestep')
        plt.ylabel("instantaneous reward")
        plt.xlim(0,STEPS)
        plt.ylim(-100, 5)
        if ylim != None:
            plt.ylim(ylim)

    # plt.suptitle(titles[which])

    plt.tight_layout()
    plt.legend()

    fig.savefig(os.path.join(log_dir,which+"_average.eps"),
                dpi=300.,
                format="eps",
                bbox_inches="tight")


def plot_regret(log_dir, which, ylim=None):
    graph = P.G
    fig = plt.figure(figsize=(11,6))

    titles = {}
    titles["rewards_per_neighb"] = "Per Agent Regrets"

    for method in METHODS:
        if method == "dec_mcts_maxmin":
            filename = os.path.join(log_dir,method,"avg_"+which+".csv")
            data_avg = np.loadtxt(filename, delimiter=",")
            best_data_avg = np.loadtxt(os.path.join(log_dir,"dec_baseline_cvxpy","avg_"+which+".csv"), delimiter=",")

            if len(data_avg) <= 5:
                num_figrows = 3
            else:
                num_figrows = 4

            STEPS = len(data_avg[0])
            for aa in range(len(data_avg)):
                data_agent = np.loadtxt(os.path.join(log_dir,method,which+"0"+".csv"), delimiter=",")[aa,:].reshape(1,-1)
                best_data_agent = np.loadtxt(os.path.join(log_dir,"dec_baseline_cvxpy",which+"0"+".csv"), delimiter=",")[aa,:].reshape(1,-1)

                regret = np.zeros((1,STEPS))

                for _ in range(STEPS):
                    regret[0, _] = 1 / (_ + 1) * (np.sum(best_data_agent[:_ + 1]) - np.sum(data_agent[:_ + 1]))

                for _ in range(1, len(os.listdir(os.path.join(log_dir,method)))-1):
                    data_agent = np.loadtxt(os.path.join(log_dir,method,which+str(_)+".csv"), delimiter=",")[aa,:].reshape(1,-1)
                    best_data_agent = np.loadtxt(os.path.join(log_dir, "dec_baseline_cvxpy", which + str(_) + ".csv"),
                                                        delimiter=",")[aa, :].reshape(1, -1)
                    r = np.zeros((1, STEPS))

                    for _ in range(STEPS):
                        r[0, _] = 1 / (_ + 1) * (np.sum(best_data_agent[:_ + 1]) - np.sum(data_agent[:_ + 1]))

                    regret = np.concatenate([regret, r], axis=0)

                std = np.std(regret, axis=0)

                regret = np.zeros(STEPS)

                for _ in range(STEPS):
                    regret[_] = 1/(_+1)*(np.sum(best_data_avg[aa][:_+1]) - np.sum(data_avg[aa][:_+1]))

                plt.subplot(2, num_figrows, int(aa + 1))
                plt.plot(range(STEPS), regret[:STEPS],
                         label=plot_format[method]["label"],
                         color=plot_format[method]["color"],
                         linestyle=plot_format[method]["ls"],
                         )
                plt.fill_between(range(STEPS), regret[:STEPS]-std[:STEPS], regret[:STEPS]+std[:STEPS], color=plot_format[method]["color_std"])

                for aaa in range(data_avg.shape[0]):
                    plt.subplot(2,num_figrows,int(aaa+1))
                    plt.title("Agent " + str(aaa + 1))
                    plt.xlabel('Timestep')
                    plt.ylabel("regret")
                    plt.xlim(0,STEPS)
                    if ylim != None:
                        plt.ylim(ylim)

                # plt.suptitle(titles[which])

                plt.tight_layout()
                plt.legend(bbox_to_anchor=(1.0, 0.9))

                fig.savefig(os.path.join(log_dir,"regret"+"_average.eps"),
                            dpi=300.,
                            format="eps",
                            bbox_inches="tight")


if __name__ == "__main__":
        main()