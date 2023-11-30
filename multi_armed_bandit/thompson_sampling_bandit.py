"""
This file contains implementation of thompson sampling to select optimal bandit at each iteration
"""

from __future__ import print_function, division
from builtins import range
# Note: you may need to update your version of future
# sudo pip install -U future

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import beta

output_directory = 'Output_directory/'
np.random.seed(43)
NUM_TRIALS = 2000
BANDIT_PROBABILITIES = [0.2, 0.5, 0.75]


class Bandit:
    def __init__(self, true_mean):
        self.true_mean = true_mean  # Unknown while actually experimenting
        self.alpha = 1
        self.beta = 1
        self.N = 0

    def pull(self):
        return np.random.random() < self.true_mean

    def sample(self):
        return np.random.beta(self.alpha, self.beta)

    def update(self, x):
        self.alpha += x
        self.beta += 1 - x
        self.N += 1


def plot_each_trial(bandits, trial, index_plot):
    # plt.subplot(3, 3, index_plot)
    x = np.linspace(0, 1, 200)
    for b in bandits:
        y = beta.pdf(x, b.alpha, b.beta)
        plt.plot(x, y,
                 label=f"Actual mean: {b.true_mean:.4f}, count(reward=1)/count(reward=0) = {b.alpha - 1}/{b.N}")
    plt.title(f"Bandit distributions after {trial} trials")
    plt.xlabel("Trials")
    plt.ylabel("Probability")
    plt.legend()
    plt.savefig(f"{output_directory}/{trial}_trials_thompson_sampling_simulation.png")
    plt.close()


def run_experiment():
    bandits = [Bandit(p) for p in BANDIT_PROBABILITIES]
    num_trials_each_expt = [5, 10, 20, 100, 200, 400, 1000, 1500, 2000]
    rewards = np.zeros(NUM_TRIALS)
    index_plot = 1
    for i in range(NUM_TRIALS):
        # Thompson sampling
        next_bandit = np.argmax([b.sample() for b in bandits])

        # plot the posteriors
        plt.close()
        plt.figure(figsize=(8, 8))
        if i in num_trials_each_expt:
            print(f"==== Trial : {i + 1} == \n")
            plot_each_trial(bandits, i, index_plot)
            index_plot += 1

        # pull the arm for the bandit with the largest sample, update rewards and update distribution of true mean
        x = bandits[next_bandit].pull()
        rewards[i] = x
        bandits[next_bandit].update(x)

    for i, b in enumerate(bandits):
        print(f"Bandit {i + 1} : Posterior of mean estimate ~ Beta({b.alpha},{b.beta}) | Actual mean : {b.true_mean}")
    print("total reward earned:", rewards.sum())
    print("overall mean estimate / win rate:", rewards.sum() / NUM_TRIALS)
    print("num times selected each bandit:", [b.N for b in bandits])


if __name__ == "__main__":
    run_experiment()
