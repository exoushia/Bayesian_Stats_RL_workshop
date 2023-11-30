"""
This code file contains implementation of epsilon greedy for different epsilon values
Here the reward will be sampled from a normal distribution from a bandit with mean=m, std=1
"""

from __future__ import print_function, division
from builtins import range
# Note: you may need to update your version of future
# sudo pip install -U future

import numpy as np
import matplotlib.pyplot as plt

output_directory = 'Output_directory/'

class Bandit:
    """
    A Bandit can be an analogy for a Recommendation for a new movie, Advertisement, Website design etc
    The idea is to sample from a real-life distribution where we don't know the true distribution
    """
    def __init__(self, true_mean):
        self.true_mean = true_mean
        self.estm_mean = 0
        self.N = 0

    def pull(self):
        # Reward is sampled from a normal distribution ~N(mean, 1)
        return np.random.randn() + self.true_mean

    def update(self, x):
        self.N += 1
        self.estm_mean = (1 - 1.0 / self.N) * self.estm_mean + 1.0 / self.N * x


def run_experiment(true_mean_1, true_mean_2, true_mean_3, epsilon, N):
    bandits = [Bandit(true_mean_1), Bandit(true_mean_2), Bandit(true_mean_3)]
    rewards = np.empty(N)

    for i in range(N):
        # epsilon greedy
        p = np.random.random()
        if p < epsilon:
            next_bandit = np.random.choice(3)
        else:
            next_bandit = np.argmax([b.estm_mean for b in bandits])
        reward = bandits[next_bandit].pull()
        bandits[next_bandit].update(reward)
        rewards[i] = reward

    cumulative_average_reward = np.cumsum(rewards) / (np.arange(N) + 1)
    print(f"\n ===== Epsilon : {epsilon} =====")
    for i, b in enumerate(bandits):
        print(f"Bandit {i+1} : Estimate Mean : {b.estm_mean} | True Mean : {b.true_mean}")

    return cumulative_average_reward


if __name__ == '__main__':
    m1, m2, m3 = 1, 2, 3
    eps1, eps2, eps3 = 0.1, 0.05, 0.01
    num_trials = 100000
    cumulative_average_reward_1 = run_experiment(m1, m2, m3, eps1, num_trials)
    cumulative_average_reward_05 = run_experiment(m1, m2, m3, eps2, num_trials)
    cumulative_average_reward_01 = run_experiment(m1, m2, m3, eps3, num_trials)

    # plot moving average of reward :
    # Plotted at log scale to see the variations in convergence
    # otherwise, because of fast convergence, differences won't be noticeable
    plt.figure()
    plt.plot(cumulative_average_reward_1, label='eps = 0.1')
    plt.plot(cumulative_average_reward_05, label='eps = 0.05')
    plt.plot(cumulative_average_reward_01, label='eps = 0.01')
    plt.plot(np.ones(num_trials) * m1, label='Bandit 1')
    plt.plot(np.ones(num_trials) * m2, label='Bandit 2')
    plt.plot(np.ones(num_trials) * m3, label='Bandit 3')
    plt.legend()
    plt.ylabel("Mean Reward")
    plt.xlabel("Trials")
    plt.xscale('log')
    plt.savefig(f"{output_directory}/comparing_diff_epsilon_epsilon_greedy.png")
    plt.xscale('log')
    plt.close()



