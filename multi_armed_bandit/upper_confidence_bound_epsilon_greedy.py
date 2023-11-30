"""
This code file contains implementation of upper confidence of probability in bandit selection
"""

import numpy as np
import matplotlib.pyplot as plt

output_directory = 'Output_directory/'
num_trials = 100000
epsilon = 0.1
# Will be unknown while conducting experiment
bandit_true_win_rates = [0.2, 0.5, 0.75]

class Bandit:
    """
    A Bandit can be an analogy for a Recommendation for a new movie, Advertisement, Website design etc
    The idea is to sample from a real-life distribution where we don't know the true distribution
    """
    def __init__(self, true_win_rate):
        self.true_win_rate = true_win_rate
        self.estimate_win_rate = 0
        self.N = 0

    def pull(self):
        return np.random.random() < self.true_win_rate

    def update_estimate(self, x):
        self.N = self.N + 1
        self.estimate_win_rate = ((self.N - 1)*self.estimate_win_rate + x) / self.N


def upper_confidence_bound_heuristic(estimate_mean, num_trials_completed, num_selections_bandit):
    return estimate_mean +  np.sqrt(2 * np.log(num_trials_completed) / num_selections_bandit)

def run_experiment():
    # Initialise bandit objects
    bandits = [Bandit(win_rate) for win_rate in bandit_true_win_rates]

    # Initialise rewards array to store reward collected in each trial
    rewards = np.zeros(num_trials)

    # Maintain count for number of times exploited (greedy choice) and explored (random) and actual optimal choice made
    num_trials_completed = 0

    # play each bandit once
    for j in range(len(bandits)):
        x = bandits[j].pull()
        num_trials_completed += 1
        bandits[j].update_estimate(x)

    for i in range(num_trials):
        # Use upper confidence bound strategy to select the next bandit
        next_bandit = np.argmax([upper_confidence_bound_heuristic(b.estimate_win_rate, num_trials_completed, b.N) for b in bandits])
        sample = bandits[next_bandit].pull()
        rewards[i] = sample
        num_trials_completed += 1
        bandits[next_bandit].update_estimate(sample)

    for i, b in enumerate(bandits):
        print(f"Bandit {i+1} : MLE win_rate : {b.estimate_win_rate} | Actual win rate : {b.true_win_rate}")

    title = f"Total Reward : {np.sum(rewards)} | Trials  : {num_trials_completed}"
    print(title)
    cumulative_rewards = np.cumsum(rewards)
    expected_win_rates = cumulative_rewards / (np.arange(num_trials) + 1)
    plt.figure()
    plt.plot(expected_win_rates, label="MA win rate")
    plt.plot(np.ones(num_trials)*np.max(bandit_true_win_rates), label="Max win rate")
    plt.xscale('log')
    plt.legend()
    plt.title(title)
    plt.xlabel("Win Rate")
    plt.ylabel("Trials")
    plt.savefig(f"{output_directory}/upper_confidence_bound.png")

if __name__ == '__main__':
    run_experiment()