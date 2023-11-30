"""
This code file contains implementation of epsilon greedy
"""

import numpy as np
import matplotlib.pyplot as plt

output_directory = 'Output_directory/'
num_trials = 10000
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

def run_experiment():
    # Initialise bandit objects
    bandits = [Bandit(win_rate) for win_rate in bandit_true_win_rates]

    # Initialise rewards array to store reward collected in each trial
    rewards = np.zeros(num_trials)

    # Maintain count for number of times exploited (greedy choice) and explored (random) and actual optimal choice made
    num_times_explored = 0
    num_times_exploited = 0
    num_optimal = 0

    # Index of the optimal bandit - unknown while conducting experiment
    actual_optimal_bandit = np.argmax([b.true_win_rate for b in bandits])

    for i in range(num_trials):

        # Use epsilon greedy strategy to select the next bandit
        if np.random.random() < epsilon:
            num_times_explored += 1
            next_bandit = np.random.randint(len(bandits))
        else:
            num_times_exploited += 1
            next_bandit = np.argmax([b.estimate_win_rate for b in bandits])

        if next_bandit == actual_optimal_bandit:
            num_optimal += 1

        # Pull the arm of the bandit and collect the reward
        sample = bandits[next_bandit].pull()
        rewards[i] = sample

        # Update the estimate of win rate of bandit
        bandits[next_bandit].update_estimate(sample)

    for i, b in enumerate(bandits):
        print(f"Bandit {i+1} : MLE win_rate : {b.estimate_win_rate} | Actual win rate : {b.true_win_rate}")

    title = f"Total Reward : {np.sum(rewards)} |\n num_times_explored : {num_times_explored} |\n num_times_exploited : {num_times_exploited} |\n num_optimal_choice : {num_optimal}"
    print(title)
    cumulative_rewards = np.cumsum(rewards)
    expected_win_rates = cumulative_rewards / (np.arange(num_trials) + 1)
    plt.figure()
    plt.plot(expected_win_rates, label="Expected win rate")
    plt.plot(np.ones(num_trials)*np.max(bandit_true_win_rates), label="Max win rate")
    plt.legend()
    plt.title(title)
    plt.xlabel("Win Rate")
    plt.ylabel("Trials")
    plt.savefig(f"{output_directory}/epsilon_greedy_expt.png")

if __name__ == '__main__':
    run_experiment()