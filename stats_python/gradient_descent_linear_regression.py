"""
This code file implements a gradient descent in python for calculating optimal parameters for
Linear Regression
"""

# Import Libraries
import copy
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

output_directory = 'Output_directory/'

def visualise_cost_over_iterations(b, W, storing_iterative_cost, title):
    batches = np.arange(storing_iterative_cost.shape[1])
    cost = storing_iterative_cost.values.squeeze()
    plt.plot(batches , cost, label=title)
    plt.legend()
    plt.xlabel("Number of iterations")
    plt.ylabel("Mean Squared Error")
    plt.title("Gradient Descent sim")

def initialise_params(n_features):
    bias = random.random()
    weight = np.random.rand(n_features)
    return bias, weight

def linear_predict(b, W, X):
    Y_pred = b + np.dot(X, W)
    return Y_pred


def mse_cost_function(Y, Y_pred):
    residue = Y - Y_pred
    samples = len(Y)
    J = np.sum(np.dot(residue.T, residue)) / samples
    return J

def update_params(X, Y, Y_pred, b_current, W_current, learning_rate):
    samples = len(Y)
    dJ_db = (2 / samples) * np.sum(Y_pred - Y)
    dJ_dw = (2 / samples) * np.sum(np.dot(Y_pred - Y, X))
    b_updated = b_current - learning_rate * dJ_db
    W_updated = W_current - learning_rate * dJ_dw
    return b_updated, W_updated

def run_grad_descent(X, Y, learning_rate, num_iterations):

    # Initialise object to store iterative cost
    storing_iterative_cost = {}

    # Initialise regression parameters y = w1x1 + w2x2 + w3x3 + b1 or Y = (W.T)X + b
    b, W = initialise_params(n_features=X.shape[1])
    print("Initialized value of bias and weight : ")
    print(f"Bias : {b}")
    print(f"Weights : Expenditure on TV ad, Radio ad and Newspaper ad [{W[0]} , {W[1]} , {W[2]}]")

    counter = 0
    for num_iter in range(num_iterations):
        Y_pred = linear_predict(b, W, X)

        # Calculate current cost
        J = mse_cost_function(Y, Y_pred)
        if num_iter == 0:
            print(f"Starting MSE : {np.round(J, 4)}")

        # Store previous parameters for update
        b_current, W_current = copy.deepcopy(b), copy.deepcopy(W)

        # Update parameters in each step
        b, W = update_params(X, Y, Y_pred, b_current, W_current, learning_rate)

        # Store cost every 10 iterations
        if num_iter%1 == 0:
            print(f" Iteration : {num_iter+1}")
            storing_iterative_cost[counter] = J
            counter = counter + 1

    print("Initialized value of bias and weight to predict Sales: ")
    print(f"Bias : {b}")
    print(f"Weights : Expenditure on TV ad, Radio ad and Newspaper ad [{W[0]} , {W[1]} , {W[2]}]")
    storing_iterative_cost = pd.DataFrame(storing_iterative_cost, index=np.arange(1))
    return b, W, storing_iterative_cost


if __name__ == '__main__':
    # Read Data
    df = pd.read_csv('Data_directory/Advertising.csv')
    print("The columns in the data : ")
    print(df.columns.tolist())

    input_columns = ["TV", "radio", "newspaper"]
    target_column = ["sales"]
    X = df[input_columns]
    Y = df[target_column]

    # Normalise the data for easy visualisation and faster convergence
    Y = np.array((Y - Y.mean()) / Y.std())
    X = X.apply(lambda rec: (rec - rec.mean()) / rec.std(), axis=0)

    # Define hyper parameters
    learning_rate = 0.001
    num_iterations = 10

    # Run Gradient descent :
    # W_j_updated = W_j_initial - learning_rate * partial_derivative_wrt_W_j_of_J
    b, W, storing_iterative_cost = run_grad_descent(X, Y, learning_rate, num_iterations)

    plt.figure()
    title = f"learning rate : {learning_rate} , num_iterations : {num_iterations}"
    visualise_cost_over_iterations(b, W, storing_iterative_cost, title)
    plt.savefig(f"{output_directory}/grad_descent_simulation.png")

    print("\n End.")





