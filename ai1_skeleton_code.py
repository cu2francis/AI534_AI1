import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# constants
path = "/home/sam/Downloads/IA1_train.csv"
convergence_treshold = 0.005


# Loads a data file from a provided file location.
def load_data():
    # Your code here:
    rawdata = pd.read_csv(path)
    data = rawdata.apply(pd.to_numeric, errors='coerce')
    return data


# Implements dataset preprocessing, with boolean options to either normalize the data or not, 
# and to either drop the sqrt_living15 column or not.
#
# Note that you will call this function multiple times to generate dataset versions that are
# / aren't normalized, or versions that have / lack sqrt_living15.
def preprocess_data(data, normalize, drop_sqrt_living15):
    # Your code here:
    return data


# Implements the feature engineering required for part 4. Quite similar to preprocess_data.
# Expand the arguments of this function however you like to control which feature modification
# approaches are / aren't active.
def modify_features(data):
    # Your code here:

    return modified_data

def mean_square_error(training_values, target, weights):
    predictions = np.dot(training_values, weights.T)
    sum_value = np.sum((predictions - target) ** 2)
    return (1 / len(target)) * sum_value


def batch_gradient_descent(training_values, target, weights, learning_rate, iterations):
    loss_history = []
    for i in range(iterations):
        prediction = np.dot(training_values, weights.T)
        weights = weights - (learning_rate / len(target)) * np.dot(prediction - target, training_values)
        loss_history.append(mean_square_error(training_values, target, weights))
        if i > 0:
            if loss_history[i-1] < loss_history[i]:
                print("Diverged. Aborting calculation")
                break
            if (loss_history[i-1] - loss_history[i]) < convergence_treshold:
                print("Converged. Aborting calculation")
                break
    return weights, loss_history


# Trains a linear model on the provided data and labels, using the supplied learning rate.
# weights should store the per-feature weights of the learned linear regression.
# losses should store the sequence of MSE losses for each epoch of training, which you will then plot.
def gd_train(data, labels, lr, iterations):
    training_values = data[labels]
    target = data['price']
    weights = np.zeros(training_values.shape[1])
    start_weight = mean_square_error(training_values, target, weights)
    print("Start-Weight: " + str(start_weight))
    training_values = (training_values - training_values.min()) / (training_values.max() - training_values.min())
    batch_weight, batch_loss_history = batch_gradient_descent(training_values, target, weights, lr, iterations)

    print("End-Weight: " + str(mean_square_error(training_values, target, batch_weight)))

    return batch_weight, batch_loss_history, start_weight


# Generates and saves plots of the training loss curves. Note that you can interpret losses as a matrix
# containing the losses of multiple training runs and then put multiple loss curves in a single plot.
def plot_losses(losses, labels, ymax):
    fig = plt.figure(figsize=(10, 6))
    ax = plt.subplot(1, 1, 1)
    ax.set_ylim([0, ymax])
    for index, y in enumerate(losses):
        ax.plot(range(len(y)), y, linewidth=1.5, label=str(labels[index]))
    fig.legend()
    fig.show()
    return

# Invoke the above functions to implement the required functionality for each part of the assignment.

# Part 0  : Data preprocessing.
# Your code here:


# Part 1 . Implement batch gradient descent and experiment with different learning rates.
# Your code here:
def task1a():
    multi_losses = []
    data = load_data()
    data = preprocess_data(data, True, True)
    labels = ["bedrooms", "bathrooms", "sqft_living", "sqft_lot", "floors",
              "waterfront", "view", "condition", "grade", "sqft_above", "sqft_basement",
              "yr_built", "yr_renovated", "zipcode", "lat", "long", "sqft_living15",
              "sqft_lot15"]
    lrs = [1, 2, 3, 4, 5, 6]

    max = 1000000

    for lr in lrs:
        learning_rate = 10 ** (-lr)
        weight_batch, loses, weight = gd_train(data, labels, learning_rate, 4000)
        if weight < max:
            max = weight
        multi_losses.append(loses)
    plot_losses(multi_losses, lrs, max)


task1a()



# Part 2 a. Training and experimenting with non-normalized data.
# Your code here:


# Part 2 b Training with redundant feature removed. 
# Your code here:



