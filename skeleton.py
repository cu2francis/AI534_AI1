import numpy as np
import pandas as pd
import plotly.express as px


def load_data(path):
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
    modified_data = ""

    return modified_data


def loss_function(training_values, target, weights):
    predictions = np.dot(training_values, weights.T)
    return (1 / len(target)) * np.sum((predictions - target) ** 2)


def batch_gradient_descent(training_values, target, weights, learning_rate, iterations):
    loss_history = [0] * iterations
    for i in range(iterations):
        prediction = np.dot(training_values, weights.T)
        weights = weights - (learning_rate / len(target)) * np.dot(prediction - target, training_values)
        loss_history[i] = loss_function(training_values, target, weights)
    return weights, loss_history


# Trains a linear model on the provided data and labels, using the supplied learning rate.
# weights should store the per-feature weights of the learned linear regression.
# losses should store the sequence of MSE losses for each epoch of training, which you will then plot.
def gd_train(data, labels, lr, iterations):
    training_values = data[labels]
    target = data['price']
    weights = np.zeros(training_values.shape[1])
    print("Start-Weight: " + str(loss_function(training_values, target, weights)))
    training_values = (training_values - training_values.min()) / (training_values.max() - training_values.min())
    batch_weight, batch_loss_history = batch_gradient_descent(training_values, target, weights, lr, iterations)

    print("End-Theta: " + str(loss_function(training_values, target, batch_weight)))

    return batch_weight, batch_loss_history


# Generates and saves plots of the training loss curves. Note that you can interpret losses as a matrix
# containing the losses of multiple training runs and then put multiple loss curves in a single plot.
def plot_losses(losses):
    fig = px.line(losses, x=range(len(losses)), y=losses, labels={'x': 'no. of iterations', 'y': 'cost function'})
    fig.show()
    return
