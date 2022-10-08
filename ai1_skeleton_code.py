import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# constants
train = "IA1_train.csv"
val = "IA1_dev.csv"
convergence_threshold = 0.0005


# Loads a data file from a provided file location.
def load_data(path):
    # Your code here:
    return pd.read_csv(path)


def renovation(year, year_built, year_renovated):
    age_since_renovated = [0.0] * len(year)
    for index, value in enumerate(year_renovated):
        if value == 0:
            age_since_renovated[index] = year[index] - year_built[index]
        else:
            age_since_renovated[index] = year[index] - year_renovated[index]
    return age_since_renovated


# Implements dataset preprocessing, with boolean options to either normalize the data or not, 
# and to either drop the sqrt_living15 column or not.
#
# Note that you will call this function multiple times to generate dataset versions that are
# / aren't normalized, or versions that have / lack sqrt_living15.
def preprocess_data(data, normalize, drop_sqft_living15):
    data = data.drop(['id'], axis=1)
    data[['month', 'day', 'year']] = data['date'].str.split('/', expand=True)
    data = data.drop(['date'], axis=1)
    data['dummy'] = 1.0
    data = data.apply(pd.to_numeric, errors='coerce')
    data['age_since_renovated'] = renovation(data['year'], data['yr_built'], data['yr_renovated'])
    data = data.drop(['yr_renovated'], axis=1)
    if normalize:
        target = data['price']
        waterfront = data['waterfront']
        data = data.drop(['price'], axis=1)
        data = data.drop(['waterfront'], axis=1)
        data = (data - data.min()) / (data.max() - data.min())
        data['price'] = target
        data['waterfront'] = waterfront

    if drop_sqft_living15:
        data = data.drop(['sqft_living15'], axis=1)
    return data


# Implements the feature engineering required for part 4. Quite similar to preprocess_data.
# Expand the arguments of this function however you like to control which feature modification
# approaches are / aren't active.
def modify_features(data):
    # Your code here:

    return data


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
            if loss_history[i - 1] < loss_history[i]:
                print("\tDiverged")
                break
            if (loss_history[i - 1] - loss_history[i]) < convergence_threshold:
                print("\tConverged")
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
    batch_weight, batch_loss_history = batch_gradient_descent(training_values, target, weights, lr, iterations)
    print("\tFinal MSE:\t\t\t" + str(mean_square_error(training_values, target, batch_weight)))

    return batch_weight, batch_loss_history, start_weight


# Generates and saves plots of the training loss curves. Note that you can interpret losses as a matrix
# containing the losses of multiple training runs and then put multiple loss curves in a single plot.
def plot_losses(losses, labels, y_max):
    fig = plt.figure(figsize=(10, 6))
    ax = plt.subplot(1, 1, 1)
    ax.set_ylim([0, y_max])
    for index, y in enumerate(losses):
        ax.plot(range(len(y)), y, linewidth=0.25, label="LR " + str(10 ** (-labels[index])))
    fig.legend()
    fig.show()
    return


def validate_weight(weights, validate_data):
    target = validate_data['price']
    value = validate_data.drop(['price'], axis=1)
    value = value.drop(['dummy'], axis=1)
    mse = mean_square_error(value, target, weights)
    print("\tValidated MSE:\t\t" + str(mse))


def test_learning_rates(lrs, data, labels, validate):
    multi_losses = []
    maximum = 100000
    for lr in lrs:
        learning_rate = 10 ** (-lr)
        print("Learning Rate " + str(learning_rate) + " =============")
        weight_batch, loses, weight = gd_train(data, labels, learning_rate, 4000)
        validate_weight(weight_batch, validate)
        if weight < maximum:
            maximum = weight
        multi_losses.append(loses)
    return multi_losses, maximum


def task1a():
    data = load_data(train)
    data = preprocess_data(data, True, False)

    validate = load_data(val)
    validate = preprocess_data(validate, True, False)
    labels = ['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors', 'view',
              'condition', 'grade', 'sqft_above', 'sqft_basement', 'yr_built',
              'zipcode', 'lat', 'long', 'sqft_living15', 'sqft_lot15', 'month', 'day',
              'year', 'age_since_renovated', 'waterfront']
    lrs = [0, 1, 2, 3, 4]

    multi_losses, maximum = test_learning_rates(lrs, data, labels, validate)
    plot_losses(multi_losses, lrs, maximum)


print("=====================================")
print("=== TASK 1")
print("=====================================")
task1a()


def task_2a():
    data = load_data(train)
    preprocess_data(data, normalize=True, drop_sqft_living15=False)
    data = preprocess_data(data, normalize=False, drop_sqft_living15=False)

    labels = ['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors', 'view',
              'condition', 'grade', 'sqft_above', 'sqft_basement', 'yr_built',
              'zipcode', 'lat', 'long', 'sqft_living15', 'sqft_lot15', 'month', 'day',
              'year', 'age_since_renovated', 'waterfront']

    lrs = [12, 11, 10, 5, 1]

    for lr in lrs:
        learning_rate = 10 ** (-lr)
        print("Learning Rate " + str(learning_rate) + " =============")
        gd_train(data, labels, learning_rate, 4000)

    learning_rate = 10 ** (-11)
    print("Best Learning Rate is " + str(learning_rate) + " =============")
    weight_batch, loses, weight = gd_train(data, labels, learning_rate, 4000)
    print("Learned Feature Weights " + str(weight_batch))


print("=====================================")
print("=== TASK 2a")
print("=====================================")
task_2a()


def task_2b():
    data = load_data(train)
    data = preprocess_data(data, True, True)

    validate = load_data(val)
    validate = preprocess_data(validate, True, True)

    labels = ['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors', 'view',
              'condition', 'grade', 'sqft_above', 'sqft_basement', 'yr_built',
              'zipcode', 'lat', 'long', 'sqft_lot15', 'month', 'day',
              'year', 'age_since_renovated', 'waterfront']

    lrs = [7, 6, 5, 1]
    test_learning_rates(lrs, data, labels, validate)


print("=====================================")
print("=== TASK 2b")
print("=====================================")
task_2b()
print("=====================================")
print("=== DONE")
