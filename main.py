import skeleton
import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.datasets import load_boston
from sklearn.metrics import mean_squared_error

path = "/home/sam/Downloads/IA1_train.csv"


def task0():
    data = skeleton.load_data(path)
    data = skeleton.preprocess_data(data, True, True)
    weight, loses = skeleton.gd_train(data, ["bedrooms", "bathrooms", "sqft_living", "sqft_lot", "floors",
                                             "waterfront", "view", "condition", "grade", "sqft_above", "sqft_basement",
                                             "yr_built", "yr_renovated", "zipcode", "lat", "long", "sqft_living15",
                                             "sqft_lot15"], 0.05, 2500)
    skeleton.plot_losses(loses)

def task1():
    data = skeleton.load_data(path)
    data = skeleton.preprocess_data(data, True, False)
    data = skeleton.gd_train(data, "1", 1)


if __name__ == '__main__':
    task0()
    # task1()
