from dotenv import load_dotenv
import os
import numpy as np
import matplotlib.pyplot as plt
from data import RandomDataset, Dataset
from models import LinearRegression
import quad

load_dotenv()

DATASET_SET_SIZE = int(os.environ["DATASET_SET_SIZE"])
DATASET_SPARCE_RATIO = int(os.environ["DATASET_SPARCE_RATIO"])
DATASET_X_LIM = int(os.environ["DATASET_X_LIM"])

# Create a random dataset
random_dataset = RandomDataset(DATASET_SET_SIZE, DATASET_SPARCE_RATIO, DATASET_X_LIM)
X, y = random_dataset.generate()

# Preprocess the dataset
dataset = Dataset(X, y, degree=3)
Xr, y = dataset.preprocess()

# Initialize theta and linear regression model
theta = np.random.rand(Xr.shape[1], 1)
linear_reg = LinearRegression(quad.cost, quad.grad, alpha=0.00000000025, max_iter=200)

# Fit the model
tf, costs = linear_reg.fit(Xr, y, theta)

