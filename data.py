import os
import numpy as np
from dotenv import load_dotenv
from utils import PolynomialFeatures

load_dotenv()

class RandomDataset:
    def __init__(self, size, sparsity, x_lim):
        self.size = size
        self.sparsity = sparsity
        self.x_lim = x_lim
        
    def generate(self):
        X = np.linspace(0, self.x_lim, self.size).reshape((self.size, 1))
        y = 3 + 2 * X + np.random.rand(self.size, 1) * self.sparsity
        
        return X, y

class Dataset:
    def __init__(self, X, y, degree=2):
        self.X = X
        self.y = y
        self.features = PolynomialFeatures(degree)
        
    def preprocess(self):
        X_poly = self.features.transform(self.X)
        Xr = np.hstack([np.ones((X_poly.shape[0], 1)), X_poly])
        return Xr, self.y
