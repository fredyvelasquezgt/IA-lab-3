import numpy as np

def cost(X, y, t):
    return np.sum((X @ t - y) ** 2) / len(y)

def grad(X, y, t):
    return 2 * X.T @ (X @ t - y) / len(y)
