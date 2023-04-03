import numpy as np

def cost(X, y, t):
    m = len(y)
    return np.sum((np.dot(X, t) - y)**2) / (2 * m)

def grad(X, y, t):
    m = len(y)
    return np.dot(X.T, (np.dot(X, t) - y)) / m
