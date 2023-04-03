import numpy as np

class LinearRegression:
    def __init__(self, cost, grad, alpha=0.1, max_iter=1000):
        self.cost = cost
        self.grad = grad
        self.alpha = alpha
        self.max_iter = max_iter
        
    def fit(self, X, y, theta):
        costs = []
        
        for i in range(self.max_iter):
            nabla = self.grad(X, y, theta)
            theta -= self.alpha * nabla
            costs.append(self.cost(X, y, theta))
            
        return theta, costs
