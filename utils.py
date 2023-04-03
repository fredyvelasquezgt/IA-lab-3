import numpy as np

class PolynomialFeatures:
    def __init__(self, degree=2):
        self.degree = degree
        
    def transform(self, X):
        if self.degree <= 1:
            return X
        
        X_poly = np.hstack([X**(i+1) for i in range(self.degree)])
        return np.hstack([X, X_poly])
