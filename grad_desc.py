import os
import numpy as np
import matplotlib.pyplot as plt
from dotenv import load_dotenv
from quad import cost, grad
from linreg import linear_regression

load_dotenv()

DATASET_SET_SIZE = int(os.environ["DATASET_SET_SIZE"])
DATASET_SPARCE_RATIO = int(os.environ["DATASET_SPARCE_RATIO"])
DATASET_X_LIM = int(os.environ["DATASET_X_LIM"])

# Obtener un dataset aleatorio
def get_random_dataset():
    X = np.linspace(0, DATASET_X_LIM, DATASET_SET_SIZE).reshape((DATASET_SET_SIZE, 1))
    Xr = np.hstack((np.ones((DATASET_SET_SIZE, 1)), X))
    y = 3 + 2 * X + np.random.rand(DATASET_SET_SIZE, 1) * DATASET_SPARCE_RATIO
    return X, Xr, y

X, Xr, y = get_random_dataset()

to = np.random.rand(Xr.shape[1], 1) # Theta inicial.

tf, costs = linear_regression(Xr, y, to, cost, grad, a=0.025, n=20) # Theta final.

print("Tf: ", tf)

xm = np.array([[0], [DATASET_X_LIM]])
xmr = np.hstack((np.ones((2, 1)), xm))
ym = xmr @ tf

plt.plot(Xr[:, 1], y, "ro")
plt.plot(xm, ym)
plt.show()

plt.plot(costs)
plt.show()
