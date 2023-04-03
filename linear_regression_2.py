import numpy as np

def linear_regressionv(X, y, t, cost, grad, a=0.1, n=1000, on_step=None):
    costs = []

    for i in range(n):
        nabla = grad(X, y, t)
        norm_nabla = np.linalg.norm(nabla)

        t -= a * nabla / norm_nabla

        costs.append(cost(X, y, t))

        if on_step:
            on_step(t)

    return t, costs

  #en lugar de calcular la norma del vector de gradiente utilizando una función lambda personalizada, se utiliza la función np.linalg.norm() de NumPy para calcular la norma euclidiana del vector nabla. Luego, se actualiza t dividiendo el vector gradiente nabla por su norma y multiplicándolo por a.
