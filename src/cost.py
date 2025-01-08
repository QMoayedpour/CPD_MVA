import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


class Cost:
    def __init__(self):
        pass

    def __call__(self, l):
        return 0

class LCost(Cost):
    def __init__(self, c):
        self.c = c
    
    def __call__(self, l):
        return self.c * l

class LogCost(Cost):
    def __init__(self, c):
        self.c = c
    
    def __call__(self, l):
        return self.c * float(np.log(l)) if l>0 else 0

class SquareCost(Cost):
    def __init__(self, c):
        self.c = c
    
    def __call__(self, l):
        return self.c * l**2


def cost_L2(signal):
    return np.linalg.norm(signal-np.mean(signal))**2


def cost_mse(signal):
    """
    Calcule la RMSE entre le signal et la régression linéaire de ce signal.
    
    :param signal: Signal réel (array numpy ou pandas Series)
    :return: RMSE entre signal et signal prédit
    """

    t = np.arange(len(signal)).reshape(-1, 1)
    model = LinearRegression()
    model.fit(t, signal)

    signal_hat = model.predict(t)

    return mean_squared_error(signal, signal_hat)
