import numpy as np

def mse(y_hat, y_pred):
    return np.mean(np.power(y_hat - y_pred, 2))

def mse_prime(y_hat, y_pred):
    return 2*(y_pred - y_hat) / np.size(y_hat)