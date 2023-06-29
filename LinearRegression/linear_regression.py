'''
Linear Regressio

-> y = wx + b
-> Cost function : MSE
'''
import numpy as np

class Linear_Regression:
    def __init__(self, lr=0.001, n_iters = 1000):
        self.lr = lr
        self.n_iters = n_iters
        self.weights = None
        self.bias = None
    
    def fit(self, X, y):
        #Initalizing for gradient descent.
        n_samples, n_feautres = X.shape
        self.weights = np.zeros(n_feautres)
        self.bias = 0

        #Gradient descent
        for _ in range(self.n_iters):
            y_hat = np.dot(X, self.weights) + self.bias

            #Calculating the derivatives.
            dw = (1/n_samples) * np.dot(X.T, (y_hat - y))
            db = (1/n_samples) * np.sum(y_hat - y)

            #Updating weights
            self.weights -= self.lr * dw
            self.bias -= self.lr * db

    def predict(self, X_test):
        y_hat = np.dot(X_test, self.weights) + self.bias

        return y_hat