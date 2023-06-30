import numpy as np

class Logistic_Regression:
    def __init__(self, lr=0.01, n_iters=1000):
        self.lr = lr
        self.n_iters = n_iters
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        #Initialising params
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        #Gradient Descent
        for _ in range(self.n_iters):
            z = np.dot(X, self.weights) + self.bias
            y_hat = self.sigmoid(z)

            #Calculating the derivatives.
            dw = (1/n_samples) * np.dot(X.T, (y_hat - y))
            db = (1/n_samples) * np.sum(y_hat - y)

            #Updating weights
            self.weights -= self.lr * dw
            self.bias -= self.lr * db


    def predict(self, X):
        z = np.dot(X, self.weights) + self.bias
        y_hat = self.sigmoid(z)

        #Defining the decision boundary.
        y_hat_class = [1 if i >= 0.5 else 0 for i in y_hat]
        return y_hat_class

    def sigmoid(self, X):
        return 1/(1 + np.exp(-X))
    
    def accuracy(self, y_true, y_pred):
        return np.sum(y_true == y_pred) / len(y_true)
