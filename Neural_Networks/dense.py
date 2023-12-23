import numpy as np
from layer import Layer

class Dense(Layer):
    def __init__(self, input_size, output_size):
        #Random Initialization of weights and bias
        self.weights = np.random.randn(output_size, input_size)
        self.bias = np.random.randn(output_size, 1)
    
    def forwardprop(self, input):
        self.input = input
        #Computing y = w.x + b
        return np.dot(self.weights, self.input) + self.bias 
    
    def backprop(self, gradient, lr):
        weight_gradient = np.dot(gradient, self.input.T)
        self.weights -= lr * weight_gradient
        self.bias -= lr * gradient
        return np.dot(self.weights.T, gradient)
