from activation import Activation
import numpy as np

class Tanh(Activation):
    def __init__(self):
        tanh = lambda x: np.tanh(x)
        #Derivative of tanh function is 1 - tanh(x)^2
        tanh_prime = lambda x: 1 - np.tanh(x) ** 2 
        super().__init__(tanh, tanh_prime)

class Sigmoid(Activation):
    def __init__(self):
        sigmoid = lambda x : 1/(1 + np.exp(-x))
        #Derivative of sigmoid function is sigmoid(x) * (1 - sigmoid)
        sigmoid_prime = lambda x : 1/(1 + np.exp(-x)) * (1 - 1/(1 + np.exp(-x)))
        super().__init__(sigmoid, sigmoid_prime)

class Relu(Activation):
    def __init__(self):
        relu = lambda x : max(0, x)
        #Derivative of relu is 1 if x > 0 else its 0
        relu_prime = lambda x : 1 if x > 0 else 0
        super().__init__(relu, relu_prime)