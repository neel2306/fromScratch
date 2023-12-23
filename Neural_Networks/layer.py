import numpy as np 

#Creating a base layer.
class Layer:
    def __init__(self):
        self.input = None
        self.output = None
    
    def forwardprop(self, input):
        pass

    def backprop(self, gradient, lr):
        pass

    