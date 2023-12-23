from dense import Dense
from activation_functions import Tanh, Relu
from losses import mse, mse_prime
import numpy as np

X = np.reshape([[0, 0], [0, 1], [1, 0], [1, 1]], (4, 2, 1))
Y = np.reshape([[0], [1], [1], [0]], (4, 1, 1))

model = [
    Dense(2,3),
    Tanh(),
    Dense(2,3),
    Relu(),
    Dense(3,1),
    Tanh()
]

epochs = 1000
lr = 0.01
#Training the model
for e in range(epochs):
    error = 0
    for x, y in zip(X,Y):
        output = x
        #Forwardprop
        for layer in model:
            output = layer.forwardprop(output)
        
        #Error
        error = mse(y, output)

        #Backprop
        grad = mse_prime(y , output)
        for layer in reversed(model):
            grad = layer.backprop(grad, lr)
    error /= len(x)
    print('%d/%d, error = %f' % (e + 1, epochs, error))