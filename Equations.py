import numpy as np

# Constants
epsilon = 1e-8

# Sigmoid activation function
sigmoid = lambda x: np.divide(1, 1 + np.exp(-x))

# Derivative of sigmoid
sigmoid_derivative = lambda x: np.multiply(sigmoid(x), 1 - sigmoid(x))

cross_entropy = lambda target, prediction: np.multiply(target, np.log(prediction + epsilon)) + np.multiply(1 - target,
                                                                                                           np.log(
                                                                                                               1 - prediction + epsilon))

# derivative of cross_entropy
cross_entropy_derivative = lambda target, prediction: np.divide(target, prediction + epsilon) - np.divide(1 - target,
                                                                                                           1 - prediction + epsilon)

# Mean Squared Error
MSE = lambda target, prediction: np.sum(np.square(target - prediction)) / 2

# Relu activation function
relu = lambda x: np.maximum(0, x)

# Derivative of relu activation function
relu_derivative = lambda x: np.where(x > 0, 1, 0)

# Softmax activation function
softmax = lambda x: np.exp(x) / np.sum(np.exp(x))

# Derivative of softmax activation function
softmax_derivative = lambda x: np.multiply(softmax(x), 1 - softmax(x))

# RBF activation function
RBF = lambda x: np.exp(-np.sum(-np.square(x)))

# Derivative of RBF activation function
RBF_derivative = lambda x: -2 * x * RBF(x)