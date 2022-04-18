from dataclasses import dataclass
import numpy as np

epsilon = 1e-8

sigmoid = lambda x: np.divide(1, 1 + np.exp(-x))

cross_entropy = lambda target, prediction: np.multiply(target, np.log(prediction + epsilon)) + np.multiply(1 - target,
                                                                                                           np.log(
                                                                                                               1 - prediction + epsilon))

MSE = lambda target, prediction: np.sum(np.square(target - prediction)) / 2

relu = lambda x: np.maximum(0, x)

softmax = lambda x: np.exp(x) / np.sum(np.exp(x))

RBF = lambda x: np.exp(-np.sum(-np.square(x)))


@dataclass
class Node:
    weights: np.ndarray
    bias: float

    # initialize the node with custom weights and bias
    # Cannot be initialized with default values
    def __init__(self, weights, bias, output=False, activation=None):
        self.error = None
        self.output = output
        self.weights = np.array(weights)
        self.bias = bias
        if activation:
            # Custom activation function
            self.activation = activation
        elif output:
            # Output layer
            self.activation = sigmoid
        else:
            # Hidden layer
            # self.activation = RBF
            self.activation = softmax

    def __str__(self):
        return f"Weights: {self.weights}, Bias: {self.bias}"

    # Forward propagation, using native python functions, slower than numpy
    # DEPRECATED
    def feed_forward(self, inputs):
        if type(inputs) is not np.ndarray:
            inputs = np.array(inputs)
        if inputs.size != self.weights.size:
            raise ValueError("Inputs and weights must have the same length")
        # return sigmoid(np.dot(inputs, self.weights) + self.bias)
        return self.activation(np.dot(inputs, self.weights) + self.bias)

    # Forward propagation, using numpy, faster than native python functions
    def feed_forward_matrices(self, inputs: np.ndarray):
        if type(inputs) is not np.ndarray:
            inputs = np.array(inputs)
        inputs = np.squeeze(inputs)
        if inputs.size != self.weights.size:
            raise ValueError("Inputs and weights must have the same length")
        z = np.dot(inputs, self.weights) + self.bias
        return self.activation(z.astype(float))

    def get_weights(self):
        return self.weights

    def set_weights(self, values):
        if type(values) is list:
            values = np.array(values)
        self.weights = values

    def get_bias(self):
        return self.bias

    def set_bias(self, param):
        if type(param) is list:
            param = np.array(param)
        self.bias = param

    # Experimental cost function, not used yet
    def cost(self, inputs, targets):
        if type(inputs) is not np.ndarray:
            inputs = np.array(inputs)
        if type(targets) is list:
            targets = np.array(targets)
        return cross_entropy(targets, self.feed_forward(inputs))

    # learn single using error and learning rate
    def learn_single(self, error, learning_rate):
        if self.output:
            self.weights = self.weights - np.multiply(learning_rate, error)
            self.bias = self.bias - np.multiply(learning_rate, error)

    def back_propagate(self, error, learning_rate):
        if self.output:
            self.weights = self.weights - np.multiply(learning_rate, error)
            self.bias = self.bias - np.multiply(learning_rate, error)
        else:
            # Calculate the error for the next layer
            self.error = np.multiply(np.dot(error, self.weights.T), self.activation(self.bias))
            self.weights = self.weights - np.multiply(learning_rate, np.dot(error, self.weights))
            self.bias = self.bias - np.multiply(learning_rate, error)

    def get_error(self):
        return self.error
