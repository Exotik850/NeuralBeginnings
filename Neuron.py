from dataclasses import dataclass
import numpy as np

epsilon = 1e-8

sigmoid = lambda x: np.divide(1, 1 + np.exp(-x))

cross_entropy = lambda target, prediction: np.multiply(target, np.log(prediction + epsilon)) + np.multiply(1 - target,
                                                                                                           np.log(
                                                                                                               1 - prediction + epsilon))

MSE = lambda target, prediction: np.sum(np.square(target - prediction)) / 2

relu = lambda x: np.maximum(0, x)

softmax = lambda x: np.exp(x) / np.sum(np.exp(x), axis=1, keepdims=True)


@dataclass
class Node:
    weights: np.ndarray
    bias: float

    def __init__(self, weights, bias, function=sigmoid):
        self.weights = np.array(weights)
        self.bias = bias
        self.activation = function

    def __str__(self):
        return f"Weights: {self.weights}, Bias: {self.bias}"

    def feed_forward(self, inputs):
        if type(inputs) is list:
            inputs = np.array(inputs)
        if inputs.size != self.weights.size:
            raise ValueError("Inputs and weights must have the same length")
        # return sigmoid(np.dot(inputs, self.weights) + self.bias)
        return sigmoid(np.dot(inputs, self.weights) + self.bias)

    def feed_forward_matrices(self, inputs: np.ndarray):
        if inputs.size != self.weights.size:
            raise ValueError("Inputs and weights must have the same length")
        z = np.dot(inputs, self.weights) + self.bias
        return self.activation(z)

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

    def cost(self, inputs, targets):
        if type(inputs) is list:
            inputs = np.array(inputs)
        if type(targets) is list:
            targets = np.array(targets)
        return cross_entropy(targets, self.feed_forward(inputs))

    def cost_gradient(self, inputs, targets):
        if type(inputs) is list:
            inputs = np.array(inputs)
        if type(targets) is list:
            targets = np.array(targets)
        return np.dot(self.weights, self.feed_forward(inputs) - targets)

    def delta_w(self, inputs, targets, learning_rate):
        if type(inputs) is list:
            inputs = np.array(inputs)
        if type(targets) is list:
            targets = np.array(targets)
        return np.multiply(learning_rate, self.cost_gradient(inputs, targets))
