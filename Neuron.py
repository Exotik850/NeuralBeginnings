from dataclasses import dataclass
import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))
relu = lambda x: np.maximum(0, x)

@dataclass
class Node:
    weights: np.ndarray
    bias: float

    def __init__(self, weights, bias):
        self.weights = np.array(weights)
        self.bias = bias

    def __str__(self):
        return f"Weights: {self.weights}, Bias: {self.bias}"

    def feed_forward(self, inputs):
        if type(inputs) is list:
            inputs = np.array(inputs)
        if inputs.size != self.weights.size:
            raise ValueError("Inputs and weights must have the same length")
        # return sigmoid(np.dot(inputs, self.weights) + self.bias)
        return sigmoid(np.dot(inputs, self.weights) + self.bias)

    def feed_forward_matrices(self, inputs : np.ndarray):
        if inputs.size != self.weights.size:
            raise ValueError("Inputs and weights must have the same length")
        return sigmoid(np.dot(inputs, self.weights) + self.bias)
    def activation(self, num):
        return 

    def get_weights(self):
        return self.weights

    def set_weights(self, weights):
        self.weights = weights

    def get_bias(self):
        return self.bias

    def update_weights(self, values):
        if type(values) is list:
            values = np.array(values)
        self.weights = values

