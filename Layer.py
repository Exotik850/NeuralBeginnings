import random
import numpy as np
from Neuron import Node


class Layer:
    # Initialize the layer with the number of neurons and the number of inputs
    # Cannot be initialized with a zero or negative number of neurons or inputs
    def __init__(self, num_nodes, num_inputs, output=False):
        if num_nodes <= 0 or num_inputs <= 0:
            raise ValueError("Number of nodes and inputs cannot be negative or zero")

        self.output = output
        self.nodes = []
        for i in range(num_nodes):
            temp_bias = random.uniform(-1, 1)
            temp_weights = [random.uniform(-1, 1) for i in range(num_inputs)]
            self.nodes.append(Node(temp_weights, temp_bias, output))
            # np.append(self.nodes, Node(temp_weights, temp_bias))

    # forward propagation using native python, slower than numpy
    # DEPRECATED
    def feed_forward(self, input_values):
        passthrough = input_values
        outputs = []
        for node in self.nodes:
            temp = node.feed_forward(passthrough)
            outputs.append(temp)
        return outputs

    # forward propagation using numpy, faster than native python
    def feed_forward_matrices(self, inputs: np.ndarray):
        if type(inputs) is not np.ndarray:
            inputs = np.array(inputs)
        passthrough = inputs
        outputs = []
        for node in self.nodes:
            temp = node.feed_forward_matrices(passthrough)
            outputs.append(temp)
        return np.array(outputs)

    # learns a single test case using error and learning rate
    def learn_single(self, error, learning_rate):
        for node in self.nodes:
            node.learn_single(error, learning_rate)

    def add_node(self, node):
        self.nodes.append(node)

    def add_nodes(self, nodes):
        for node in nodes:
            self.add_node(node)

    def get_nodes(self):
        return self.nodes

    @property
    def num_nodes(self):
        return len(self.nodes)

    def get_biases(self):
        biases = []
        for node in self.nodes:
            biases.append(node.get_bias())
        return biases

    def get_weights(self):
        weights = []
        for node in self.nodes:
            weights.append(node.get_weights())
        return np.array(weights)

    def set_weights(self, weights):
        for i in range(len(self.nodes)):
            self.nodes[i].set_weights(weights[i])

    def set_biases(self, param):
        for i in range(len(self.nodes)):
            self.nodes[i].set_bias(param[i])
