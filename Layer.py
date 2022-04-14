import random
import numpy as np
from Neuron import Node, epsilon, cross_entropy


def MSE(target, prediction):
    return np.mean(np.power(target - prediction), 2)


class Layer:
    def __init__(self, num_nodes, num_inputs):
        self.nodes = []
        for i in range(num_nodes):
            temp_bias = random.uniform(-1, 1)
            temp_weights = [random.uniform(-1, 1) for i in range(num_inputs)]
            self.nodes.append(Node(temp_weights, temp_bias))
            # np.append(self.nodes, Node(temp_weights, temp_bias))

    def feed_forward(self, input_values):
        passthrough = input_values
        outputs = []
        for node in self.nodes:
            temp = node.feed_forward(passthrough)
            outputs.append(temp)
        return outputs

    def feed_forward_matrices(self, inputs: np.ndarray):
        if type(inputs) == list:
            inputs = np.array(inputs)
        passthrough = inputs
        outputs = []
        for node in self.nodes:
            temp = node.feed_forward_matrices(passthrough)
            outputs.append(temp)
        return outputs

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
        return weights

    def set_weights(self, weights):
        for i in range(len(self.nodes)):
            self.nodes[i].set_weights(weights[i])

    def set_biases(self, param):
        for i in range(len(self.nodes)):
            self.nodes[i].set_bias(param[i])
