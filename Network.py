import numpy as np

from Layer import Layer

epsilon = 1e-8


def MSE(target, prediction):
    return np.mean((target - prediction) ** 2)


def cross_entropy(target, prediction):
    return -np.mean(target * np.log(prediction + epsilon) + (1 - target) * np.log(1 - prediction + epsilon))


class Network:
    def __init__(self, sizes):
        self.num_layers = len(sizes)
        self.layers = []
        for i in range(len(sizes)):
            if i == 0:
                self.layers.append(Layer(sizes[i], sizes[i]))
            else:
                self.layers.append(Layer(sizes[i], sizes[i - 1]))

    def feed_forward(self, inputs):
        passthrough = inputs
        for layer in self.layers:
            temp = layer.feed_forward(passthrough)
            passthrough = temp
        return passthrough

    def feed_forward_matrices(self, inputs: np.ndarray) -> np.ndarray:
        if type(inputs) == list:
            inputs = np.array(inputs)
        passthrough = inputs
        for layer in self.layers:
            temp = layer.feed_forward_matrices(passthrough)
            passthrough = temp
        return passthrough

    def cost(self, inputs, targets):
        outputs = self.feed_forward(inputs)
        return cross_entropy(targets, outputs)

    def gradiant(self):


    def back_propagate(self, inputs, targets):
        pass

    def get_weights(self) -> np.ndarray:
        weights = []
        for layer in self.layers:
            weights.append(layer.get_weights())
        return np.array(weights)

    def get_biases(self) -> np.ndarray:
        biases = []
        for layer in self.layers:
            biases.append(layer.get_biases())
        return np.array(biases)

    def set_weights(self, weights: np.ndarray):
        if weights.shape != self.get_weights().shape:
            raise ValueError("The weights must have the same shape as the network's weights.")
        for i in range(len(self.layers)):
            self.layers[i].set_weights(weights[i])

    def get_nodes(self) -> np.ndarray:
        nodes = []
        for layer in self.layers:
            nodes.append(layer.get_nodes())
        return np.array(nodes)
