import numpy as np
from Layer import Layer
from Neuron import Node


class Network:
    # initialize the network with the sizes of the layers to be created
    def __init__(self, sizes, weights=None, biases=None):
        self.num_layers = len(sizes)
        self.layers = []
        for i in range(len(sizes)):
            if i == 0:
                self.layers.append(Layer(sizes[i], sizes[i]))
            else:
                self.layers.append(Layer(sizes[i], sizes[i - 1]))
        if weights is not None:
            if weights.shape != self.get_weights().shape:
                raise ValueError("The weights must have the same shape as the network's weights.")
            self.set_weights(weights)
        if biases is not None:
            if biases.shape != self.get_biases().shape:
                raise ValueError("The biases must have the same shape as the network's biases.")
            self.set_biases(biases)

    # learns the network using the given training data and the given learning rate
    def learn(self, training_data, learning_rate: float = .01):
        for single_train in training_data:
            for inputs, target in single_train:
                if type(inputs) is not np.ndarray:
                    inputs = np.array(inputs)
                if type(target) is not np.ndarray:
                    target = np.array(target)
                self.learn_single(inputs, target, learning_rate)

    # learns the network with a single training example
    def learn_single(self, inputs: np.ndarray, target: np.ndarray, learning_rate: float):
        output = self.feed_forward_matrices(inputs)
        output_error = np.array(target * np.log(output), dtype=object)
        error_gradient_output = np.array(output - target, dtype=object)
        if len(self.layers) == 1:
            self.layers[0].learn_single(output_error, learning_rate)
        else:
            for i in range(len(self.layers) - 1, -1, -1):
                if i == len(self.layers) - 1:
                    self.layers[i].learn_single(output_error, learning_rate)
                else:
                    hidden_error = np.multiply(self.layers[i + 1].get_weights().T, output_error)
                    print(hidden_error)
                    hidden_error = np.multiply(hidden_error, np.multiply(output, 1 - output))
                    self.layers[i].learn_single(hidden_error, learning_rate)

    # Initial function for the network, slower because of native python matrix multiplication
    # Deprecated
    def feed_forward(self, inputs):
        passthrough = inputs
        for layer in self.layers:
            temp = layer.feed_forward(passthrough)
            passthrough = temp
        return passthrough

    # Newer function for the propagation, faster because of numpy matrix multiplication
    def feed_forward_matrices(self, inputs: np.ndarray) -> np.ndarray:
        if type(inputs) is not np.ndarray:
            inputs = np.array(inputs)
        passthrough = inputs
        for layer in self.layers:
            temp = layer.feed_forward_matrices(passthrough)
            passthrough = temp
        return passthrough

    # gets the weights of the nodes in the network
    def get_weights(self) -> np.ndarray:
        weights = []
        for layer in self.layers:
            weights.append(layer.get_weights())
        return np.array(weights)

    # gets the biases of the nodes in the network
    def get_biases(self) -> np.ndarray:
        biases = []
        for layer in self.layers:
            biases.append(layer.get_biases())
        return np.array(biases)

    # sets the biases of the nodes in the network
    def set_biases(self, biases: np.ndarray):
        if biases.shape != self.get_biases().shape:
            raise ValueError("The biases must have the same shape as the network's biases.")
        for i in range(len(self.layers)):
            self.layers[i].set_biases(biases[i])

    # sets the weights of the nodes in the network
    def set_weights(self, weights: np.ndarray):
        if weights.shape != self.get_weights().shape:
            raise ValueError("The weights must have the same shape as the network's weights.")
        for i in range(len(self.layers)):
            self.layers[i].set_weights(weights[i])

    # gets the nods in the network
    def get_nodes(self) -> np.ndarray:
        nodes = []
        for layer in self.layers:
            nodes.append(layer.get_nodes())
        return np.array(nodes)
