import numpy as np
from Layer import Layer
from Neuron import epsilon


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
    def learn(self, training_data: list[list], learning_rate: float):
        for single_train in training_data:
            for input, target in single_train:
                self.learn_single(input, target, learning_rate)

    # learns the network with a single training example
    def learn_single(self, inputs: np.ndarray, target: np.ndarray, learning_rate: float):
        output = self.feed_forward_matrices(inputs)
        errors = target - output

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
        if type(inputs) == list:
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
