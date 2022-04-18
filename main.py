import random
from Network import Network
import time


def create_test_cases(function, num_test_cases, num_inputs, num_outputs):
    test_cases = []
    for i in range(num_test_cases):
        inputs = []
        for j in range(num_inputs):
            inputs.append(random.random())
        outputs = function(inputs)
        test_cases.append((inputs, outputs))
    return test_cases


if __name__ == '__main__':
    sizes = [2, 3, 2]
    network = Network(sizes)
    start_time = time.perf_counter()
    # print(network.get_nodes())
    test_data = create_test_cases(lambda x: x * 2, 1, 2, 2)
    # print(test_data)
    # print(network.feed_forward_matrices([0.1]))
    # print(network.get_nodes())
    network.learn(test_data, 0.1)
    # print(network.feed_forward_matrices([0.1]))
    # print(time.perf_counter() - start_time)
