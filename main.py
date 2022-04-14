from Network import Network
import time

if __name__ == '__main__':
    sizes = [1]
    network = Network(sizes)
    start_time = time.perf_counter()
    # print(network.get_nodes())
    print(network.feed_forward_matrices([0.5]))
    print(time.perf_counter() - start_time)
