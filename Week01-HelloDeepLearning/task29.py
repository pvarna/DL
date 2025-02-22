import numpy as np
import matplotlib.pyplot as plt


def create_dataset(n):
    return [(x, 2 * x) for x in range(n)]


def initialize_weights(x, y):
    return np.random.uniform(x, y)


def main():
    print(create_dataset(4))
    print(initialize_weights(0, 100))
    print(initialize_weights(0, 10))

    # model has 1 parameter (w)
    # accepts a number (x) from the user
    # returns (x * w)


if __name__ == '__main__':
    main()
