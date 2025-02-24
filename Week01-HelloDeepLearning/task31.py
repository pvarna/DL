import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

EPS = 0.001


def create_dataset(n):
    return [(x, 2 * x) for x in range(n)]


def initialize_weights(x, y):
    return np.random.uniform(x, y)


def model():
    w = initialize_weights(0, 10)
    return w


def calculate_loss(w, dataset):
    losses = [(y - (x * w))**2 for x, y in dataset]
    return np.mean(losses)


def approximate_derivative(w, dataset):
    return (calculate_loss(w + EPS, dataset) -
            calculate_loss(w, dataset)) / EPS


def main():
    dataset = create_dataset(6)

    w = model()
    loss = calculate_loss(w, dataset)
    print(f"Initial loss: {loss}")

    epochs = 10
    learning_rate = 0.001
    current_loss = loss
    for _ in range(epochs):
        L = approximate_derivative(w, dataset)
        w -= learning_rate * L
        current_loss = calculate_loss(w, dataset)
        print(f"Current loss: {current_loss}")

    print(f"Final loss: {current_loss}")


if __name__ == '__main__':
    main()
