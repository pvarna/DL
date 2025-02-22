import numpy as np
import matplotlib.pyplot as plt

EPS = 0.001
LEARNING_RATE = 0.001
EPOCHS = 100000


def create_dataset_and():
    return [(0, 0, 0), (0, 1, 0), (1, 0, 0), (1, 1, 1)]


def create_dataset_or():
    return [(0, 0, 0), (0, 1, 1), (1, 0, 1), (1, 1, 1)]


def initialize_weights(x, y):
    return np.random.uniform(x, y)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def model():
    w1 = initialize_weights(0, 1)
    w2 = initialize_weights(0, 1)
    b = initialize_weights(0, 1)
    return w1, w2, b


def calculate_loss(w1, w2, bias, dataset):
    losses = [(y - sigmoid(x1 * w1 + x2 * w2 + bias))**2
              for x1, x2, y in dataset]
    return np.mean(losses)


def approximate_partial_derivatives(w1, w2, bias, dataset):
    partial_derivative1 = (calculate_loss(w1 + EPS, w2, bias, dataset) -
                           calculate_loss(w1, w2, bias, dataset)) / EPS
    partial_derivative2 = (calculate_loss(w1, w2 + EPS, bias, dataset) -
                           calculate_loss(w1, w2, bias, dataset)) / EPS
    partial_derivative3 = (calculate_loss(w1, w2, bias + EPS, dataset) -
                           calculate_loss(w1, w2, bias, dataset)) / EPS
    return partial_derivative1, partial_derivative2, partial_derivative3


def train_model(w1, w2, bias, dataset):
    loss = calculate_loss(w1, w2, bias, dataset)
    print(
        f"Initial loss: {loss}, initial w1: {w1}, initial w2: {w2}, initial bias: {bias}"
    )

    current_loss = loss
    for _ in range(EPOCHS):
        L1, L2, L3 = approximate_partial_derivatives(w1, w2, bias, dataset)
        w1 -= LEARNING_RATE * L1
        w2 -= LEARNING_RATE * L2
        bias -= LEARNING_RATE * L3
        current_loss = calculate_loss(w1, w2, bias, dataset)

    print(
        f"Final loss: {current_loss}, final w1: {w1}, final w2: {w2}, final bias: {bias}"
    )


def main():
    w1_and, w2_and, bias_and = model()
    w1_or, w2_or, bias_or = model()

    print("Model for AND:")
    train_model(w1_and, w2_and, bias_and, create_dataset_and())
    print("Model for OR:")
    train_model(w1_or, w2_or, bias_or, create_dataset_or())

    # the sigmoid function changes w1 and w1 to be bigger than 1


if __name__ == '__main__':
    main()
