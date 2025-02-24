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


def model():
    w1 = initialize_weights(0, 1)
    w2 = initialize_weights(0, 1)
    return w1, w2


def calculate_loss(w1, w2, dataset):
    losses = [(y - (x1 * w1 + x2 * w2))**2 for x1, x2, y in dataset]
    return np.mean(losses)


def approximate_partial_derivatives(w1, w2, dataset):
    partial_derivative1 = (calculate_loss(w1 + EPS, w2, dataset) -
                           calculate_loss(w1, w2, dataset)) / EPS
    partial_derivative2 = (calculate_loss(w1, w2 + EPS, dataset) -
                           calculate_loss(w1, w2, dataset)) / EPS
    return partial_derivative1, partial_derivative2


def train_model(w1, w2, dataset):
    loss = calculate_loss(w1, w2, dataset)
    print(f"Initial loss: {loss}, initial w1: {w1}, initial w2: {w2}")

    current_loss = loss
    for _ in range(EPOCHS):
        L1, L2 = approximate_partial_derivatives(w1, w2, dataset)
        w1 -= LEARNING_RATE * L1
        w2 -= LEARNING_RATE * L2
        current_loss = calculate_loss(w1, w2, dataset)

    print(f"Final loss: {current_loss}, final w1: {w1}, final w2: {w2}")


def main():
    # model has 2 parameters (w1, w2)
    # accepts 2 numbers (x1, x2) from the user
    # returns (x1 * w1 + x2 * w2) (linear combination)

    w1_and, w2_and = model()
    w1_or, w2_or = model()

    print("Model for AND:")
    train_model(w1_and, w2_and, create_dataset_and())
    print("Model for OR:")
    train_model(w1_or, w2_or, create_dataset_or())


if __name__ == '__main__':
    main()
