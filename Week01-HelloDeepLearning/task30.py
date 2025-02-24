import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)


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


def main():
    dataset = create_dataset(6)

    w = model()
    loss = calculate_loss(w, dataset)
    print(f"MSE: {loss}")

    # experiment_loss1 = calculate_loss(w + 0.001 * 2, dataset)
    # experiment_loss2 = calculate_loss(w + 0.001, dataset)
    # experiment_loss3 = calculate_loss(w - 0.001, dataset)
    # experiment_loss4 = calculate_loss(w - 0.001 * 2, dataset)
    # print(
    #     f"{experiment_loss1}, {experiment_loss2}, {experiment_loss3}, {experiment_loss4}"
    # )
    # with increasing the parameter 'w', the loss also increases
    # with decreasing the parameter 'w', the loss also decreases


if __name__ == '__main__':
    main()
