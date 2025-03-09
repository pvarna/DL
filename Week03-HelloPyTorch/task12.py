import random


def main():
    learning_rates = []
    momentums = []

    for _ in range(10):
        learning_rate = random.uniform(0.0001, 0.01)
        learning_rates.append(learning_rate)

        momentum = random.uniform(0.85, 0.99)
        momentums.append(momentum)

    hyperparameter_pairs = list(zip(learning_rates, momentums))
    print(hyperparameter_pairs)


if __name__ == '__main__':
    main()
