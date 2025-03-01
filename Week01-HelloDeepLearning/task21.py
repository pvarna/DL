import numpy as np
import matplotlib.pyplot as plt

np.random.seed(123)


def main():
    random_float = np.random.rand()
    random_integer1 = np.random.randint(1, 7)
    random_integer2 = np.random.randint(1, 7)

    print(f"Random float: {random_float}")
    print(f"Random integer 1: {random_integer1}")
    print(f"Random integer 2: {random_integer2}")

    current_step = 50
    print(f"Before throw step = {current_step}")

    steps_up = 0
    if random_integer1 in {1, 2}:
        steps_up = -1
    elif random_integer1 in {3, 4, 5}:
        steps_up = 1
    else:
        steps_up = np.random.randint(1, 7)

    current_step += steps_up
    print(f"After throw dice = {random_integer1}")
    print(f"After throw step = {current_step}")


if __name__ == '__main__':
    main()
