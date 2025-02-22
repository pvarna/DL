import numpy as np
import matplotlib.pyplot as plt

np.random.seed(123)


def make_one_move(current_step):
    dice = np.random.randint(1, 7)

    steps_up = 0
    if dice in [1, 2]:
        steps_up = -1
    elif dice in [3, 4, 5]:
        steps_up = 1
    else:
        steps_up = np.random.randint(1, 7)

    return max(0, current_step + steps_up)


def main():
    walk = [0]
    for _ in range(100):
        walk.append(make_one_move(walk[-1]))

    print(walk)


if __name__ == '__main__':
    main()
