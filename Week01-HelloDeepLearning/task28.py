import numpy as np
import matplotlib.pyplot as plt

np.random.seed(123)

def make_one_move(current_step):
    clumsyness = np.random.rand()
    dice = np.random.randint(1, 7)

    steps_up = 0
    if dice in [1, 2]:
        steps_up = -1
    elif dice in [3, 4, 5]:
        steps_up = 1
    else:
        steps_up = np.random.randint(1, 7)


    return 0 if clumsyness <= 0.005 else max(0, steps_up + current_step)

def make_one_walk():
    walk = [0]
    for _ in range(100):
        walk.append(make_one_move(walk[-1]))

    return walk

def main():
    all_walks = [make_one_walk() for _ in range(500)]
    
    np_all_walks = np.array(all_walks)
    last_steps = np_all_walks[:, -1]

    plt.hist(last_steps)
    plt.show()

    # percent = (np.sum(last_steps > 60) / last_steps.size) * 100
    # print(percent)
    # 59.4%


if __name__ == '__main__':
    main()
