import numpy as np
import matplotlib.pyplot as plt
import torch


def function(x):
    return x**4 + x**3 - 5 * x**2


def optimize_and_plot(lr, momentum):
    if lr > 0.05:
        raise ValueError('Choose a learning <= 0.05')
    x = torch.tensor(2.0, requires_grad=True)
    buffer = torch.zeros_like(x.data)
    values = []
    for i in range(20):

        y = function(x)
        values.append((x.clone(), y.clone()))
        y.backward()

        d_p = x.grad.data
        if momentum != 0:
            buffer.mul_(momentum).add_(d_p)
            d_p = buffer

        x.data.add_(d_p, alpha=-lr)
        x.grad.zero_()

    x = np.arange(-3, 2, 0.001)
    y = function(x)

    plt.figure(figsize=(10, 5))
    plt.plot([v[0].item() for v in values], [v[1].item() for v in values],
             'r-X',
             linewidth=2,
             markersize=7)
    for i in range(20):
        plt.text(values[i][0].item() + 0.1,
                 values[i][1].item(),
                 f'step {i}',
                 fontdict={'color': 'r'})
    plt.plot(x, y, linewidth=2)
    plt.grid()
    plt.tick_params(axis='both', which='major', labelsize=12)
    plt.legend(['Optimizer steps', 'Square function'])
    plt.tight_layout()
    plt.show()


def main():
    optimize_and_plot(0.025, 0.85)


if __name__ == '__main__':
    main()
