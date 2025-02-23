import numpy as np
import matplotlib.pyplot as plt


def initialize_weights(x, y):
    return np.random.uniform(x, y)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


class Xor:
    dataset = [(0, 0, 0), (0, 1, 1), (1, 0, 1), (1, 1, 0)]
    eps = 0.001
    # learning_rate = 0.001 --> it gets worse if I use it

    def __init__(self, w11, w12, b1, w13, w14, b2, w21, w22, b3):
        self.w11 = w11
        self.w12 = w12
        self.b1 = b1
        self.w13 = w13
        self.w14 = w14
        self.b2 = b2
        self.w21 = w21
        self.w22 = w22
        self.b3 = b3

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def calculate_loss(self, w11, w12, b1, w13, w14, b2, w21, w22, b3):
        losses = []
        for x1, x2, y in self.dataset:
            or_output = self.sigmoid(w11 * x1 + w12 * x2 + b1)
            nand_output = self.sigmoid(w13 * x1 + w14 * x2 + b2)
            and_output = self.sigmoid(w21 * or_output + w22 * nand_output + b3)

            losses.append((y - and_output)**2)

        return np.mean(losses)

    def approximate_partial_derivatives(self, w11, w12, b1, w13, w14, b2, w21,
                                        w22, b3):
        params = [w11, w12, b1, w13, w14, b2, w21, w22, b3]
        partial_derivatives = []

        for i, _ in enumerate(params):
            params_plus_eps = params[:]
            params_plus_eps[i] += self.eps

            loss_plus_eps = self.calculate_loss(*params_plus_eps)
            loss = self.calculate_loss(*params)

            partial_derivative = (loss_plus_eps - loss) / self.eps
            partial_derivatives.append(partial_derivative)

        return partial_derivatives

    def train(self, epochs):
        params = [self.w11, self.w12, self.b1, self.w13, self.w14, self.b2, self.w21, self.w22, self.b3]
        initial_loss = self.calculate_loss(*params)
        print(f"Initial loss: {initial_loss}")

        losses = [initial_loss]
        current_loss = initial_loss
        for _ in range(epochs):
            partial_derivatives = self.approximate_partial_derivatives(*params)
            params = [p - d for p, d in zip(params, partial_derivatives)]
            current_loss = self.calculate_loss(*params)
            losses.append(current_loss)

        self.w11, self.w12, self.b1, self.w13, self.w14, self.b2, self.w21, self.w22, self.b3 = params

        print(f"Final loss: {losses[-1]}")

    def forward(self, x1, x2):
        or_output = self.sigmoid(self.w11 * x1 + self.w12 * x2 + self.b1)
        nand_output = self.sigmoid(self.w13 * x1 + self.w14 * x2 + self.b2)
        and_output = self.sigmoid(self.w21 * or_output + self.w22 * nand_output + self.b3)

        return and_output
    
    def test(self):
        for x1, x2, y in self.dataset:
            output = self.forward(x1, x2)
            print(f"XOR({x1}, {x2}) = {output} (Expected: {y})")


def main():
    xor = Xor(*[initialize_weights(0, 1) for _ in range(9)])

    xor.train(100000)
    xor.test()


if __name__ == '__main__':
    main()
