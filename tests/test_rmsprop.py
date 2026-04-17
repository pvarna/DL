import unittest

import torch

from dl_lib.optim import RMSprop


class TestRMSpropStep(unittest.TestCase):

    def test_when_called_once_then_params_are_updated(self):
        # Arrange
        p = torch.tensor([1.0], requires_grad=True)
        p.grad = torch.tensor([2.0])
        optimizer = RMSprop([p], lr=0.1, alpha=0.9)
        expected = 1.0 - 0.1 * 2.0 / (0.4**0.5 + 1e-8)

        # Act
        optimizer.step()
        actual = p.data.item()

        # Assert
        self.assertAlmostEqual(actual, expected)

    def test_when_called_twice_then_squared_average_follows_exponential_moving_average(
            self):
        # Arrange
        p = torch.tensor([0.0], requires_grad=True)
        p.grad = torch.tensor([1.0])
        optimizer = RMSprop([p], lr=0.1, alpha=0.9)
        expected = torch.tensor([0.19])

        # Act
        optimizer.step()

        p.grad = torch.tensor([1.0])
        optimizer.step()

        actual = optimizer.squared_average[0]

        # Assert
        self.assertTrue(torch.allclose(actual, expected, atol=1e-6))


if __name__ == '__main__':
    unittest.main()
