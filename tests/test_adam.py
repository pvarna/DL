import unittest

import torch

from dl_lib.optim import Adam


class TestAdamStep(unittest.TestCase):

    def test_when_called_once_then_params_are_updated(self):
        # Arrange
        p = torch.tensor([1.0], requires_grad=True)
        p.grad = torch.tensor([1.0])
        optimizer = Adam([p], lr=0.001)
        expected = 1.0 - 0.001 * (0.1 / 0.1) / ((0.001 / 0.001)**0.5 + 1e-8)

        # Act
        optimizer.step()
        actual = p.data.item()

        # Assert
        self.assertAlmostEqual(actual, expected)

    def test_when_called_once_then_step_counter_is_incremented(self):
        # Arrange
        p = torch.tensor([1.0], requires_grad=True)
        p.grad = torch.tensor([1.0])
        optimizer = Adam([p])
        expected = 1

        # Act
        optimizer.step()
        actual = optimizer.t

        # Assert
        self.assertEqual(actual, expected)


if __name__ == '__main__':
    unittest.main()
