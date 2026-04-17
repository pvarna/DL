import unittest

import torch

from dl_lib.nn import Tanh


class TestTanhForward(unittest.TestCase):

    def test_when_called_with_zero_then_returns_zero(self):
        # Arrange
        layer = Tanh()
        expected = 0.0

        # Act
        actual = layer(torch.tensor(0.0)).item()

        # Assert
        self.assertEqual(actual, expected)

    def test_when_called_with_tensor_then_returns_tensor(self):
        # Arrange
        x = torch.randn(5)
        expected = torch.Tensor

        # Act
        actual = Tanh()(x)

        # Assert
        self.assertIsInstance(actual, expected)

    def test_when_called_with_tensor_then_output_shape_matches_input(self):
        # Arrange
        x = torch.randn(3, 4)
        expected = x.shape

        # Act
        actual = Tanh()(x).shape

        # Assert
        self.assertEqual(actual, expected)

    def test_when_called_with_tensor_then_all_values_in_minus_one_one_range(
            self):
        # Arrange
        x = torch.randn(100)

        # Act
        actual = Tanh()(x)

        # Assert
        self.assertTrue((actual >= -1).all() and (actual <= 1).all())


if __name__ == '__main__':
    unittest.main()
