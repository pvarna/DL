import unittest

import torch

from dl_lib.nn import Sigmoid


class TestSigmoidForward(unittest.TestCase):

    def test_when_called_with_zero_then_returns_half(self):
        # Arrange
        layer = Sigmoid()
        expected = 0.5

        # Act
        actual = layer(torch.tensor(0.0)).item()

        # Assert
        self.assertEqual(actual, expected)

    def test_when_called_with_large_positive_then_returns_close_to_one(self):
        # Arrange
        layer = Sigmoid()
        expected = 1.0

        # Act
        actual = layer(torch.tensor(100.0)).item()

        # Assert
        self.assertAlmostEqual(actual, expected)

    def test_when_called_with_large_negative_then_returns_close_to_zero(self):
        # Arrange
        layer = Sigmoid()
        expected = 0.0

        # Act
        actual = layer(torch.tensor(-100.0)).item()

        # Assert
        self.assertAlmostEqual(actual, expected)

    def test_when_called_with_tensor_then_returns_tensor(self):
        # Arrange
        x = torch.randn(5)
        expected = torch.Tensor

        # Act
        actual = Sigmoid()(x)

        # Assert
        self.assertIsInstance(actual, expected)

    def test_when_called_with_tensor_then_output_shape_matches_input(self):
        # Arrange
        x = torch.randn(3, 4)
        expected = x.shape

        # Act
        actual = Sigmoid()(x).shape

        # Assert
        self.assertEqual(actual, expected)

    def test_when_called_with_tensor_then_all_values_in_zero_one_range(self):
        # Arrange
        x = torch.randn(100)

        # Act
        actual = Sigmoid()(x)

        # Assert
        self.assertTrue((actual >= 0).all() and (actual <= 1).all())


if __name__ == '__main__':
    unittest.main()
