import unittest

import torch

from dl_lib.nn import LeakyReLU


class TestLeakyReLUForward(unittest.TestCase):

    def test_when_called_with_positive_value_then_returns_same_value(self):
        # Arrange
        layer = LeakyReLU()
        expected = 3.0

        # Act
        actual = layer(torch.tensor(3.0)).item()

        # Assert
        self.assertEqual(actual, expected)

    def test_when_called_with_negative_value_then_returns_scaled_value(self):
        # Arrange
        layer = LeakyReLU(negative_slope=0.1)
        expected = -0.2

        # Act
        actual = layer(torch.tensor(-2.0)).item()

        # Assert
        self.assertAlmostEqual(actual, expected)

    def test_when_called_with_zero_then_returns_zero(self):
        # Arrange
        layer = LeakyReLU()
        expected = 0.0

        # Act
        actual = layer(torch.tensor(0.0)).item()

        # Assert
        self.assertEqual(actual, expected)

    def test_when_default_slope_then_negative_value_scaled_by_one_hundredth(
            self):
        # Arrange
        layer = LeakyReLU()
        expected = -0.1

        # Act
        actual = layer(torch.tensor(-10.0)).item()

        # Assert
        self.assertAlmostEqual(actual, expected)

    def test_when_called_with_tensor_then_returns_tensor(self):
        # Arrange
        x = torch.randn(5)
        expected = torch.Tensor

        # Act
        actual = LeakyReLU()(x)

        # Assert
        self.assertIsInstance(actual, expected)

    def test_when_called_with_tensor_then_output_shape_matches_input(self):
        # Arrange
        x = torch.randn(3, 4)
        expected = x.shape

        # Act
        actual = LeakyReLU()(x).shape

        # Assert
        self.assertEqual(actual, expected)

    def test_when_called_with_mixed_tensor_then_negatives_are_scaled(self):
        # Arrange
        x = torch.tensor([-2.0, 0.0, 2.0])
        expected = torch.tensor([-1.0, 0.0, 2.0])

        # Act
        actual = LeakyReLU(negative_slope=0.5)(x)

        # Assert
        self.assertTrue(torch.allclose(actual, expected))


if __name__ == '__main__':
    unittest.main()
