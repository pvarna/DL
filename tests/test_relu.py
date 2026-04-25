import unittest

import torch

from dl_lib.nn import ReLU


class TestReLUForward(unittest.TestCase):

    def test_when_called_with_positive_value_then_returns_same_value(self):
        # Arrange
        layer = ReLU()
        expected = 3.0

        # Act
        actual = layer(torch.tensor(3.0)).item()

        # Assert
        self.assertEqual(actual, expected)

    def test_when_called_with_negative_value_then_returns_zero(self):
        # Arrange
        layer = ReLU()
        expected = 0.0

        # Act
        actual = layer(torch.tensor(-5.0)).item()

        # Assert
        self.assertEqual(actual, expected)

    def test_when_called_with_zero_then_returns_zero(self):
        # Arrange
        layer = ReLU()
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
        actual = ReLU()(x)

        # Assert
        self.assertIsInstance(actual, expected)

    def test_when_called_with_tensor_then_output_shape_matches_input(self):
        # Arrange
        x = torch.randn(3, 4)
        expected = x.shape

        # Act
        actual = ReLU()(x).shape

        # Assert
        self.assertEqual(actual, expected)

    def test_when_called_with_mixed_tensor_then_negatives_become_zero(self):
        # Arrange
        x = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
        expected = torch.tensor([0.0, 0.0, 0.0, 1.0, 2.0])

        # Act
        actual = ReLU()(x)

        # Assert
        self.assertTrue(torch.allclose(actual, expected))
