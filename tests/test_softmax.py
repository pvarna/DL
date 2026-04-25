import unittest

import torch

from dl_lib.nn import Softmax


class TestSoftmaxInit(unittest.TestCase):

    def test_when_dim_not_specified_then_defaults_to_none(self):
        # Arrange
        expected = None

        # Act
        layer = Softmax()
        actual = layer.dim

        # Assert
        self.assertEqual(actual, expected)


class TestSoftmaxForward(unittest.TestCase):

    def test_when_dim_is_none_then_raises_type_error(self):
        # Arrange
        layer = Softmax()
        x = torch.randn(5)

        # Act & Assert
        self.assertRaises(TypeError, layer, x)

    def test_when_called_with_1d_tensor_then_output_sums_to_one(self):
        # Arrange
        x = torch.tensor([1.0, 2.0, 3.0])
        expected = 1.0

        # Act
        actual = Softmax(dim=0)(x).sum().item()

        # Assert
        self.assertAlmostEqual(actual, expected)

    def test_when_called_with_2d_tensor_then_each_row_sums_to_one(self):
        # Arrange
        x = torch.randn(4, 5)
        expected = torch.ones(4)

        # Act
        actual = Softmax(dim=1)(x).sum(dim=1)

        # Assert
        self.assertTrue(torch.allclose(actual, expected, atol=1e-5))

    def test_when_called_then_all_values_are_positive(self):
        # Arrange
        x = torch.randn(10)

        # Act
        actual = Softmax(dim=0)(x)

        # Assert
        self.assertTrue((actual > 0).all())

    def test_when_called_with_tensor_then_output_shape_matches_input(self):
        # Arrange
        x = torch.randn(3, 4)
        expected = x.shape

        # Act
        actual = Softmax(dim=1)(x).shape

        # Assert
        self.assertEqual(actual, expected)

    def test_when_called_with_tensor_then_returns_tensor(self):
        # Arrange
        x = torch.randn(5)
        expected = torch.Tensor

        # Act
        actual = Softmax(dim=0)(x)

        # Assert
        self.assertIsInstance(actual, expected)
