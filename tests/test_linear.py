import unittest

import torch

from dl_lib.nn import Linear


class TestLinearInit(unittest.TestCase):

    def test_when_called_with_in_features_none_then_raises_value_error(self):
        # Arrange
        in_features = None
        out_features = 3

        # Act & Assert
        with self.assertRaises(ValueError):
            Linear(in_features, out_features)

    def test_when_called_with_out_features_none_then_raises_value_error(self):
        # Arrange
        in_features = 4
        out_features = None

        # Act & Assert
        with self.assertRaises(ValueError):
            Linear(in_features, out_features)

    def test_when_bias_not_specified_then_bias_tensor_is_created(self):
        # Act
        layer = Linear(4, 3)

        # Assert
        self.assertIsNotNone(layer.bias)

    def test_when_constructed_then_weight_shape_is_correct(self):
        # Arrange
        expected = (3, 4)

        # Act
        layer = Linear(4, 3)
        actual = layer.weight.shape

        # Assert
        self.assertEqual(actual, expected)

    def test_when_bias_is_true_then_bias_shape_is_correct(self):
        # Arrange
        expected = (3,)

        # Act
        layer = Linear(4, 3, bias=True)
        actual = layer.bias.shape

        # Assert
        self.assertEqual(actual, expected)

    def test_when_bias_is_false_then_bias_is_none(self):
        # Act
        layer = Linear(4, 3, bias=False)

        # Assert
        self.assertIsNone(layer.bias)

    def test_when_constructed_then_weights_are_in_valid_range(self):
        # Arrange
        in_features = 100
        sqrt_k = (1 / in_features)**0.5

        # Act
        layer = Linear(in_features, 50)

        # Assert
        self.assertTrue((layer.weight >= -sqrt_k).all()
                        and (layer.weight <= sqrt_k).all())

    def test_when_bias_is_true_then_bias_values_are_in_valid_range(self):
        # Arrange
        in_features = 100
        sqrt_k = (1 / in_features)**0.5

        # Act
        layer = Linear(in_features, 50, bias=True)

        # Assert
        self.assertTrue((layer.bias >= -sqrt_k).all()
                        and (layer.bias <= sqrt_k).all())


class TestLinearForward(unittest.TestCase):

    def test_when_called_with_1d_input_then_returns_tensor(self):
        # Arrange
        x = torch.randn(3)
        expected = torch.Tensor

        # Act
        actual = Linear(3, 4)(x)

        # Assert
        self.assertIsInstance(actual, expected)

    def test_when_called_with_1d_input_then_output_shape_is_correct(self):
        # Arrange
        x = torch.randn(3)
        expected = (4,)

        # Act
        actual = Linear(3, 4)(x).shape

        # Assert
        self.assertEqual(actual, expected)

    def test_when_called_with_2d_input_then_output_shape_is_correct(self):
        # Arrange
        x = torch.randn(5, 3)
        expected = (5, 4)

        # Act
        actual = Linear(3, 4)(x).shape

        # Assert
        self.assertEqual(actual, expected)

    def test_when_bias_is_false_then_output_equals_weight_matmul(self):
        # Arrange
        layer = Linear(3, 2, bias=False)
        layer.weight = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        x = torch.tensor([2.0, 3.0, 4.0])
        expected = torch.tensor([2.0, 3.0])

        # Act
        actual = layer(x)

        # Assert
        self.assertTrue(torch.allclose(actual, expected))

    def test_when_bias_is_true_then_output_equals_weight_matmul_plus_bias(
            self):
        # Arrange
        layer = Linear(2, 2, bias=True)
        layer.weight = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        layer.bias = torch.tensor([1.0, 2.0])
        x = torch.tensor([3.0, 4.0])
        expected = torch.tensor([4.0, 6.0])

        # Act
        actual = layer(x)

        # Assert
        self.assertTrue(torch.allclose(actual, expected))
