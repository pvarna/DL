import unittest

import torch

from dl_lib.nn import Conv1d


class TestConv1dInit(unittest.TestCase):

    def test_when_called_with_in_channels_none_then_raises_value_error(self):
        # Arrange
        in_channels = None
        out_channels = 16
        kernel_size = 3

        # Act & Assert
        with self.assertRaises(ValueError):
            Conv1d(in_channels, out_channels, kernel_size)

    def test_when_called_with_out_channels_none_then_raises_value_error(self):
        # Arrange
        in_channels = 3
        out_channels = None
        kernel_size = 3

        # Act & Assert
        with self.assertRaises(ValueError):
            Conv1d(in_channels, out_channels, kernel_size)

    def test_when_called_with_kernel_size_none_then_raises_value_error(self):
        # Arrange
        in_channels = 3
        out_channels = 16
        kernel_size = None

        # Act & Assert
        with self.assertRaises(ValueError):
            Conv1d(in_channels, out_channels, kernel_size)

    def test_when_called_with_valid_arguments_then_initializes_attributes(
            self):
        # Arrange
        in_channels = 3
        out_channels = 16
        kernel_size = 3
        stride = 2
        padding = 1

        # Act
        layer = Conv1d(in_channels, out_channels, kernel_size, stride, padding)

        # Assert
        self.assertEqual(layer.in_channels, in_channels)
        self.assertEqual(layer.out_channels, out_channels)
        self.assertEqual(layer.kernel_size, kernel_size)
        self.assertEqual(layer.stride, stride)
        self.assertEqual(layer.padding, padding)

    def test_when_stride_not_specified_then_defaults_to_one(self):
        # Arrange
        expected = 1

        # Act
        layer = Conv1d(in_channels=1, out_channels=1, kernel_size=3)
        actual = layer.stride

        # Assert
        self.assertEqual(actual, expected)

    def test_when_padding_not_specified_then_defaults_to_zero(self):
        # Arrange
        expected = 0

        # Act
        layer = Conv1d(in_channels=1, out_channels=1, kernel_size=3)
        actual = layer.padding

        # Assert
        self.assertEqual(actual, expected)

    def test_when_bias_not_specified_then_bias_tensor_is_created(self):
        # Act
        layer = Conv1d(in_channels=1, out_channels=1, kernel_size=3)

        # Assert
        self.assertIsNotNone(layer.bias)

    def test_when_constructed_then_weight_shape_is_correct(self):
        # Arrange
        in_channels = 3
        out_channels = 16
        kernel_size = 3
        expected = (out_channels, in_channels, kernel_size)

        # Act
        layer = Conv1d(in_channels, out_channels, kernel_size)
        actual = layer.weight.shape

        # Assert
        self.assertEqual(actual, expected)

    def test_when_bias_is_true_then_bias_shape_is_correct(self):
        # Arrange
        out_channels = 16
        expected = (out_channels, )

        # Act
        layer = Conv1d(in_channels=3,
                       out_channels=out_channels,
                       kernel_size=3,
                       bias=True)
        actual = layer.bias.shape

        # Assert
        self.assertEqual(actual, expected)

    def test_when_bias_is_false_then_bias_is_none(self):
        # Act
        layer = Conv1d(in_channels=3,
                       out_channels=16,
                       kernel_size=3,
                       bias=False)

        # Assert
        self.assertIsNone(layer.bias)

    def test_when_constructed_then_weights_are_in_valid_range(self):
        # Arrange
        in_channels = 3
        kernel_size = 3
        sqrt_k = (1 / (in_channels * kernel_size))**0.5

        # Act
        layer = Conv1d(in_channels, out_channels=16, kernel_size=kernel_size)

        # Assert
        self.assertTrue((layer.weight >= -sqrt_k).all()
                        and (layer.weight <= sqrt_k).all())


class TestConv1dForward(unittest.TestCase):

    def test_when_called_then_returns_tensor(self):
        # Arrange
        x = torch.randn(1, 1, 5)
        expected = torch.Tensor

        # Act
        actual = Conv1d(in_channels=1, out_channels=1, kernel_size=3)(x)

        # Assert
        self.assertIsInstance(actual, expected)

    def test_when_called_then_output_shape_is_correct(self):
        # Arrange
        x = torch.randn(1, 1, 5)
        expected = (1, 1, 3)

        # Act
        actual = Conv1d(in_channels=1, out_channels=1, kernel_size=3)(x).shape

        # Assert
        self.assertEqual(actual, expected)

    def test_when_called_with_stride_2_then_output_length_is_correct(self):
        # Arrange
        x = torch.randn(1, 1, 8)
        expected = (1, 1, 3)

        # Act
        actual = Conv1d(in_channels=1, out_channels=1, kernel_size=3,
                        stride=2)(x).shape

        # Assert
        self.assertEqual(actual, expected)

    def test_when_called_with_integer_padding_then_output_length_accounts_for_padding(
            self):
        # Arrange
        x = torch.randn(1, 1, 5)
        expected = (1, 1, 5)

        # Act
        actual = Conv1d(in_channels=1,
                        out_channels=1,
                        kernel_size=3,
                        padding=1)(x).shape

        # Assert
        self.assertEqual(actual, expected)

    def test_when_called_with_padding_same_then_output_length_equals_input_length(
            self):
        # Arrange
        x = torch.randn(1, 1, 5)
        expected = (1, 1, 5)

        # Act
        actual = Conv1d(in_channels=1,
                        out_channels=1,
                        kernel_size=3,
                        padding='same')(x).shape

        # Assert
        self.assertEqual(actual, expected)

    def test_when_called_with_known_weights_then_output_matches_expected(self):
        # Arrange
        layer = Conv1d(in_channels=1,
                       out_channels=1,
                       kernel_size=3,
                       bias=False)
        layer.weight = torch.ones(1, 1, 3)
        x = torch.tensor([[[1.0, 2.0, 3.0, 4.0, 5.0]]])
        expected = 6.0

        # Act
        actual = layer(x)[0, 0, 0].item()

        # Assert
        self.assertEqual(actual, expected)

    def test_when_bias_is_false_then_output_does_not_include_bias(self):
        # Arrange
        layer = Conv1d(in_channels=1,
                       out_channels=1,
                       kernel_size=3,
                       bias=False)
        layer.weight = torch.zeros(1, 1, 3)
        x = torch.randn(1, 1, 5)
        expected = torch.zeros(1, 1, 3)

        # Act
        actual = layer(x)

        # Assert
        self.assertTrue(torch.allclose(actual, expected))
