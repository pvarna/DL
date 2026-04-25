import unittest

import torch

from dl_lib.nn import MaxPool2d


class TestMaxPool2dInit(unittest.TestCase):

    def test_when_called_with_kernel_size_none_then_raises_value_error(self):
        # Arrange
        kernel_size = None
        stride = 2
        padding = 0

        # Act & Assert
        with self.assertRaises(ValueError):
            MaxPool2d(kernel_size, stride, padding)

    def test_when_optional_parameters_not_specified_then_defaults_are_applied(
            self):
        # Arrange
        kernel_size = 2
        expected_stride = kernel_size
        expected_padding = 0

        # Act
        layer = MaxPool2d(kernel_size)

        # Assert
        self.assertEqual(layer.stride, expected_stride)
        self.assertEqual(layer.padding, expected_padding)

    def test_when_called_with_valid_arguments_then_initializes_attributes(
            self):
        # Arrange
        kernel_size = (2, 2)
        stride = (2, 2)
        padding = (1, 1)

        # Act
        layer = MaxPool2d(kernel_size, stride, padding)

        # Assert
        self.assertEqual(layer.kernel_size, kernel_size)
        self.assertEqual(layer.stride, stride)
        self.assertEqual(layer.padding, padding)


class TestMaxPool2dForward(unittest.TestCase):

    def test_when_called_then_returns_tensor(self):
        # Arrange
        x = torch.randn(1, 1, 4, 4)
        expected = torch.Tensor

        # Act
        actual = MaxPool2d(kernel_size=2, stride=2, padding=0)(x)

        # Assert
        self.assertIsInstance(actual, expected)

    def test_when_called_with_kernel_size_2_and_stride_2_then_output_shape_is_correct(
            self):
        # Arrange
        x = torch.randn(1, 1, 4, 4)
        expected = (1, 1, 2, 2)

        # Act
        actual = MaxPool2d(kernel_size=2, stride=2, padding=0)(x).shape

        # Assert
        self.assertEqual(actual, expected)

    def test_when_called_with_stride_1_then_output_shape_is_correct(self):
        # Arrange
        x = torch.randn(1, 1, 4, 4)
        expected = (1, 1, 3, 3)

        # Act
        actual = MaxPool2d(kernel_size=2, stride=1, padding=0)(x).shape

        # Assert
        self.assertEqual(actual, expected)

    def test_when_called_with_padding_1_then_output_shape_accounts_for_padding(
            self):
        # Arrange
        x = torch.randn(1, 1, 4, 4)
        expected = (1, 1, 3, 3)

        # Act
        actual = MaxPool2d(kernel_size=2, stride=2, padding=1)(x).shape

        # Assert
        self.assertEqual(actual, expected)

    def test_when_called_with_batch_size_greater_than_one_then_output_shape_is_correct(
            self):
        # Arrange
        x = torch.randn(3, 1, 4, 4)
        expected = (3, 1, 2, 2)

        # Act
        actual = MaxPool2d(kernel_size=2, stride=2, padding=0)(x).shape

        # Assert
        self.assertEqual(actual, expected)

    def test_when_called_with_multiple_channels_then_output_shape_is_correct(
            self):
        # Arrange
        x = torch.randn(1, 3, 4, 4)
        expected = (1, 3, 2, 2)

        # Act
        actual = MaxPool2d(kernel_size=2, stride=2, padding=0)(x).shape

        # Assert
        self.assertEqual(actual, expected)

    def test_when_called_then_each_output_value_is_max_of_corresponding_window(
            self):
        # Arrange
        x = torch.tensor([[[[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0],
                            [9.0, 10.0, 11.0, 12.0], [13.0, 14.0, 15.0,
                                                      16.0]]]])
        expected = torch.tensor([[[[6.0, 8.0], [14.0, 16.0]]]])

        # Act
        actual = MaxPool2d(kernel_size=2, stride=2, padding=0)(x)

        # Assert
        self.assertTrue(torch.allclose(actual, expected))
