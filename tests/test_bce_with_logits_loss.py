import unittest

import torch

from dl_lib.nn import BCEWithLogitsLoss


class TestBCEWithLogitsLossInit(unittest.TestCase):

    def test_when_reduction_not_specified_then_defaults_to_mean(self):
        # Arrange
        expected = 'mean'

        # Act
        layer = BCEWithLogitsLoss()
        actual = layer.reduction

        # Assert
        self.assertEqual(actual, expected)

    def test_when_pos_weight_not_specified_then_defaults_to_none(self):
        # Act
        layer = BCEWithLogitsLoss()

        # Assert
        self.assertIsNone(layer.pos_weight)


class TestBCEWithLogitsLossForward(unittest.TestCase):

    def test_when_reduction_is_none_then_output_shape_matches_input(self):
        # Arrange
        loss_function = BCEWithLogitsLoss(reduction='none')
        input = torch.tensor([2.0, -1.0, 0.5])
        target = torch.tensor([1.0, 0.0, 1.0])
        expected = input.shape

        # Act
        actual = loss_function(input, target).shape

        # Assert
        self.assertEqual(actual, expected)

    def test_when_reduction_is_mean_then_returns_scalar(self):
        # Arrange
        loss_function = BCEWithLogitsLoss(reduction='mean')
        input = torch.tensor([2.0, -1.0, 0.5])
        target = torch.tensor([1.0, 0.0, 1.0])
        expected = torch.Size([])

        # Act
        actual = loss_function(input, target).shape

        # Assert
        self.assertEqual(actual, expected)

    def test_when_reduction_is_mean_then_value_equals_mean_of_none(self):
        # Arrange
        input = torch.tensor([2.0, -1.0, 0.5])
        target = torch.tensor([1.0, 0.0, 1.0])
        expected = BCEWithLogitsLoss(reduction='none')(input, target).mean()

        # Act
        actual = BCEWithLogitsLoss(reduction='mean')(input, target)

        # Assert
        self.assertTrue(torch.allclose(actual, expected))

    def test_when_reduction_is_sum_then_value_equals_sum_of_none(self):
        # Arrange
        input = torch.tensor([2.0, -1.0, 0.5])
        target = torch.tensor([1.0, 0.0, 1.0])
        expected = BCEWithLogitsLoss(reduction='none')(input, target).sum()

        # Act
        actual = BCEWithLogitsLoss(reduction='sum')(input, target)

        # Assert
        self.assertTrue(torch.allclose(actual, expected))

    def test_when_called_with_tensor_then_returns_tensor(self):
        # Arrange
        input = torch.tensor([2.0, -1.0])
        target = torch.tensor([1.0, 0.0])
        expected = torch.Tensor

        # Act
        actual = BCEWithLogitsLoss()(input, target)

        # Assert
        self.assertIsInstance(actual, expected)
