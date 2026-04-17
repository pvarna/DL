import unittest

import torch

from dl_lib.nn import CrossEntropyLoss


class TestCrossEntropyLossForward(unittest.TestCase):

    def test_when_reduction_is_none_then_output_length_matches_batch(self):
        # Arrange
        loss_function = CrossEntropyLoss(reduction='none')
        input = torch.randn(4, 5)
        target = torch.randint(0, 5, (4,))
        expected = (4,)

        # Act
        actual = loss_function(input, target).shape

        # Assert
        self.assertEqual(actual, expected)

    def test_when_reduction_is_mean_then_returns_scalar(self):
        # Arrange
        loss_function = CrossEntropyLoss(reduction='mean')
        input = torch.randn(4, 5)
        target = torch.randint(0, 5, (4,))
        expected = torch.Size([])

        # Act
        actual = loss_function(input, target).shape

        # Assert
        self.assertEqual(actual, expected)

    def test_when_reduction_is_mean_then_value_equals_mean_of_none(self):
        # Arrange
        input = torch.randn(4, 5)
        target = torch.randint(0, 5, (4,))
        expected = CrossEntropyLoss(reduction='none')(input, target).mean()

        # Act
        actual = CrossEntropyLoss(reduction='mean')(input, target)

        # Assert
        self.assertTrue(torch.allclose(actual, expected))

    def test_when_reduction_is_sum_then_value_equals_sum_of_none(self):
        # Arrange
        input = torch.randn(4, 5)
        target = torch.randint(0, 5, (4,))
        expected = CrossEntropyLoss(reduction='none')(input, target).sum()

        # Act
        actual = CrossEntropyLoss(reduction='sum')(input, target)

        # Assert
        self.assertTrue(torch.allclose(actual, expected))

    def test_when_called_with_tensor_then_returns_tensor(self):
        # Arrange
        input = torch.randn(3, 4)
        target = torch.randint(0, 4, (3,))
        expected = torch.Tensor

        # Act
        actual = CrossEntropyLoss()(input, target)

        # Assert
        self.assertIsInstance(actual, expected)


if __name__ == '__main__':
    unittest.main()
