import unittest

import torch

from dl_lib.nn import Dropout


class TestDropoutForward(unittest.TestCase):

    def test_when_p_is_zero_then_output_equals_input(self):
        # Arrange
        x = torch.tensor([1.0, 2.0, 3.0])
        expected = x

        # Act
        actual = Dropout(p=0.0)(x)

        # Assert
        self.assertTrue(torch.allclose(actual, expected))

    def test_when_called_with_tensor_then_returns_tensor(self):
        # Arrange
        x = torch.randn(5)
        expected = torch.Tensor

        # Act
        actual = Dropout()(x)

        # Assert
        self.assertIsInstance(actual, expected)

    def test_when_called_then_output_shape_matches_input(self):
        # Arrange
        x = torch.randn(3, 4)
        expected = x.shape

        # Act
        actual = Dropout()(x).shape

        # Assert
        self.assertEqual(actual, expected)

    def test_when_called_with_large_tensor_then_some_elements_are_zeroed(self):
        # Arrange
        x = torch.ones(1000)

        # Act
        actual = Dropout(p=0.5)(x)

        # Assert
        self.assertTrue((actual == 0.0).any())


if __name__ == '__main__':
    unittest.main()
