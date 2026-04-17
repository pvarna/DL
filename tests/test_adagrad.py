import unittest

import torch

from dl_lib.optim import AdaGrad


class TestAdaGradStep(unittest.TestCase):

    def test_when_called_once_then_params_are_updated(self):
        # Arrange
        p = torch.tensor([1.0], requires_grad=True)
        p.grad = torch.tensor([2.0])
        optimizer = AdaGrad([p], lr=0.1)
        expected = 0.9

        # Act
        optimizer.step()
        actual = p.data.item()

        # Assert
        self.assertAlmostEqual(actual, expected)

    def test_when_called_twice_then_second_update_is_smaller_than_first(self):
        # Arrange
        p = torch.tensor([2.0], requires_grad=True)
        p.grad = torch.tensor([1.0])
        optimizer = AdaGrad([p], lr=0.1)
        p_before = p.data.clone()

        # Act
        optimizer.step()
        p_after_first = p.data.clone()

        p.grad = torch.tensor([1.0])
        optimizer.step()
        p_after_second = p.data.clone()

        # Assert
        delta_first = (p_before - p_after_first).abs()
        delta_second = (p_after_first - p_after_second).abs()
        self.assertTrue((delta_second < delta_first).all())


if __name__ == '__main__':
    unittest.main()
