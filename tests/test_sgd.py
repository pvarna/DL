import unittest

import torch

from dl_lib.optim import SGD


class TestSGDStep(unittest.TestCase):

    def test_when_called_once_then_params_are_updated(self):
        # Arrange
        p = torch.tensor([2.0, 3.0], requires_grad=True)
        p.grad = torch.tensor([4.0, 6.0])
        optimizer = SGD([p], lr=0.1)
        expected = torch.tensor([1.6, 2.4])

        # Act
        optimizer.step()
        actual = p.data

        # Assert
        self.assertTrue(torch.allclose(actual, expected))

    def test_when_called_once_then_params_decrease_when_gradient_is_positive(
            self):
        # Arrange
        p = torch.tensor([5.0], requires_grad=True)
        p.grad = torch.tensor([2.0])
        p_before = p.data.clone()
        optimizer = SGD([p], lr=0.1)

        # Act
        optimizer.step()

        # Assert
        self.assertTrue((p.data < p_before).all())

    def test_when_momentum_is_nonzero_then_second_step_uses_accumulated_velocity(
            self):
        # Arrange
        p = torch.tensor([1.0], requires_grad=True)
        optimizer = SGD([p], lr=0.1, momentum=0.9)
        expected = torch.tensor([0.71])

        # Act
        p.grad = torch.tensor([1.0])
        optimizer.step()

        p.grad = torch.tensor([1.0])
        optimizer.step()

        actual = p.data

        # Assert
        self.assertTrue(torch.allclose(actual, expected, atol=1e-6))


class TestSGDZeroGrad(unittest.TestCase):

    def test_when_called_then_gradients_are_zeroed(self):
        # Arrange
        p = torch.tensor([3.0, 5.0], requires_grad=True)
        p.grad = torch.tensor([3.0, 5.0])
        optimizer = SGD([p], lr=0.1)
        expected = torch.zeros(2)

        # Act
        optimizer.zero_grad()
        actual = p.grad

        # Assert
        self.assertTrue(torch.allclose(actual, expected))


if __name__ == '__main__':
    unittest.main()
