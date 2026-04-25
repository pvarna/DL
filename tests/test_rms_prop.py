import unittest

import torch

from dl_lib.optim import RMSprop


class TestRMSpropInit(unittest.TestCase):

    def test_when_parameters_none_then_raises_value_error(self):
        # Act & Assert
        with self.assertRaises(ValueError):
            RMSprop(None)

    def test_when_lr_not_specified_then_defaults_to_one_hundredth(self):
        # Arrange
        p = torch.tensor([1.0], requires_grad=True)
        expected = 0.01

        # Act
        optimizer = RMSprop([p])
        actual = optimizer.lr

        # Assert
        self.assertEqual(actual, expected)

    def test_when_alpha_not_specified_then_defaults_to_ninety_nine_hundredths(
            self):
        # Arrange
        p = torch.tensor([1.0], requires_grad=True)
        expected = 0.99

        # Act
        optimizer = RMSprop([p])
        actual = optimizer.alpha

        # Assert
        self.assertEqual(actual, expected)

    def test_when_eps_not_specified_then_defaults_to_one_hundred_millionth(
            self):
        # Arrange
        p = torch.tensor([1.0], requires_grad=True)
        expected = 1e-8

        # Act
        optimizer = RMSprop([p])
        actual = optimizer.eps

        # Assert
        self.assertEqual(actual, expected)


class TestRMSpropStep(unittest.TestCase):

    def test_when_called_once_then_params_are_updated(self):
        # Arrange
        p = torch.tensor([1.0], requires_grad=True)
        p.grad = torch.tensor([2.0])
        optimizer = RMSprop([p], lr=0.1, alpha=0.9)
        expected = 1.0 - 0.1 * 2.0 / (0.4**0.5 + 1e-8)

        # Act
        optimizer.step()
        actual = p.data.item()

        # Assert
        self.assertAlmostEqual(actual, expected)

    def test_when_called_twice_then_squared_average_follows_exponential_moving_average(
            self):
        # Arrange
        p = torch.tensor([0.0], requires_grad=True)
        p.grad = torch.tensor([1.0])
        optimizer = RMSprop([p], lr=0.1, alpha=0.9)
        expected = torch.tensor([0.19])

        # Act
        optimizer.step()

        p.grad = torch.tensor([1.0])
        optimizer.step()

        actual = optimizer.squared_average[0]

        # Assert
        self.assertTrue(torch.allclose(actual, expected, atol=1e-6))
