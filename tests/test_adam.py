import unittest

import torch

from dl_lib.optim import Adam


class TestAdamInit(unittest.TestCase):

    def test_when_parameters_none_then_raises_value_error(self):
        # Act & Assert
        with self.assertRaises(ValueError):
            Adam(None)

    def test_when_lr_not_specified_then_defaults_to_one_thousandth(self):
        # Arrange
        p = torch.tensor([1.0], requires_grad=True)
        expected = 0.001

        # Act
        optimizer = Adam([p])
        actual = optimizer.lr

        # Assert
        self.assertEqual(actual, expected)

    def test_when_beta1_not_specified_then_defaults_to_nine_tenths(self):
        # Arrange
        p = torch.tensor([1.0], requires_grad=True)
        expected = 0.9

        # Act
        optimizer = Adam([p])
        actual = optimizer.beta1

        # Assert
        self.assertEqual(actual, expected)

    def test_when_beta2_not_specified_then_defaults_to_nine_hundred_ninety_nine_thousandths(
            self):
        # Arrange
        p = torch.tensor([1.0], requires_grad=True)
        expected = 0.999

        # Act
        optimizer = Adam([p])
        actual = optimizer.beta2

        # Assert
        self.assertEqual(actual, expected)

    def test_when_eps_not_specified_then_defaults_to_one_hundred_millionth(
            self):
        # Arrange
        p = torch.tensor([1.0], requires_grad=True)
        expected = 1e-8

        # Act
        optimizer = Adam([p])
        actual = optimizer.eps

        # Assert
        self.assertEqual(actual, expected)


class TestAdamStep(unittest.TestCase):

    def test_when_called_once_then_params_are_updated(self):
        # Arrange
        p = torch.tensor([1.0], requires_grad=True)
        p.grad = torch.tensor([1.0])
        optimizer = Adam([p], lr=0.001)
        expected = 1.0 - 0.001 * (0.1 / 0.1) / ((0.001 / 0.001)**0.5 + 1e-8)

        # Act
        optimizer.step()
        actual = p.data.item()

        # Assert
        self.assertAlmostEqual(actual, expected)

    def test_when_called_once_then_step_counter_is_incremented(self):
        # Arrange
        p = torch.tensor([1.0], requires_grad=True)
        p.grad = torch.tensor([1.0])
        optimizer = Adam([p])
        expected = 1

        # Act
        optimizer.step()
        actual = optimizer.t

        # Assert
        self.assertEqual(actual, expected)
