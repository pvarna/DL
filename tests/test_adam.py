import unittest

import torch

from dl_lib.optim import Adam


class TestAdamInit(unittest.TestCase):

    def test_when_parameters_none_then_raises_value_error(self):
        # Act & Assert
        with self.assertRaises(ValueError):
            Adam(None)

    def test_when_optional_parameters_not_specified_then_defaults_are_applied(
            self):
        # Arrange
        p = torch.tensor([1.0], requires_grad=True)
        expected_lr = 0.001
        expected_beta1 = 0.9
        expected_beta2 = 0.999
        expected_eps = 1e-8

        # Act
        optimizer = Adam([p])

        # Assert
        self.assertEqual(optimizer.lr, expected_lr)
        self.assertEqual(optimizer.beta1, expected_beta1)
        self.assertEqual(optimizer.beta2, expected_beta2)
        self.assertEqual(optimizer.eps, expected_eps)


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
