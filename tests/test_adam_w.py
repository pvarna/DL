import unittest

import torch

from dl_lib.optim import Adam, AdamW


class TestAdamWInit(unittest.TestCase):

    def test_when_parameters_none_then_raises_value_error(self):
        # Act & Assert
        with self.assertRaises(ValueError):
            AdamW(None)

    def test_when_optional_parameters_not_specified_then_defaults_are_applied(
            self):
        # Arrange
        p = torch.tensor([1.0], requires_grad=True)
        expected_lr = 0.001
        expected_beta1 = 0.9
        expected_beta2 = 0.999
        expected_eps = 1e-8
        expected_weight_decay = 0.01

        # Act
        optimizer = AdamW([p])

        # Assert
        self.assertEqual(optimizer.lr, expected_lr)
        self.assertEqual(optimizer.beta1, expected_beta1)
        self.assertEqual(optimizer.beta2, expected_beta2)
        self.assertEqual(optimizer.eps, expected_eps)
        self.assertEqual(optimizer.weight_decay, expected_weight_decay)


class TestAdamWStep(unittest.TestCase):

    def test_when_called_once_then_params_are_updated(self):
        # Arrange
        p = torch.tensor([1.0], requires_grad=True)
        p.grad = torch.tensor([1.0])
        optimizer = AdamW([p], lr=0.001, weight_decay=0.1)
        p_before_adam = 1.0 - 0.001 * 0.1 * 1.0
        m_hat = 1.0
        v_hat = 1.0
        expected = p_before_adam - 0.001 * m_hat / (v_hat**0.5 + 1e-8)

        # Act
        optimizer.step()
        actual = p.data.item()

        # Assert
        self.assertAlmostEqual(actual, expected)

    def test_when_called_once_then_weight_decay_shrinks_params_more_than_adam(
            self):
        # Arrange
        p_adam = torch.tensor([1.0], requires_grad=True)
        p_adam.grad = torch.tensor([1.0])
        p_adamw = torch.tensor([1.0], requires_grad=True)
        p_adamw.grad = torch.tensor([1.0])
        opt_adam = Adam([p_adam], lr=0.001)
        opt_adamw = AdamW([p_adamw], lr=0.001, weight_decay=0.1)

        # Act
        opt_adam.step()
        opt_adamw.step()
        actual_adamw = p_adamw.data.item()
        actual_adam = p_adam.data.item()

        # Assert
        self.assertTrue(actual_adamw < actual_adam)
