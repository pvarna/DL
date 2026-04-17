import unittest

import torch

from dl_lib.optim import Adam, AdamW


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


if __name__ == '__main__':
    unittest.main()
