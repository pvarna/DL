import unittest

import torch

from dl_lib.nn import LeakyReLU, ReLU, Sequential, Sigmoid, Tanh


class TestSigmoidForward(unittest.TestCase):
    def test_when_called_with_zero_then_returns_half(self):
        # Arrange
        layer = Sigmoid()

        # Act
        actual = layer(torch.tensor(0.0)).item()

        # Assert
        expected = 0.5
        self.assertAlmostEqual(actual, expected)

    def test_when_called_with_large_positive_then_returns_close_to_one(self):
        # Arrange
        layer = Sigmoid()

        # Act
        actual = layer(torch.tensor(100.0)).item()

        # Assert
        expected = 1.0
        self.assertAlmostEqual(actual, expected, places=5)

    def test_when_called_with_large_negative_then_returns_close_to_zero(self):
        # Arrange
        layer = Sigmoid()

        # Act
        actual = layer(torch.tensor(-100.0)).item()

        # Assert
        expected = 0.0
        self.assertAlmostEqual(actual, expected, places=5)

    def test_when_called_with_tensor_then_returns_tensor(self):
        # Arrange
        x = torch.randn(5)

        # Act
        actual = Sigmoid()(x)

        # Assert
        expected = torch.Tensor
        self.assertIsInstance(actual, expected)

    def test_when_called_with_tensor_then_output_shape_matches_input(self):
        # Arrange
        x = torch.randn(3, 4)

        # Act
        actual = Sigmoid()(x).shape

        # Assert
        expected = x.shape
        self.assertEqual(actual, expected)

    def test_when_called_with_tensor_then_all_values_in_zero_one_range(self):
        # Arrange
        x = torch.randn(100)

        # Act
        actual = Sigmoid()(x)

        # Assert
        self.assertTrue((actual >= 0).all() and (actual <= 1).all())


class TestTanhForward(unittest.TestCase):
    def test_when_called_with_zero_then_returns_zero(self):
        # Arrange
        layer = Tanh()

        # Act
        actual = layer(torch.tensor(0.0)).item()

        # Assert
        expected = 0.0
        self.assertAlmostEqual(actual, expected)

    def test_when_called_with_tensor_then_returns_tensor(self):
        # Arrange
        x = torch.randn(5)

        # Act
        actual = Tanh()(x)

        # Assert
        expected = torch.Tensor
        self.assertIsInstance(actual, expected)

    def test_when_called_with_tensor_then_output_shape_matches_input(self):
        # Arrange
        x = torch.randn(3, 4)

        # Act
        actual = Tanh()(x).shape

        # Assert
        expected = x.shape
        self.assertEqual(actual, expected)

    def test_when_called_with_tensor_then_all_values_in_minus_one_one_range(
            self):
        # Arrange
        x = torch.randn(100)

        # Act
        actual = Tanh()(x)

        # Assert
        self.assertTrue((actual >= -1).all() and (actual <= 1).all())


class TestReLUForward(unittest.TestCase):
    def test_when_called_with_positive_value_then_returns_same_value(self):
        # Arrange
        layer = ReLU()

        # Act
        actual = layer(torch.tensor(3.0)).item()

        # Assert
        expected = 3.0
        self.assertAlmostEqual(actual, expected)

    def test_when_called_with_negative_value_then_returns_zero(self):
        # Arrange
        layer = ReLU()

        # Act
        actual = layer(torch.tensor(-5.0)).item()

        # Assert
        expected = 0.0
        self.assertAlmostEqual(actual, expected)

    def test_when_called_with_zero_then_returns_zero(self):
        # Arrange
        layer = ReLU()

        # Act
        actual = layer(torch.tensor(0.0)).item()

        # Assert
        expected = 0.0
        self.assertAlmostEqual(actual, expected)

    def test_when_called_with_tensor_then_returns_tensor(self):
        # Arrange
        x = torch.randn(5)

        # Act
        actual = ReLU()(x)

        # Assert
        expected = torch.Tensor
        self.assertIsInstance(actual, expected)

    def test_when_called_with_tensor_then_output_shape_matches_input(self):
        # Arrange
        x = torch.randn(3, 4)

        # Act
        actual = ReLU()(x).shape

        # Assert
        expected = x.shape
        self.assertEqual(actual, expected)

    def test_when_called_with_mixed_tensor_then_negatives_become_zero(self):
        # Arrange
        x = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])

        # Act
        actual = ReLU()(x)

        # Assert
        expected = torch.tensor([0.0, 0.0, 0.0, 1.0, 2.0])
        self.assertTrue(torch.allclose(actual, expected))


class TestLeakyReLUForward(unittest.TestCase):
    def test_when_called_with_positive_value_then_returns_same_value(self):
        # Arrange
        layer = LeakyReLU()

        # Act
        actual = layer(torch.tensor(3.0)).item()

        # Assert
        expected = 3.0
        self.assertAlmostEqual(actual, expected)

    def test_when_called_with_negative_value_then_returns_scaled_value(self):
        # Arrange
        layer = LeakyReLU(negative_slope=0.1)

        # Act
        actual = layer(torch.tensor(-2.0)).item()

        # Assert
        expected = -0.2
        self.assertAlmostEqual(actual, expected, places=5)

    def test_when_called_with_zero_then_returns_zero(self):
        # Arrange
        layer = LeakyReLU()

        # Act
        actual = layer(torch.tensor(0.0)).item()

        # Assert
        expected = 0.0
        self.assertAlmostEqual(actual, expected)

    def test_when_default_slope_then_negative_value_scaled_by_one_hundredth(
            self):
        # Arrange
        layer = LeakyReLU()

        # Act
        actual = layer(torch.tensor(-10.0)).item()

        # Assert
        expected = -0.1
        self.assertAlmostEqual(actual, expected, places=5)

    def test_when_called_with_tensor_then_returns_tensor(self):
        # Arrange
        x = torch.randn(5)

        # Act
        actual = LeakyReLU()(x)

        # Assert
        expected = torch.Tensor
        self.assertIsInstance(actual, expected)

    def test_when_called_with_tensor_then_output_shape_matches_input(self):
        # Arrange
        x = torch.randn(3, 4)

        # Act
        actual = LeakyReLU()(x).shape

        # Assert
        expected = x.shape
        self.assertEqual(actual, expected)

    def test_when_called_with_mixed_tensor_then_negatives_are_scaled(self):
        # Arrange
        x = torch.tensor([-2.0, 0.0, 2.0])

        # Act
        actual = LeakyReLU(negative_slope=0.5)(x)

        # Assert
        expected = torch.tensor([-1.0, 0.0, 2.0])
        self.assertTrue(torch.allclose(actual, expected))


class TestSequentialForward(unittest.TestCase):
    def test_when_called_with_single_module_then_applies_it(self):
        # Arrange
        model = Sequential(ReLU())

        # Act
        actual = model(torch.tensor(-1.0)).item()

        # Assert
        expected = 0.0
        self.assertAlmostEqual(actual, expected)

    def test_when_called_with_chained_modules_then_applies_in_order(self):
        # Arrange
        x = torch.tensor([-1.0, 0.0, 1.0])
        model = Sequential(ReLU(), Sigmoid())

        # Act
        actual = model(x)

        # Assert
        expected = Sigmoid()(ReLU()(x))
        self.assertTrue(torch.allclose(actual, expected))

    def test_when_called_with_tensor_then_returns_tensor(self):
        # Arrange
        x = torch.randn(5)

        # Act
        actual = Sequential(ReLU())(x)

        # Assert
        expected = torch.Tensor
        self.assertIsInstance(actual, expected)

    def test_when_called_with_tensor_then_output_shape_matches_input(self):
        # Arrange
        x = torch.randn(3, 4)

        # Act
        actual = Sequential(ReLU(), Tanh())(x).shape

        # Assert
        expected = x.shape
        self.assertEqual(actual, expected)

    def test_when_empty_then_returns_input_unchanged(self):
        # Arrange
        x = torch.tensor([1.0, 2.0, 3.0])

        # Act
        actual = Sequential()(x)

        # Assert
        expected = x
        self.assertTrue(torch.allclose(actual, expected))


class TestSequentialAppend(unittest.TestCase):
    def test_when_module_appended_then_it_is_applied_last(self):
        # Arrange
        model = Sequential(ReLU())
        model.append(Sigmoid())
        x = torch.tensor([-1.0, 1.0])

        # Act
        actual = model(x)

        # Assert
        expected = Sigmoid()(ReLU()(x))
        self.assertTrue(torch.allclose(actual, expected))

    def test_when_module_appended_to_empty_then_it_is_applied(self):
        # Arrange
        model = Sequential()
        model.append(ReLU())

        # Act
        actual = model(torch.tensor(-3.0)).item()

        # Assert
        expected = 0.0
        self.assertAlmostEqual(actual, expected)


class TestSequentialExtend(unittest.TestCase):
    def test_when_extended_then_new_modules_are_applied_after_existing(self):
        # Arrange
        model1 = Sequential(ReLU())
        model2 = Sequential(Sigmoid(), Tanh())
        model1.extend(model2)
        x = torch.randn(5)

        # Act
        actual = model1(x)

        # Assert
        expected = Tanh()(Sigmoid()(ReLU()(x)))
        self.assertTrue(torch.allclose(actual, expected))

    def test_when_extended_with_empty_sequential_then_behavior_unchanged(self):
        # Arrange
        model = Sequential(ReLU())
        model.extend(Sequential())
        x = torch.tensor([-1.0, 1.0])

        # Act
        actual = model(x)

        # Assert
        expected = ReLU()(x)
        self.assertTrue(torch.allclose(actual, expected))


class TestSequentialInsert(unittest.TestCase):
    def test_when_inserted_at_zero_then_module_is_applied_first(self):
        # Arrange
        model = Sequential(Sigmoid())
        model.insert(0, ReLU())
        x = torch.tensor([-1.0, 1.0])

        # Act
        actual = model(x)

        # Assert
        expected = Sigmoid()(ReLU()(x))
        self.assertTrue(torch.allclose(actual, expected))

    def test_when_inserted_at_end_then_module_is_applied_last(self):
        # Arrange
        model = Sequential(ReLU())
        model.insert(1, Tanh())
        x = torch.tensor([-1.0, 1.0])

        # Act
        actual = model(x)

        # Assert
        expected = Tanh()(ReLU()(x))
        self.assertTrue(torch.allclose(actual, expected))

    def test_when_inserted_in_middle_then_order_is_correct(self):
        # Arrange
        model = Sequential(ReLU(), Sigmoid())
        model.insert(1, Tanh())
        x = torch.tensor([-1.0, 0.0, 1.0])

        # Act
        actual = model(x)

        # Assert
        expected = Sigmoid()(Tanh()(ReLU()(x)))
        self.assertTrue(torch.allclose(actual, expected))


if __name__ == '__main__':
    unittest.main()
