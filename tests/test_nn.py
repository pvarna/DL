import unittest

import torch

from dl_lib.nn import LeakyReLU, Linear, ReLU, Sequential, Sigmoid, Tanh


class TestSigmoidForward(unittest.TestCase):

    def test_when_called_with_zero_then_returns_half(self):
        # Arrange
        layer = Sigmoid()
        expected = 0.5

        # Act
        actual = layer(torch.tensor(0.0))

        # Assert
        self.assertAlmostEqual(actual.item(), expected)

    def test_when_called_with_large_positive_then_returns_close_to_one(self):
        # Arrange
        layer = Sigmoid()
        expected = 1.0

        # Act
        actual = layer(torch.tensor(100.0))

        # Assert
        self.assertAlmostEqual(actual.item(), expected, places=5)

    def test_when_called_with_large_negative_then_returns_close_to_zero(self):
        # Arrange
        layer = Sigmoid()
        expected = 0.0

        # Act
        actual = layer(torch.tensor(-100.0))

        # Assert
        self.assertAlmostEqual(actual.item(), expected, places=5)

    def test_when_called_with_tensor_then_returns_tensor(self):
        # Arrange
        x = torch.randn(5)

        # Act
        actual = Sigmoid()(x)

        # Assert
        self.assertIsInstance(actual, torch.Tensor)

    def test_when_called_with_tensor_then_output_shape_matches_input(self):
        # Arrange
        x = torch.randn(3, 4)

        # Act
        actual = Sigmoid()(x)

        # Assert
        self.assertEqual(actual.shape, x.shape)

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
        expected = 0.0

        # Act
        actual = layer(torch.tensor(0.0))

        # Assert
        self.assertAlmostEqual(actual.item(), expected)

    def test_when_called_with_tensor_then_returns_tensor(self):
        # Arrange
        x = torch.randn(5)

        # Act
        actual = Tanh()(x)

        # Assert
        self.assertIsInstance(actual, torch.Tensor)

    def test_when_called_with_tensor_then_output_shape_matches_input(self):
        # Arrange
        x = torch.randn(3, 4)

        # Act
        actual = Tanh()(x)

        # Assert
        self.assertEqual(actual.shape, x.shape)

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
        expected = 3.0

        # Act
        actual = layer(torch.tensor(3.0))

        # Assert
        self.assertAlmostEqual(actual.item(), expected)

    def test_when_called_with_negative_value_then_returns_zero(self):
        # Arrange
        layer = ReLU()
        expected = 0.0

        # Act
        actual = layer(torch.tensor(-5.0))

        # Assert
        self.assertAlmostEqual(actual.item(), expected)

    def test_when_called_with_zero_then_returns_zero(self):
        # Arrange
        layer = ReLU()
        expected = 0.0

        # Act
        actual = layer(torch.tensor(0.0))

        # Assert
        self.assertAlmostEqual(actual.item(), expected)

    def test_when_called_with_tensor_then_returns_tensor(self):
        # Arrange
        x = torch.randn(5)

        # Act
        actual = ReLU()(x)

        # Assert
        self.assertIsInstance(actual, torch.Tensor)

    def test_when_called_with_tensor_then_output_shape_matches_input(self):
        # Arrange
        x = torch.randn(3, 4)

        # Act
        actual = ReLU()(x)

        # Assert
        self.assertEqual(actual.shape, x.shape)

    def test_when_called_with_mixed_tensor_then_negatives_become_zero(self):
        # Arrange
        x = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
        expected = torch.tensor([0.0, 0.0, 0.0, 1.0, 2.0])

        # Act
        actual = ReLU()(x)

        # Assert
        self.assertTrue(torch.allclose(actual, expected))


class TestLeakyReLUForward(unittest.TestCase):

    def test_when_called_with_positive_value_then_returns_same_value(self):
        # Arrange
        layer = LeakyReLU()
        expected = 3.0

        # Act
        actual = layer(torch.tensor(3.0))

        # Assert
        self.assertAlmostEqual(actual.item(), expected)

    def test_when_called_with_negative_value_then_returns_scaled_value(self):
        # Arrange
        layer = LeakyReLU(negative_slope=0.1)
        expected = -0.2

        # Act
        actual = layer(torch.tensor(-2.0))

        # Assert
        self.assertAlmostEqual(actual.item(), expected, places=5)

    def test_when_called_with_zero_then_returns_zero(self):
        # Arrange
        layer = LeakyReLU()
        expected = 0.0

        # Act
        actual = layer(torch.tensor(0.0))

        # Assert
        self.assertAlmostEqual(actual.item(), expected)

    def test_when_default_slope_then_negative_value_scaled_by_one_hundredth(
            self):
        # Arrange
        layer = LeakyReLU()
        expected = -0.1

        # Act
        actual = layer(torch.tensor(-10.0))

        # Assert
        self.assertAlmostEqual(actual.item(), expected, places=5)

    def test_when_called_with_tensor_then_returns_tensor(self):
        # Arrange
        x = torch.randn(5)

        # Act
        actual = LeakyReLU()(x)

        # Assert
        self.assertIsInstance(actual, torch.Tensor)

    def test_when_called_with_tensor_then_output_shape_matches_input(self):
        # Arrange
        x = torch.randn(3, 4)

        # Act
        actual = LeakyReLU()(x)

        # Assert
        self.assertEqual(actual.shape, x.shape)

    def test_when_called_with_mixed_tensor_then_negatives_are_scaled(self):
        # Arrange
        x = torch.tensor([-2.0, 0.0, 2.0])
        expected = torch.tensor([-1.0, 0.0, 2.0])

        # Act
        actual = LeakyReLU(negative_slope=0.5)(x)

        # Assert
        self.assertTrue(torch.allclose(actual, expected))


class TestSequentialForward(unittest.TestCase):

    def test_when_called_with_single_module_then_applies_it(self):
        # Arrange
        model = Sequential(ReLU())
        expected = 0.0

        # Act
        actual = model(torch.tensor(-1.0))

        # Assert
        self.assertAlmostEqual(actual.item(), expected)

    def test_when_called_with_chained_modules_then_applies_in_order(self):
        # Arrange
        x = torch.tensor([-1.0, 0.0, 1.0])
        model = Sequential(ReLU(), Sigmoid())
        expected = Sigmoid()(ReLU()(x))

        # Act
        actual = model(x)

        # Assert
        self.assertTrue(torch.allclose(actual, expected))

    def test_when_called_with_tensor_then_returns_tensor(self):
        # Arrange
        x = torch.randn(5)

        # Act
        actual = Sequential(ReLU())(x)

        # Assert
        self.assertIsInstance(actual, torch.Tensor)

    def test_when_called_with_tensor_then_output_shape_matches_input(self):
        # Arrange
        x = torch.randn(3, 4)

        # Act
        actual = Sequential(ReLU(), Tanh())(x)

        # Assert
        self.assertEqual(actual.shape, x.shape)

    def test_when_empty_then_returns_input_unchanged(self):
        # Arrange
        x = torch.tensor([1.0, 2.0, 3.0])

        # Act
        actual = Sequential()(x)

        # Assert
        self.assertTrue(torch.allclose(actual, x))


class TestSequentialAppend(unittest.TestCase):

    def test_when_module_appended_then_it_is_applied_last(self):
        # Arrange
        model = Sequential(ReLU())
        model.append(Sigmoid())
        x = torch.tensor([-1.0, 1.0])
        expected = Sigmoid()(ReLU()(x))

        # Act
        actual = model(x)

        # Assert
        self.assertTrue(torch.allclose(actual, expected))

    def test_when_module_appended_to_empty_then_it_is_applied(self):
        # Arrange
        model = Sequential()
        model.append(ReLU())
        expected = 0.0

        # Act
        actual = model(torch.tensor(-3.0))

        # Assert
        self.assertAlmostEqual(actual.item(), expected)


class TestSequentialExtend(unittest.TestCase):

    def test_when_extended_then_new_modules_are_applied_after_existing(self):
        # Arrange
        model1 = Sequential(ReLU())
        model2 = Sequential(Sigmoid(), Tanh())
        model1.extend(model2)
        x = torch.randn(5)
        expected = Tanh()(Sigmoid()(ReLU()(x)))

        # Act
        actual = model1(x)

        # Assert
        self.assertTrue(torch.allclose(actual, expected))

    def test_when_extended_with_empty_sequential_then_behavior_unchanged(self):
        # Arrange
        model = Sequential(ReLU())
        model.extend(Sequential())
        x = torch.tensor([-1.0, 1.0])
        expected = ReLU()(x)

        # Act
        actual = model(x)

        # Assert
        self.assertTrue(torch.allclose(actual, expected))


class TestSequentialInsert(unittest.TestCase):

    def test_when_inserted_at_zero_then_module_is_applied_first(self):
        # Arrange
        model = Sequential(Sigmoid())
        model.insert(0, ReLU())
        x = torch.tensor([-1.0, 1.0])
        expected = Sigmoid()(ReLU()(x))

        # Act
        actual = model(x)

        # Assert
        self.assertTrue(torch.allclose(actual, expected))

    def test_when_inserted_at_end_then_module_is_applied_last(self):
        # Arrange
        model = Sequential(ReLU())
        model.insert(1, Tanh())
        x = torch.tensor([-1.0, 1.0])
        expected = Tanh()(ReLU()(x))

        # Act
        actual = model(x)

        # Assert
        self.assertTrue(torch.allclose(actual, expected))

    def test_when_inserted_in_middle_then_order_is_correct(self):
        # Arrange
        model = Sequential(ReLU(), Sigmoid())
        model.insert(1, Tanh())
        x = torch.tensor([-1.0, 0.0, 1.0])
        expected = Sigmoid()(Tanh()(ReLU()(x)))

        # Act
        actual = model(x)

        # Assert
        self.assertTrue(torch.allclose(actual, expected))


class TestLinearInit(unittest.TestCase):

    def test_when_constructed_then_weight_shape_is_correct(self):
        # Arrange & Act
        layer = Linear(4, 3)

        # Assert
        expected = (3, 4)
        self.assertEqual(layer.weight.shape, expected)

    def test_when_bias_is_true_then_bias_shape_is_correct(self):
        # Arrange & Act
        layer = Linear(4, 3, bias=True)

        # Assert
        expected = (3,)
        self.assertEqual(layer.bias.shape, expected)

    def test_when_bias_is_false_then_bias_is_none(self):
        # Arrange & Act
        layer = Linear(4, 3, bias=False)

        # Assert
        self.assertIsNone(layer.bias)

    def test_when_constructed_then_weights_are_in_valid_range(self):
        # Arrange
        in_features = 100

        # Act
        layer = Linear(in_features, 50)

        # Assert
        sqrt_k = (1 / in_features)**0.5
        self.assertTrue(
            (layer.weight >= -sqrt_k).all() and
            (layer.weight <= sqrt_k).all())

    def test_when_bias_is_true_then_bias_values_are_in_valid_range(self):
        # Arrange
        in_features = 100

        # Act
        layer = Linear(in_features, 50, bias=True)

        # Assert
        sqrt_k = (1 / in_features)**0.5
        self.assertTrue(
            (layer.bias >= -sqrt_k).all() and (layer.bias <= sqrt_k).all())


class TestLinearForward(unittest.TestCase):

    def test_when_called_with_1d_input_then_returns_tensor(self):
        # Arrange
        x = torch.randn(3)

        # Act
        actual = Linear(3, 4)(x)

        # Assert
        self.assertIsInstance(actual, torch.Tensor)

    def test_when_called_with_1d_input_then_output_shape_is_correct(self):
        # Arrange
        x = torch.randn(3)
        expected = (4,)

        # Act
        actual = Linear(3, 4)(x)

        # Assert
        self.assertEqual(actual.shape, expected)

    def test_when_called_with_2d_input_then_output_shape_is_correct(self):
        # Arrange
        x = torch.randn(5, 3)
        expected = (5, 4)

        # Act
        actual = Linear(3, 4)(x)

        # Assert
        self.assertEqual(actual.shape, expected)

    def test_when_bias_is_false_then_output_equals_weight_matmul(self):
        # Arrange
        layer = Linear(3, 2, bias=False)
        layer.weight = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        x = torch.tensor([2.0, 3.0, 4.0])
        expected = torch.tensor([2.0, 3.0])

        # Act
        actual = layer(x)

        # Assert
        self.assertTrue(torch.allclose(actual, expected))

    def test_when_bias_is_true_then_output_equals_weight_matmul_plus_bias(
            self):
        # Arrange
        layer = Linear(2, 2, bias=True)
        layer.weight = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        layer.bias = torch.tensor([1.0, 2.0])
        x = torch.tensor([3.0, 4.0])
        expected = torch.tensor([4.0, 6.0])

        # Act
        actual = layer(x)

        # Assert
        self.assertTrue(torch.allclose(actual, expected))


if __name__ == '__main__':
    unittest.main()
