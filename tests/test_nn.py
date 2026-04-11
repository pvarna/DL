import unittest

import torch

from dl_lib.nn import LeakyReLU, ReLU, Sequential, Sigmoid, Tanh


class TestSigmoidForward(unittest.TestCase):

    def test_when_called_with_zero_then_returns_half(self):
        result = Sigmoid()(torch.tensor(0.0))
        self.assertAlmostEqual(result.item(), 0.5)

    def test_when_called_with_large_positive_then_returns_close_to_one(self):
        result = Sigmoid()(torch.tensor(100.0))
        self.assertAlmostEqual(result.item(), 1.0, places=5)

    def test_when_called_with_large_negative_then_returns_close_to_zero(self):
        result = Sigmoid()(torch.tensor(-100.0))
        self.assertAlmostEqual(result.item(), 0.0, places=5)

    def test_when_called_with_tensor_then_returns_tensor(self):
        result = Sigmoid()(torch.randn(5))
        self.assertIsInstance(result, torch.Tensor)

    def test_when_called_with_tensor_then_output_shape_matches_input(self):
        x = torch.randn(3, 4)
        result = Sigmoid()(x)
        self.assertEqual(result.shape, x.shape)

    def test_when_called_with_tensor_then_all_values_in_zero_one_range(self):
        result = Sigmoid()(torch.randn(100))
        self.assertTrue((result >= 0).all() and (result <= 1).all())


class TestTanhForward(unittest.TestCase):

    def test_when_called_with_zero_then_returns_zero(self):
        result = Tanh()(torch.tensor(0.0))
        self.assertAlmostEqual(result.item(), 0.0)

    def test_when_called_with_tensor_then_returns_tensor(self):
        result = Tanh()(torch.randn(5))
        self.assertIsInstance(result, torch.Tensor)

    def test_when_called_with_tensor_then_output_shape_matches_input(self):
        x = torch.randn(3, 4)
        result = Tanh()(x)
        self.assertEqual(result.shape, x.shape)

    def test_when_called_with_tensor_then_all_values_in_minus_one_one_range(
            self):
        result = Tanh()(torch.randn(100))
        self.assertTrue((result >= -1).all() and (result <= 1).all())


class TestReLUForward(unittest.TestCase):

    def test_when_called_with_positive_value_then_returns_same_value(self):
        result = ReLU()(torch.tensor(3.0))
        self.assertAlmostEqual(result.item(), 3.0)

    def test_when_called_with_negative_value_then_returns_zero(self):
        result = ReLU()(torch.tensor(-5.0))
        self.assertAlmostEqual(result.item(), 0.0)

    def test_when_called_with_zero_then_returns_zero(self):
        result = ReLU()(torch.tensor(0.0))
        self.assertAlmostEqual(result.item(), 0.0)

    def test_when_called_with_tensor_then_returns_tensor(self):
        result = ReLU()(torch.randn(5))
        self.assertIsInstance(result, torch.Tensor)

    def test_when_called_with_tensor_then_output_shape_matches_input(self):
        x = torch.randn(3, 4)
        result = ReLU()(x)
        self.assertEqual(result.shape, x.shape)

    def test_when_called_with_mixed_tensor_then_negatives_become_zero(self):
        x = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
        result = ReLU()(x)
        expected = torch.tensor([0.0, 0.0, 0.0, 1.0, 2.0])
        self.assertTrue(torch.allclose(result, expected))


class TestLeakyReLUForward(unittest.TestCase):

    def test_when_called_with_positive_value_then_returns_same_value(self):
        result = LeakyReLU()(torch.tensor(3.0))
        self.assertAlmostEqual(result.item(), 3.0)

    def test_when_called_with_negative_value_then_returns_scaled_value(self):
        result = LeakyReLU(negative_slope=0.1)(torch.tensor(-2.0))
        self.assertAlmostEqual(result.item(), -0.2, places=5)

    def test_when_called_with_zero_then_returns_zero(self):
        result = LeakyReLU()(torch.tensor(0.0))
        self.assertAlmostEqual(result.item(), 0.0)

    def test_when_default_slope_then_negative_value_scaled_by_one_hundredth(
            self):
        result = LeakyReLU()(torch.tensor(-10.0))
        self.assertAlmostEqual(result.item(), -0.1, places=5)

    def test_when_called_with_tensor_then_returns_tensor(self):
        result = LeakyReLU()(torch.randn(5))
        self.assertIsInstance(result, torch.Tensor)

    def test_when_called_with_tensor_then_output_shape_matches_input(self):
        x = torch.randn(3, 4)
        result = LeakyReLU()(x)
        self.assertEqual(result.shape, x.shape)

    def test_when_called_with_mixed_tensor_then_negatives_are_scaled(self):
        x = torch.tensor([-2.0, 0.0, 2.0])
        result = LeakyReLU(negative_slope=0.5)(x)
        expected = torch.tensor([-1.0, 0.0, 2.0])
        self.assertTrue(torch.allclose(result, expected))


class TestSequentialForward(unittest.TestCase):

    def test_when_called_with_single_module_then_applies_it(self):
        model = Sequential(ReLU())
        result = model(torch.tensor(-1.0))
        self.assertAlmostEqual(result.item(), 0.0)

    def test_when_called_with_chained_modules_then_applies_in_order(self):
        x = torch.tensor([-1.0, 0.0, 1.0])
        model = Sequential(ReLU(), Sigmoid())
        result = model(x)
        expected = Sigmoid()(ReLU()(x))
        self.assertTrue(torch.allclose(result, expected))

    def test_when_called_with_tensor_then_returns_tensor(self):
        result = Sequential(ReLU())(torch.randn(5))
        self.assertIsInstance(result, torch.Tensor)

    def test_when_called_with_tensor_then_output_shape_matches_input(self):
        x = torch.randn(3, 4)
        result = Sequential(ReLU(), Tanh())(x)
        self.assertEqual(result.shape, x.shape)

    def test_when_empty_then_returns_input_unchanged(self):
        x = torch.tensor([1.0, 2.0, 3.0])
        result = Sequential()(x)
        self.assertTrue(torch.allclose(result, x))


class TestSequentialAppend(unittest.TestCase):

    def test_when_module_appended_then_it_is_applied_last(self):
        model = Sequential(ReLU())
        model.append(Sigmoid())
        x = torch.tensor([-1.0, 1.0])
        result = model(x)
        expected = Sigmoid()(ReLU()(x))
        self.assertTrue(torch.allclose(result, expected))

    def test_when_module_appended_to_empty_then_it_is_applied(self):
        model = Sequential()
        model.append(ReLU())
        result = model(torch.tensor(-3.0))
        self.assertAlmostEqual(result.item(), 0.0)


class TestSequentialExtend(unittest.TestCase):

    def test_when_extended_then_new_modules_are_applied_after_existing(self):
        model1 = Sequential(ReLU())
        model2 = Sequential(Sigmoid(), Tanh())
        model1.extend(model2)
        x = torch.randn(5)
        result = model1(x)
        expected = Tanh()(Sigmoid()(ReLU()(x)))
        self.assertTrue(torch.allclose(result, expected))

    def test_when_extended_with_empty_sequential_then_behavior_unchanged(self):
        model = Sequential(ReLU())
        model.extend(Sequential())
        x = torch.tensor([-1.0, 1.0])
        result = model(x)
        expected = ReLU()(x)
        self.assertTrue(torch.allclose(result, expected))


class TestSequentialInsert(unittest.TestCase):

    def test_when_inserted_at_zero_then_module_is_applied_first(self):
        model = Sequential(Sigmoid())
        model.insert(0, ReLU())
        x = torch.tensor([-1.0, 1.0])
        result = model(x)
        expected = Sigmoid()(ReLU()(x))
        self.assertTrue(torch.allclose(result, expected))

    def test_when_inserted_at_end_then_module_is_applied_last(self):
        model = Sequential(ReLU())
        model.insert(1, Tanh())
        x = torch.tensor([-1.0, 1.0])
        result = model(x)
        expected = Tanh()(ReLU()(x))
        self.assertTrue(torch.allclose(result, expected))

    def test_when_inserted_in_middle_then_order_is_correct(self):
        model = Sequential(ReLU(), Sigmoid())
        model.insert(1, Tanh())
        x = torch.tensor([-1.0, 0.0, 1.0])
        result = model(x)
        expected = Sigmoid()(Tanh()(ReLU()(x)))
        self.assertTrue(torch.allclose(result, expected))


if __name__ == '__main__':
    unittest.main()
