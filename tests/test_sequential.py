import unittest

import torch

from dl_lib.nn import ReLU, Sequential, Sigmoid, Tanh


class TestSequentialForward(unittest.TestCase):

    def test_when_called_with_single_module_then_applies_it(self):
        # Arrange
        model = Sequential(ReLU())
        expected = 0.0

        # Act
        actual = model(torch.tensor(-1.0)).item()

        # Assert
        self.assertEqual(actual, expected)

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
        expected = torch.Tensor

        # Act
        actual = Sequential(ReLU())(x)

        # Assert
        self.assertIsInstance(actual, expected)

    def test_when_called_with_tensor_then_output_shape_matches_input(self):
        # Arrange
        x = torch.randn(3, 4)
        expected = x.shape

        # Act
        actual = Sequential(ReLU(), Tanh())(x).shape

        # Assert
        self.assertEqual(actual, expected)

    def test_when_empty_then_returns_input_unchanged(self):
        # Arrange
        x = torch.tensor([1.0, 2.0, 3.0])
        expected = x

        # Act
        actual = Sequential()(x)

        # Assert
        self.assertTrue(torch.allclose(actual, expected))


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
        actual = model(torch.tensor(-3.0)).item()

        # Assert
        self.assertEqual(actual, expected)


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


if __name__ == '__main__':
    unittest.main()
