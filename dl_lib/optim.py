import abc

import torch


class Optimizer(abc.ABC):

    def __init__(self, parameters: list[torch.Tensor]) -> None:
        self.parameters = list(parameters)

    @abc.abstractmethod
    def step(self) -> None:
        pass

    def zero_grad(self) -> None:
        for parameter in self.parameters:
            if parameter.grad is not None:
                parameter.grad.zero_()


class SGD(Optimizer):

    def __init__(self,
                 parameters: list[torch.Tensor],
                 lr: float = 0.01,
                 momentum: float = 0.0) -> None:
        super().__init__(parameters)
        self.lr = lr
        self.momentum = momentum
        self.velocities = [
            torch.zeros_like(parameter) for parameter in self.parameters
        ]

    def step(self) -> None:
        for i, parameter in enumerate(self.parameters):
            if parameter.grad is None:
                continue
            self.velocities[i] = (self.momentum * self.velocities[i] -
                                  self.lr * parameter.grad)
            parameter.data += self.velocities[i]


class AdaGrad(Optimizer):

    def __init__(self,
                 parameters: list[torch.Tensor],
                 lr: float = 0.01,
                 eps: float = 1e-8) -> None:
        super().__init__(parameters)
        self.lr = lr
        self.eps = eps
        self.sum_of_squared_gradients = [
            torch.zeros_like(parameter) for parameter in self.parameters
        ]

    def step(self) -> None:
        for i, parameter in enumerate(self.parameters):
            if parameter.grad is None:
                continue
            self.sum_of_squared_gradients[i] = (
                self.sum_of_squared_gradients[i] + parameter.grad**2)
            parameter.data -= (
                self.lr * parameter.grad /
                (self.eps + self.sum_of_squared_gradients[i]**0.5))


class RMSprop(Optimizer):

    def __init__(self,
                 parameters: list[torch.Tensor],
                 lr: float = 0.01,
                 alpha: float = 0.99,
                 eps: float = 1e-8) -> None:
        super().__init__(parameters)
        self.lr = lr
        self.alpha = alpha
        self.eps = eps
        self.squared_average = [
            torch.zeros_like(parameter) for parameter in self.parameters
        ]

    def step(self) -> None:
        for i, parameter in enumerate(self.parameters):
            if parameter.grad is None:
                continue
            self.squared_average[i] = (self.alpha * self.squared_average[i] +
                                       (1 - self.alpha) * parameter.grad**2)
            parameter.data -= (self.lr * parameter.grad /
                               (self.eps + self.squared_average[i]**0.5))


class Adam(Optimizer):

    def __init__(self,
                 parameters: list[torch.Tensor],
                 lr: float = 0.001,
                 beta1: float = 0.9,
                 beta2: float = 0.999,
                 eps: float = 1e-8) -> None:
        super().__init__(parameters)
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.m = [torch.zeros_like(parameter) for parameter in self.parameters]
        self.v = [torch.zeros_like(parameter) for parameter in self.parameters]
        self.t = 0

    def step(self) -> None:
        self.t += 1

        for i, parameter in enumerate(self.parameters):
            if parameter.grad is None:
                continue
            self.m[i] = self.beta1 * self.m[i] + (1 -
                                                  self.beta1) * parameter.grad
            self.v[i] = self.beta2 * self.v[i] + (
                1 - self.beta2) * parameter.grad**2
            m_hat = self.m[i] / (1 - self.beta1**self.t)
            v_hat = self.v[i] / (1 - self.beta2**self.t)
            parameter.data -= self.lr * m_hat / (self.eps + v_hat**0.5)


class AdamW(Optimizer):

    def __init__(self,
                 parameters: list[torch.Tensor],
                 lr: float = 0.001,
                 beta1: float = 0.9,
                 beta2: float = 0.999,
                 eps: float = 1e-8,
                 weight_decay: float = 0.01) -> None:
        super().__init__(parameters)
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weight_decay = weight_decay
        self.m = [torch.zeros_like(parameter) for parameter in self.parameters]
        self.v = [torch.zeros_like(parameter) for parameter in self.parameters]
        self.t = 0

    def step(self) -> None:
        self.t += 1

        for i, parameter in enumerate(self.parameters):
            if parameter.grad is None:
                continue
            parameter.data -= self.lr * self.weight_decay * parameter.data
            self.m[i] = self.beta1 * self.m[i] + (1 -
                                                  self.beta1) * parameter.grad
            self.v[i] = self.beta2 * self.v[i] + (
                1 - self.beta2) * parameter.grad**2
            m_hat = self.m[i] / (1 - self.beta1**self.t)
            v_hat = self.v[i] / (1 - self.beta2**self.t)
            parameter.data -= self.lr * m_hat / (v_hat**0.5 + self.eps)
