import abc

import torch


class Module(abc.ABC):
    @abc.abstractmethod
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        pass

    def __call__(self, input: torch.Tensor) -> torch.Tensor:
        return self.forward(input)


class Sigmoid(Module):
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return 1 / (1 + torch.exp(-input))


class Tanh(Module):
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        exp_pos = torch.exp(input)
        exp_neg = torch.exp(-input)
        
        return (exp_pos - exp_neg) / (exp_pos + exp_neg)


class ReLU(Module):
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return torch.where(input > 0, input, torch.zeros_like(input))


class LeakyReLU(Module):
    def __init__(self, negative_slope: float = 1e-2) -> None:
        self.negative_slope = negative_slope

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        positive = torch.where(input > 0, input, torch.zeros_like(input))
        negative = torch.where(input < 0, input, torch.zeros_like(input))
        
        return positive + self.negative_slope * negative


class Sequential(Module):
    def __init__(self, *modules: Module) -> None:
        self._modules: list[Module] = list(modules)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        output = input
        
        for module in self._modules:
            output = module(output)
        
        return output

    def append(self, module: Module) -> None:
        self._modules.append(module)

    def extend(self, sequential: "Sequential") -> None:
        self._modules.extend(sequential._modules)

    def insert(self, index: int, module: Module) -> None:
        self._modules.insert(index, module)
