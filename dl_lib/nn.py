import abc

import torch


class Module(abc.ABC):

    @abc.abstractmethod
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        pass

    def __call__(self, *args, **kwargs) -> torch.Tensor:
        return self.forward(*args, **kwargs)


class Sigmoid(Module):

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return 1 / (1 + torch.exp(-input))


class Tanh(Module):

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        exp_positive = torch.exp(input)
        exp_negative = torch.exp(-input)

        return (exp_positive - exp_negative) / (exp_positive + exp_negative)


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


class Linear(Module):

    def __init__(self,
                 in_features: int,
                 out_features: int,
                 bias: bool = True) -> None:
        if in_features is None:
            raise ValueError("in_features must be specified")
        self.in_features = in_features

        if out_features is None:
            raise ValueError("out_features must be specified")
        self.out_features = out_features

        sqrt_k = (1 / in_features)**0.5
        self.weight = torch.empty(out_features,
                                  in_features).uniform_(-sqrt_k, sqrt_k)
        self.bias = torch.empty(out_features).uniform_(
            -sqrt_k, sqrt_k) if bias else None

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        output = input @ self.weight.T

        if self.bias is not None:
            output = output + self.bias

        return output


class Softmax(Module):

    def __init__(self, dim: int = None) -> None:
        self.dim = dim

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if self.dim is None:
            raise TypeError("Dim should not be None")

        exp_input = torch.exp(input)

        return exp_input / exp_input.sum(dim=self.dim, keepdim=True)


class Dropout(Module):

    def __init__(self, p: float = 0.5) -> None:
        self.p = p

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if self.p == 0.0:
            return input

        mask = torch.rand_like(input) >= self.p

        return input * mask / (1 - self.p)


class BCEWithLogitsLoss(Module):

    def __init__(self, reduction: str = 'mean', pos_weight=None) -> None:
        self.reduction = reduction
        self.pos_weight = pos_weight

    def forward(self, input: torch.Tensor,
                target: torch.Tensor) -> torch.Tensor:
        probabilities = Sigmoid()(input)
        weight = self.pos_weight if self.pos_weight is not None else 1.0

        loss = -(weight * target * torch.log(probabilities) +
                 (1 - target) * torch.log(1 - probabilities))

        if self.reduction == 'none':
            return loss
        if self.reduction == 'mean':
            return loss.mean()
        if self.reduction == 'sum':
            return loss.sum()


class CrossEntropyLoss(Module):

    def __init__(self, reduction: str = 'mean') -> None:
        self.reduction = reduction

    def forward(self, input: torch.Tensor,
                target: torch.Tensor) -> torch.Tensor:
        probabilities = Softmax(dim=-1)(input)
        correct_probabilities = probabilities[torch.arange(input.shape[0]),
                                              target]
        loss = -torch.log(correct_probabilities)

        if self.reduction == 'none':
            return loss
        if self.reduction == 'mean':
            return loss.mean()
        if self.reduction == 'sum':
            return loss.sum()
