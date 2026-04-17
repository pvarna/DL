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


class _Pool2d(Module):

    def __init__(self, kernel_size: tuple[int, int] | int,
                 stride: tuple[int, int] | int, padding: tuple[int, int] | int,
                 padding_fill_value: float, pooling_function) -> None:
        if kernel_size is None:
            raise ValueError("kernel_size must be specified")
        self.kernel_size = kernel_size

        if stride is None:
            self.stride = kernel_size
        else:
            self.stride = stride

        if padding is None:
            self.padding = 0
        else:
            self.padding = padding

        self._padding_fill_value = padding_fill_value
        self._pooling_function = pooling_function

    def _get_kernel_size_value(self) -> tuple[int, int]:
        return self.kernel_size if isinstance(
            self.kernel_size, tuple) else (self.kernel_size, self.kernel_size)

    def _get_stride_value(self) -> tuple[int, int]:
        return self.stride if isinstance(self.stride, tuple) else (self.stride,
                                                                   self.stride)

    def _get_padding_value(self) -> tuple[int, int]:
        return self.padding if isinstance(
            self.padding, tuple) else (self.padding, self.padding)

    def _add_padding(self, input: torch.Tensor) -> torch.Tensor:
        padding_height, padding_width = self._get_padding_value()

        if padding_height == 0 and padding_width == 0:
            return input

        batch_size, channels, height, width = input.shape
        padded_height = height + 2 * padding_height
        padded_width = width + 2 * padding_width

        padded_input = torch.full(
            (batch_size, channels, padded_height, padded_width),
            fill_value=self._padding_fill_value,
            dtype=input.dtype,
            device=input.device)

        padded_input[:, :, padding_height:padding_height + height,
                     padding_width:padding_width + width] = input

        return padded_input

    def _get_output_dimensions(
            self, input: torch.Tensor) -> tuple[int, int, int, int]:
        batch_size, channels, height, width = input.shape

        kernel_height, kernel_width = self._get_kernel_size_value()
        stride_height, stride_width = self._get_stride_value()

        output_height = (height - kernel_height) // stride_height + 1
        output_width = (width - kernel_width) // stride_width + 1

        return batch_size, channels, output_height, output_width

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        input = self._add_padding(input)
        batch_size, channels, output_height, output_width = self._get_output_dimensions(
            input)

        kernel_height, kernel_width = self._get_kernel_size_value()
        stride_height, stride_width = self._get_stride_value()

        output = torch.empty(
            (batch_size, channels, output_height, output_width),
            dtype=input.dtype,
            device=input.device)

        for b in range(batch_size):
            for c in range(channels):
                for i in range(output_height):
                    for j in range(output_width):
                        height_start = i * stride_height
                        height_end = height_start + kernel_height
                        width_start = j * stride_width
                        width_end = width_start + kernel_width

                        output[b, c, i, j] = self._pooling_function(
                            input[b, c, height_start:height_end,
                                  width_start:width_end])

        return output


class MaxPool2d(_Pool2d):

    def __init__(self, kernel_size: tuple[int, int] | int,
                 stride: tuple[int, int] | int,
                 padding: tuple[int, int] | int) -> None:
        super().__init__(kernel_size, stride, padding, float('-inf'),
                         torch.max)


class AvgPool2d(_Pool2d):

    def __init__(self, kernel_size: tuple[int, int] | int,
                 stride: tuple[int, int] | int,
                 padding: tuple[int, int] | int) -> None:
        super().__init__(kernel_size, stride, padding, 0.0, torch.mean)
