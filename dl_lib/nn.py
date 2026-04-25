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
                 stride: tuple[int, int] | int = None,
                 padding: tuple[int, int] | int = None) -> None:
        super().__init__(kernel_size, stride, padding, float('-inf'),
                         torch.max)


class AvgPool2d(_Pool2d):

    def __init__(self, kernel_size: tuple[int, int] | int,
                 stride: tuple[int, int] | int = None,
                 padding: tuple[int, int] | int = None) -> None:
        super().__init__(kernel_size, stride, padding, 0.0, torch.mean)


class Conv1d(Module):

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int,
                 stride: int = 1,
                 padding: str | int | tuple[int, int] = 0,
                 bias: bool = True) -> None:
        if in_channels is None:
            raise ValueError("in_channels must be specified")
        self.in_channels = in_channels

        if out_channels is None:
            raise ValueError("out_channels must be specified")
        self.out_channels = out_channels

        if kernel_size is None:
            raise ValueError("kernel_size must be specified")
        self.kernel_size = kernel_size

        self.stride = stride
        self.padding = padding

        sqrt_k = (1 / (in_channels * kernel_size))**0.5
        self.weight = torch.empty(out_channels, in_channels,
                                  kernel_size).uniform_(-sqrt_k, sqrt_k)
        self.bias = torch.empty(out_channels).uniform_(
            -sqrt_k, sqrt_k) if bias else None

    def _get_padding_size(self) -> tuple[int, int]:
        if self.padding == 'valid':
            return (0, 0)

        if self.padding == 'same':
            total_padding = self.kernel_size - 1
            left_padding = total_padding // 2

            return (left_padding, total_padding - left_padding)

        if isinstance(self.padding, tuple):
            return self.padding

        return (self.padding, self.padding)

    def _add_padding(self, input: torch.Tensor) -> torch.Tensor:
        batch_size, in_channels, length = input.shape
        left_padding, right_padding = self._get_padding_size()

        if left_padding == 0 and right_padding == 0:
            return input

        padded_length = length + left_padding + right_padding
        padded_input = torch.zeros((batch_size, in_channels, padded_length),
                                   dtype=input.dtype,
                                   device=input.device)

        padded_input[:, :, left_padding:left_padding + length] = input

        return padded_input

    def _get_output_length(self, input: torch.Tensor) -> int:
        length = input.shape[2]

        return (length - self.kernel_size) // self.stride + 1

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        input = self._add_padding(input)
        batch_size = input.shape[0]
        output_length = self._get_output_length(input)

        output = torch.empty((batch_size, self.out_channels, output_length),
                             dtype=input.dtype,
                             device=input.device)

        for b in range(batch_size):
            for o in range(self.out_channels):
                for i in range(output_length):
                    length_start = i * self.stride
                    length_end = length_start + self.kernel_size

                    output[b, o, i] = torch.sum(
                        self.weight[o] * input[b, :, length_start:length_end])

        if self.bias is not None:
            output = output + self.bias[None, :, None]

        return output


class Conv2d(Module):

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: tuple[int, int] | int,
                 stride: tuple[int, int] | int = 1,
                 padding: str | int | tuple[int, int] = 0,
                 bias: bool = True) -> None:
        if in_channels is None:
            raise ValueError("in_channels must be specified")
        self.in_channels = in_channels

        if out_channels is None:
            raise ValueError("out_channels must be specified")
        self.out_channels = out_channels

        if kernel_size is None:
            raise ValueError("kernel_size must be specified")
        self.kernel_size = kernel_size

        self.stride = stride
        self.padding = padding

        kernel_height, kernel_width = self._get_kernel_size_value()
        sqrt_k = (1 / (in_channels * kernel_height * kernel_width))**0.5
        self.weight = torch.empty(out_channels, in_channels, kernel_height,
                                  kernel_width).uniform_(-sqrt_k, sqrt_k)
        self.bias = torch.empty(out_channels).uniform_(
            -sqrt_k, sqrt_k) if bias else None

    def _get_kernel_size_value(self) -> tuple[int, int]:
        return self.kernel_size if isinstance(
            self.kernel_size, tuple) else (self.kernel_size, self.kernel_size)

    def _get_stride_value(self) -> tuple[int, int]:
        return self.stride if isinstance(self.stride, tuple) else (self.stride,
                                                                   self.stride)

    def _get_padding_size(self) -> tuple[int, int, int, int]:
        if self.padding == 'valid':
            return (0, 0, 0, 0)

        kernel_height, kernel_width = self._get_kernel_size_value()

        if self.padding == 'same':
            total_height_padding = kernel_height - 1
            top_padding = total_height_padding // 2
            bottom_padding = total_height_padding - top_padding

            total_width_padding = kernel_width - 1
            left_padding = total_width_padding // 2
            right_padding = total_width_padding - left_padding

            return (top_padding, bottom_padding, left_padding, right_padding)

        if isinstance(self.padding, tuple):
            height_padding, width_padding = self.padding

            return (height_padding, height_padding, width_padding,
                    width_padding)

        return (self.padding, self.padding, self.padding, self.padding)

    def _add_padding(self, input: torch.Tensor) -> torch.Tensor:
        batch_size, in_channels, height, width = input.shape
        top_padding, bottom_padding, left_padding, right_padding = self._get_padding_size(
        )

        if top_padding == 0 and bottom_padding == 0 and left_padding == 0 and right_padding == 0:
            return input

        padded_height = height + top_padding + bottom_padding
        padded_width = width + left_padding + right_padding

        padded_input = torch.zeros(
            (batch_size, in_channels, padded_height, padded_width),
            dtype=input.dtype,
            device=input.device)

        padded_input[:, :, top_padding:top_padding + height,
                     left_padding:left_padding + width] = input

        return padded_input

    def _get_output_dimensions(self, input: torch.Tensor) -> tuple[int, int]:
        height, width = input.shape[2], input.shape[3]

        kernel_height, kernel_width = self._get_kernel_size_value()
        stride_height, stride_width = self._get_stride_value()

        output_height = (height - kernel_height) // stride_height + 1
        output_width = (width - kernel_width) // stride_width + 1

        return output_height, output_width

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        input = self._add_padding(input)
        batch_size = input.shape[0]
        output_height, output_width = self._get_output_dimensions(input)

        kernel_height, kernel_width = self._get_kernel_size_value()
        stride_height, stride_width = self._get_stride_value()

        output = torch.empty(
            (batch_size, self.out_channels, output_height, output_width),
            dtype=input.dtype,
            device=input.device)

        for b in range(batch_size):
            for o in range(self.out_channels):
                for i in range(output_height):
                    for j in range(output_width):
                        height_start = i * stride_height
                        height_end = height_start + kernel_height
                        width_start = j * stride_width
                        width_end = width_start + kernel_width

                        output[b, o, i, j] = torch.sum(
                            self.weight[o] *
                            input[b, :, height_start:height_end,
                                  width_start:width_end])

        if self.bias is not None:
            output = output + self.bias[None, :, None, None]

        return output


class Conv3d(Module):

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: tuple[int, int, int] | int,
                 stride: tuple[int, int, int] | int = 1,
                 padding: str | int | tuple[int, int, int] = 0,
                 bias: bool = True) -> None:
        if in_channels is None:
            raise ValueError("in_channels must be specified")
        self.in_channels = in_channels

        if out_channels is None:
            raise ValueError("out_channels must be specified")
        self.out_channels = out_channels

        if kernel_size is None:
            raise ValueError("kernel_size must be specified")
        self.kernel_size = kernel_size

        self.stride = stride
        self.padding = padding

        kernel_depth, kernel_height, kernel_width = self._get_kernel_size_value(
        )
        sqrt_k = (
            1 /
            (in_channels * kernel_depth * kernel_height * kernel_width))**0.5
        self.weight = torch.empty(out_channels, in_channels, kernel_depth,
                                  kernel_height,
                                  kernel_width).uniform_(-sqrt_k, sqrt_k)
        self.bias = torch.empty(out_channels).uniform_(
            -sqrt_k, sqrt_k) if bias else None

    def _get_kernel_size_value(self) -> tuple[int, int, int]:
        return self.kernel_size if isinstance(
            self.kernel_size, tuple) else (self.kernel_size, self.kernel_size,
                                           self.kernel_size)

    def _get_stride_value(self) -> tuple[int, int, int]:
        return self.stride if isinstance(self.stride, tuple) else (self.stride,
                                                                   self.stride,
                                                                   self.stride)

    def _get_padding_size(self) -> tuple[int, int, int, int, int, int]:
        if self.padding == 'valid':
            return (0, 0, 0, 0, 0, 0)

        kernel_depth, kernel_height, kernel_width = self._get_kernel_size_value(
        )

        if self.padding == 'same':
            total_depth_padding = kernel_depth - 1
            front_padding = total_depth_padding // 2
            back_padding = total_depth_padding - front_padding

            total_height_padding = kernel_height - 1
            top_padding = total_height_padding // 2
            bottom_padding = total_height_padding - top_padding

            total_width_padding = kernel_width - 1
            left_padding = total_width_padding // 2
            right_padding = total_width_padding - left_padding

            return (front_padding, back_padding, top_padding, bottom_padding,
                    left_padding, right_padding)

        if isinstance(self.padding, tuple):
            depth_padding, height_padding, width_padding = self.padding

            return (depth_padding, depth_padding, height_padding,
                    height_padding, width_padding, width_padding)

        return (self.padding, self.padding, self.padding, self.padding,
                self.padding, self.padding)

    def _add_padding(self, input: torch.Tensor) -> torch.Tensor:
        batch_size, in_channels, depth, height, width = input.shape
        front_padding, back_padding, top_padding, bottom_padding, left_padding, right_padding = self._get_padding_size(
        )

        if front_padding == 0 and back_padding == 0 and top_padding == 0 and bottom_padding == 0 and left_padding == 0 and right_padding == 0:
            return input

        padded_depth = depth + front_padding + back_padding
        padded_height = height + top_padding + bottom_padding
        padded_width = width + left_padding + right_padding

        padded_input = torch.zeros((batch_size, in_channels, padded_depth,
                                    padded_height, padded_width),
                                   dtype=input.dtype,
                                   device=input.device)

        padded_input[:, :, front_padding:front_padding + depth,
                     top_padding:top_padding + height,
                     left_padding:left_padding + width] = input

        return padded_input

    def _get_output_dimensions(self,
                               input: torch.Tensor) -> tuple[int, int, int]:
        depth, height, width = input.shape[2], input.shape[3], input.shape[4]

        kernel_depth, kernel_height, kernel_width = self._get_kernel_size_value(
        )
        stride_depth, stride_height, stride_width = self._get_stride_value()

        output_depth = (depth - kernel_depth) // stride_depth + 1
        output_height = (height - kernel_height) // stride_height + 1
        output_width = (width - kernel_width) // stride_width + 1

        return output_depth, output_height, output_width

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        input = self._add_padding(input)
        batch_size = input.shape[0]
        output_depth, output_height, output_width = self._get_output_dimensions(
            input)

        kernel_depth, kernel_height, kernel_width = self._get_kernel_size_value(
        )
        stride_depth, stride_height, stride_width = self._get_stride_value()

        output = torch.empty((batch_size, self.out_channels, output_depth,
                              output_height, output_width),
                             dtype=input.dtype,
                             device=input.device)

        for b in range(batch_size):
            for o in range(self.out_channels):
                for d in range(output_depth):
                    for i in range(output_height):
                        for j in range(output_width):
                            depth_start = d * stride_depth
                            depth_end = depth_start + kernel_depth
                            height_start = i * stride_height
                            height_end = height_start + kernel_height
                            width_start = j * stride_width
                            width_end = width_start + kernel_width

                            output[b, o, d, i, j] = torch.sum(
                                self.weight[o] *
                                input[b, :, depth_start:depth_end,
                                      height_start:height_end,
                                      width_start:width_end])

        if self.bias is not None:
            output = output + self.bias[None, :, None, None, None]

        return output
