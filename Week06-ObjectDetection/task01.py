import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn


class BinaryCNN(nn.Module):

    def __init__(self):
        super(BinaryCNN, self).__init__()

        self.feature_extractor = nn.ModuleList([
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        ])

        self.binary_classifier = nn.ModuleList([
            nn.Flatten(),
            # 64 / 2 / 2 / 2 = 8
            nn.Linear(64 * 8 * 8, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        ])

    def forward(self, input_image):
        for layer in self.feature_extractor:
            input_image = layer(input_image)

        for layer in self.binary_classifier:
            input_image = layer(input_image)

        return input_image


class MultiClassCNN(nn.Module):

    def __init__(self, num_classes):
        super(MultiClassCNN, self).__init__()

        self.feature_extractor = nn.ModuleList([
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        ])

        self.classifier = nn.ModuleList([
            nn.Flatten(),
            # 64 / 2 / 2 / 2 = 8
            nn.Linear(64 * 8 * 8, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        ])

    def forward(self, input_image):
        for layer in self.feature_extractor:
            input_image = layer(input_image)

        for layer in self.classifier:
            input_image = layer(input_image)

        return input_image


def count_parameters(model):
    return sum(param.numel() for param in model.parameters())


def main():
    binary_net = BinaryCNN()

    num_classes = 4
    multiclass_net = MultiClassCNN(num_classes)

    print(
        f"Number of parameters in CNN for binary classification: {count_parameters(binary_net)}"
    )
    print(
        f"Number of parameters in CNN for multiclass classification ({num_classes} classes): {count_parameters(multiclass_net)}"
    )

    new_module = nn.ModuleList([
        # in_cnannels == out channels in order to not modify the classifier
        nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
        nn.ReLU()
    ])

    binary_net.add_module("new_module", new_module)

    print(
        f"Number of parameters in CNN for binary classification with new module: {count_parameters(binary_net)}"
    )
    print(binary_net)

    # in order for the new_module to take place between the other two modules, we should tweak the `forward` function


if __name__ == '__main__':
    main()
