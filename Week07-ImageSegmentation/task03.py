import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn


class MyUNet(nn.Module):

    def __init__(self, input_channels, output_channels):
        super().__init__()

        self.initial_conv_layer = self.my_convolutional_block(
            input_channels, 64)

        self.encoder = nn.ModuleList([
            self.my_convolutional_block(64, 128),
            self.my_convolutional_block(128, 256),
            self.my_convolutional_block(256, 512)
        ])

        self.pooling_layers = nn.ModuleList([
            nn.MaxPool2d(kernel_size=2),
            nn.MaxPool2d(kernel_size=2),
            nn.MaxPool2d(kernel_size=2)
        ])

        self.upconv_layers = nn.ModuleList([
            nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2),
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        ])

        self.decoder = nn.ModuleList([
            self.my_convolutional_block(512, 256),
            self.my_convolutional_block(256, 128),
            self.my_convolutional_block(128, 64)
        ])

        self.final_conv_layer = nn.Conv2d(64, output_channels, kernel_size=1)

    def forward(self, x):
        skip_connections = []

        x = self.initial_conv_layer(x)

        for encoder_block, pool in zip(self.encoder, self.pooling_layers):
            skip_connections.append(x)
            x = pool(x)
            x = encoder_block(x)

        for upconv, decoder_block in zip(self.upconv_layers, self.decoder):
            x = upconv(x)
            skip_x = skip_connections.pop()
            x = torch.cat((x, skip_x), dim=1)
            x = decoder_block(x)

        x = self.final_conv_layer(x)
        return x

    def my_convolutional_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True))


def main():
    input_channels = 16
    output_channels = 5
    net = MyUNet(input_channels, output_channels)

    print(
        f"Total number of parameters in UNet({input_channels}, {output_channels}): {sum(p.numel() for p in net.parameters()):,}"
    )


if __name__ == '__main__':
    main()
