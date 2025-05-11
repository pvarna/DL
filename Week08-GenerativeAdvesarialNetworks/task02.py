import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class LinearGenerator(nn.Module):

    def __init__(self, input_dimensions, output_dimensions):
        super().__init__()

        self.generator = nn.Sequential(self.gen_block(input_dimensions, 256),
                                       self.gen_block(256, 512),
                                       self.gen_block(512, 1024),
                                       nn.Linear(1024, output_dimensions),
                                       nn.Sigmoid())

    def gen_block(self, input_dimensions, output_dimensions):
        return nn.Sequential(nn.Linear(input_dimensions, output_dimensions),
                             nn.BatchNorm1d(output_dimensions), nn.ReLU())

    def forward(self, x):
        return self.generator(x)


class LinearDiscriminator(nn.Module):

    def __init__(self, input_dimensions):
        super().__init__()

        self.discriminator = nn.Sequential(
            self.disc_block(input_dimensions, 1024),
            self.disc_block(1024, 512), self.disc_block(512, 256),
            nn.Linear(256, 1))

    def disc_block(self, input_dimensions, output_dimensions):
        return nn.Sequential(nn.Linear(input_dimensions, output_dimensions),
                             nn.LeakyReLU(0.2))

    def forward(self, x):
        return self.discriminator(x)


class ConvolutionalGenerator(nn.Module):

    def __init__(self, input_dimensions, output_dimensions):
        super().__init__()

        self.generator = nn.Sequential(
            self.gen_block(input_dimensions, 256), self.gen_block(256, 512),
            self.gen_block(512, 1024),
            nn.ConvTranspose2d(1024,
                               output_dimensions,
                               kernel_size=1,
                               stride=1), nn.Sigmoid())

    def gen_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels,
                               out_channels,
                               kernel_size=1,
                               stride=1), nn.BatchNorm2d(out_channels),
            nn.ReLU())

    def forward(self, x):
        x = x.unsqueeze(-1).unsqueeze(-1)
        x = self.generator(x)
        return x.view(x.size(0), -1)


class ConvolutionalDiscriminator(nn.Module):

    def __init__(self, input_dimensions):
        super().__init__()

        self.discriminator = nn.Sequential(
            self.disc_block(input_dimensions, 1024),
            self.disc_block(1024, 512), self.disc_block(512, 256),
            nn.Conv2d(256, 1, kernel_size=1, stride=1))

    def disc_block(self, input_dimensions, output_dimensions):
        return nn.Sequential(
            nn.Conv2d(input_dimensions,
                      output_dimensions,
                      kernel_size=1,
                      stride=1), nn.LeakyReLU(0.2))

    def forward(self, x):
        x = x.unsqueeze(-1).unsqueeze(-1)
        x = self.discriminator(x)
        return x.view(x.size(0), -1)


class GCGANGenerator(nn.Module):

    def __init__(self, input_dimensions):
        super().__init__()

        self.generator = nn.Sequential(
            self.dc_gen_block(input_dimensions, 1024),
            self.dc_gen_block(1024, 512), self.dc_gen_block(512, 256),
            nn.ConvTranspose2d(256, 3, kernel_size=4, stride=2), nn.Tanh())

    def forward(self, x):
        return self.generator(x)

    def dc_gen_block(self, input_dimensions, output_dimensions):
        return nn.Sequential(
            nn.ConvTranspose2d(input_dimensions,
                               output_dimensions,
                               kernel_size=4,
                               stride=2), nn.BatchNorm2d(output_dimensions),
            nn.ReLU())


class GCGANDiscriminator(nn.Module):

    def __init__(self):
        super().__init__()

        self.discriminator = nn.Sequential(
            self.dc_disc_block(3, 256), self.dc_disc_block(256, 512),
            self.dc_disc_block(512, 1024),
            nn.Conv2d(1024, 1, kernel_size=4, stride=2))

    def dc_disc_block(self, input_dimensions, output_dimensions):
        return nn.Sequential(
            nn.Conv2d(input_dimensions,
                      output_dimensions,
                      kernel_size=4,
                      stride=2), nn.BatchNorm2d(output_dimensions),
            nn.LeakyReLU(0.2))

    def forward(self, x):
        return self.discriminator(x)


def main():
    linear_generator = LinearGenerator(5, 16)
    print(
        f"Number of total parameters in Generator (linear only): {count_parameters(linear_generator):,}."
    )

    linear_discriminator = LinearDiscriminator(16)
    print(
        f"Number of total parameters in Discriminator (linear only): {count_parameters(linear_discriminator):,}"
    )

    convolutional_generator = ConvolutionalGenerator(5, 16)
    print(
        f"Number of total parameters in Generator (convolutions only): {count_parameters(convolutional_generator):,}."
    )

    convolutional_discriminator = ConvolutionalDiscriminator(16)
    print(
        f"Number of total parameters in Discriminator (convolutions only): {count_parameters(convolutional_discriminator):,}."
    )

    gcgan_generator = GCGANGenerator(5)
    print(
        f"Number of total parameters in Generator (GCGAN): {count_parameters(gcgan_generator):,}."
    )

    gcgan_discrminator = GCGANDiscriminator()
    print(
        f"Number of total parameters in Discriminator (GCGAN): {count_parameters(gcgan_discrminator):,}."
    )


if __name__ == '__main__':
    main()
