import torch
import torch.nn as nn


def main():
    temperature_observation = [3, 4, 6, 2, 3, 6, 8, 9]
    input_tensor = torch.tensor(temperature_observation).float()

    input_size = len(temperature_observation)
    output_size = 1

    model = nn.Sequential(nn.Linear(input_size, output_size), nn.Sigmoid())

    output_tensor = model(input_tensor)
    print(output_tensor.item())
    # B (it is only in the range [0, 1] because of the sigmoid function)


if __name__ == '__main__':
    main()
