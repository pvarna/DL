import torch
import torch.nn as nn


def main():
    temperature_observation = [2, 3, 6, 7, 9, 3, 2, 1]
    input_tensor = torch.tensor(temperature_observation).float()

    input_size = len(temperature_observation)
    output_size = 1

    model = nn.Sequential(nn.Linear(input_size, 5), nn.Linear(5, output_size))

    output_tensor = model(input_tensor)
    print(output_tensor)


if __name__ == '__main__':
    main()
