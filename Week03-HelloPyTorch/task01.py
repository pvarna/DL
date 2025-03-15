import torch


def main():
    raw_temperatures = [[72, 75, 78], [70, 73, 76]]
    tensor_temperatures = torch.tensor(raw_temperatures)
    print(f"Temperatures: {tensor_temperatures}")
    print(f"Shape of temperatures: {tensor_temperatures.shape}")
    print(f"Data type of temperatures: {tensor_temperatures.dtype}")

    corrected_temperatures = tensor_temperatures + 2
    print(f"Corrected temperatures: {corrected_temperatures}")


if __name__ == '__main__':
    main()
