import torch.nn as nn


def main():
    model = nn.Sequential(nn.Linear(5, 50), nn.Linear(50, 2), nn.Linear(2, 2),
                          nn.Linear(2, 1))

    for layer in model:
        nn.init.uniform_(layer.weight)
        nn.init.uniform_(layer.bias)

    for layer in list(model)[:2]:
        for param in layer.parameters():
            param.requires_grad = False

    for i, layer in enumerate(model):
        print(f"\nLayer {i+1}:")
        print("Weights:\n", layer.weight[:5])  # Print first 5 neurons
        print("Biases:\n", layer.bias[:5])


if __name__ == '__main__':
    main()
