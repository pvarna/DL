import torch.nn as nn

INPUT_SIZE = 8
OUTPUT_SIZE = 2


def main():
    model1 = nn.Sequential(nn.Linear(INPUT_SIZE, 5), nn.Linear(5, 9),
                           nn.Linear(9, OUTPUT_SIZE))

    model2 = nn.Sequential(nn.Linear(INPUT_SIZE, 5), nn.Linear(5, 9),
                           nn.Linear(9, 2), nn.Linear(2, OUTPUT_SIZE))

    print(
        f"Number of parameters in network 1: {sum(p.numel() for p in model1.parameters())}"
    )
    print(
        f"Number of parameters in network 2: {sum(p.numel() for p in model2.parameters())}"
    )


if __name__ == '__main__':
    main()
