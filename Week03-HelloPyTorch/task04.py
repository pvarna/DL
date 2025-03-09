import torch
import torch.nn as nn


def main():
    y = [2]
    scores = torch.tensor([[0.1, 6.0, -2.0, 3.2]]).float()

    criterion = nn.CrossEntropyLoss()
    print(criterion(scores, torch.tensor(y)))


if __name__ == '__main__':
    main()
