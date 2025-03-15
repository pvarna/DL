import numpy as np
import torch
from torch.utils.data import TensorDataset


def main():
    dataset_raw = np.random.rand(12, 9)
    features = dataset_raw[:, :-1]
    target = dataset_raw[:, -1]

    dataset = TensorDataset(
        torch.tensor(features).float(),
        torch.tensor(target).float())
    last_sample, last_label = dataset[-1]

    print(f"Last sample: {last_sample}")
    print(f"Last label: {last_label}")


if __name__ == '__main__':
    main()
