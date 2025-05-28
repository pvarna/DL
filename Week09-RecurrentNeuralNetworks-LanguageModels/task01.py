import numpy as np
import torch
import pandas as pd
import os
from torch.utils.data import TensorDataset

TRAIN_FILE = os.path.join("..", "DATA", "electricity_consumption",
                          "electricity_train.csv")


def create_sequences(pd_dataframe, sequence_length):
    second_column = list(pd_dataframe.iloc[:, 1])

    input_sequences, targets = [], []
    for starting_index in range(len(pd_dataframe) - sequence_length):
        input_sequences.append(second_column[starting_index:starting_index +
                                             sequence_length])
        targets.append(second_column[starting_index + sequence_length])

    return np.array(input_sequences), np.array(targets)


def main():
    electricity_pd_dataframe = pd.read_csv(TRAIN_FILE)

    dummy_pd_dataframe = pd.DataFrame({
        "first": list(range(100)),
        "second": list(range(100))
    })

    dummy_input_sequences, dummy_targets = create_sequences(
        dummy_pd_dataframe, 5)

    print(f"First five training examples: {dummy_input_sequences[:5]}")
    print(f"First five target values: {dummy_targets[:5]}")

    X_train, y_train = create_sequences(electricity_pd_dataframe, 96)
    print(f"{X_train.shape=}")
    print(f"{y_train.shape=}")

    dataset = TensorDataset(torch.from_numpy(X_train),
                            torch.from_numpy(y_train))
    print(f"Length of training TensorDataset: {len(dataset)}")


if __name__ == '__main__':
    main()
