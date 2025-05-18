import numpy as np
import torch
import pandas as pd
import os
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, random_split
from tqdm import tqdm
from torchmetrics.regression import R2Score
import matplotlib.pyplot as plt

TRAIN_FILE = os.path.join("..", "DATA", "electricity_consumption",
                          "electricity_train.csv")
TEST_FILE = os.path.join("..", "DATA", "electricity_consumption",
                         "electricity_test.csv")

NUM_EPOCHS = 3
BATCH_SIZE = 32
SEQUENCE_LENGTH = 32
LEARNING_RATE = 0.001


class ElectricityConsumptionNet(nn.Module):

    def __init__(self, input_size=1, hidden_size=32, num_layers=2):
        super(ElectricityConsumptionNet, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_size=input_size,
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            batch_first=True)

        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        x = x.unsqueeze(-1)

        h0 = torch.zeros(self.num_layers, x.size(0),
                         self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0),
                         self.hidden_size).to(x.device)

        out, _ = self.lstm(x, (h0, c0))

        last_output = out[:, -1, :]
        out = self.fc(last_output)

        return out.squeeze(1)


def create_sequences(pd_dataframe, sequence_length):
    second_column = list(pd_dataframe.iloc[:, 1])

    input_sequences, targets = [], []
    for starting_index in range(len(pd_dataframe) - sequence_length):
        input_sequences.append(second_column[starting_index:starting_index +
                                             sequence_length])
        targets.append(second_column[starting_index + sequence_length])

    return np.array(input_sequences), np.array(targets)


def train_epoch(train_dataloader, model, optimizer, criterion):
    running_loss = 0.0

    for batch_X, batch_y in tqdm(train_dataloader):
        batch_X, batch_y = batch_X.float(), batch_y.float()

        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        running_loss += loss.item() * batch_X.size(0)
        loss.backward()
        optimizer.step()

    epoch_loss = running_loss / len(train_dataloader.dataset)
    return epoch_loss


def train_model(train_dataloader, val_dataloader, model, optimizer, criterion,
                num_epochs):
    model.train()
    train_losses = []
    val_losses, val_r2_scores = [], []

    for epoch in range(num_epochs):
        train_loss = train_epoch(train_dataloader, model, optimizer, criterion)
        val_loss, val_r2 = evaluate_model(val_dataloader, model, criterion)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_r2_scores.append(val_r2)

        print(f"Epoch {epoch}:")
        print(f"Train MSE loss: {train_loss}")
        print(f"Val MSE loss: {val_loss}")
        print(f"Val R^2: {val_r2}")

    plot_curves(train_losses, val_losses, val_r2_scores)


def evaluate_model(dataloader, model, criterion):
    model.eval()
    running_loss = 0.0
    r2_metric = R2Score()

    with torch.no_grad():
        for batch_X, batch_y in tqdm(dataloader):
            batch_X, batch_y = batch_X.float(), batch_y.float()

            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            running_loss += loss.item() * batch_X.size(0)

            r2_metric.update(outputs, batch_y)

    average_loss = running_loss / len(dataloader.dataset)
    r2 = r2_metric.compute().item()

    return average_loss, r2


def get_val_test_datasets(test_dir):
    electricity_pd_dataframe_test = pd.read_csv(test_dir)

    X_full, y_full = create_sequences(electricity_pd_dataframe_test,
                                      SEQUENCE_LENGTH)

    full_dataset = TensorDataset(torch.from_numpy(X_full),
                                 torch.from_numpy(y_full))

    val_size = len(full_dataset) // 2
    test_size = len(full_dataset) - val_size

    val_dataset, test_dataset = random_split(full_dataset,
                                             [val_size, test_size])

    return val_dataset, test_dataset


def get_train_dataset(train_dir):
    electricity_pd_dataframe_train = pd.read_csv(train_dir)

    X_train, y_train = create_sequences(electricity_pd_dataframe_train,
                                        SEQUENCE_LENGTH)

    train_dataset = TensorDataset(torch.from_numpy(X_train),
                                  torch.from_numpy(y_train))

    return train_dataset


def plot_curves(train_losses,
                val_losses,
                val_r2_scores,
                title="Training progress"):
    epochs = range(1, len(train_losses) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(15, 4))

    # --- Loss ---
    axes[0].plot(epochs, train_losses, label="Train")
    axes[0].plot(epochs, val_losses, label="Val")
    axes[0].set_title("MSE Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()
    axes[0].grid(True)

    # --- R^2 (val only) ---
    axes[1].plot(epochs, val_r2_scores, label="Val R^2", color="tab:green")
    axes[1].set_ylim(0, 1)
    axes[1].set_title("R^2")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()
    axes[1].grid(True)

    fig.suptitle(title)
    plt.tight_layout()
    plt.show()


def main():
    model = ElectricityConsumptionNet()
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    train_dataset = get_train_dataset(TRAIN_FILE)
    val_dataset, test_dataset = get_val_test_datasets(TEST_FILE)

    train_dataloader = DataLoader(train_dataset,
                                  batch_size=BATCH_SIZE,
                                  shuffle=True)
    val_dataloader = DataLoader(val_dataset,
                                batch_size=BATCH_SIZE,
                                shuffle=False)
    test_dataloader = DataLoader(test_dataset,
                                 batch_size=BATCH_SIZE,
                                 shuffle=False)

    train_model(train_dataloader, val_dataloader, model, optimizer, criterion,
                NUM_EPOCHS)

    test_loss, test_r2 = evaluate_model(test_dataloader, model, criterion)
    print(f"Test MSE loss: {test_loss}")
    print(f"Test R^2: {test_r2}")


if __name__ == '__main__':
    main()
