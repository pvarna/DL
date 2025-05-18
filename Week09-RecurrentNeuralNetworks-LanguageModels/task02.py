import numpy as np
import torch
import pandas as pd
import os
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm

TRAIN_FILE = os.path.join("..", "DATA", "electricity_consumption",
                          "electricity_train.csv")
TEST_FILE = os.path.join("..", "DATA", "electricity_consumption",
                          "electricity_test.csv")

NUM_EPOCHS = 3
BATCH_SIZE = 32
SAMPLES_PER_BATCH = 32
LEARNING_RATE = 0.0001

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

def train_model(train_dataloader, model, optimizer, criterion, num_epochs):
    model.train()

    for epoch in range(num_epochs):
        epoch_loss = train_epoch(train_dataloader, model, optimizer, criterion)
        print(f"Epoch {epoch}, Average MSE loss: {epoch_loss}")

def evaluate_model(dataloader, model, criterion):
    model.eval()
    running_loss = 0.0

    with torch.no_grad():
        for batch_X, batch_y in tqdm(dataloader):
            batch_X, batch_y = batch_X.float(), batch_y.float()

            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            running_loss += loss.item() * batch_X.size(0)

    average_loss = running_loss / len(dataloader.dataset)
    return average_loss

def main():
    electricity_pd_dataframe_train = pd.read_csv(TRAIN_FILE)
    electricity_pd_dataframe_test = pd.read_csv(TEST_FILE)
    model = ElectricityConsumptionNet()
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    X_train, y_train = create_sequences(electricity_pd_dataframe_train, SAMPLES_PER_BATCH)
    X_test, y_test = create_sequences(electricity_pd_dataframe_test, SAMPLES_PER_BATCH)
    
    train_dataset = TensorDataset(torch.from_numpy(X_train),
                            torch.from_numpy(y_train))
    test_dataset = TensorDataset(torch.from_numpy(X_test),
                            torch.from_numpy(y_test))
    
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    train_model(train_dataloader, model, optimizer, criterion, NUM_EPOCHS)
    test_loss = evaluate_model(test_dataloader, model, criterion)
    print(f"Test MSE: {test_loss}")


if __name__ == '__main__':
    main()
