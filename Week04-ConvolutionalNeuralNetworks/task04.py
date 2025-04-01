import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from torch import optim
import torch.nn.init as init

torch.manual_seed(42)

BATCH_SIZE = 32
LEARNING_RATE = 0.0005
DROPOUT_PROBABILITY = 0.2

class Net(nn.Module):

    def __init__(self, dropout_probability=DROPOUT_PROBABILITY):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(9, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.d1 = nn.Dropout(p=dropout_probability)

        self.fc2 = nn.Linear(128, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.d2 = nn.Dropout(p=dropout_probability)

        self.fc3 = nn.Linear(128, 64)
        self.bn3 = nn.BatchNorm1d(64)
        self.d3 = nn.Dropout(p=dropout_probability)

        self.fc4 = nn.Linear(64, 32)
        self.bn4 = nn.BatchNorm1d(32)
        self.d4 = nn.Dropout(p=dropout_probability)

        self.fc5 = nn.Linear(32, 1)

        init.kaiming_uniform_(self.fc1.weight, nonlinearity="leaky_relu")
        init.kaiming_uniform_(self.fc2.weight, nonlinearity="leaky_relu")
        init.kaiming_uniform_(self.fc3.weight, nonlinearity="leaky_relu")
        init.kaiming_uniform_(self.fc4.weight, nonlinearity="leaky_relu")
        init.kaiming_uniform_(self.fc5.weight, nonlinearity="sigmoid")

    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)
        x = F.leaky_relu(x)
        x = self.d1(x)

        x = self.fc2(x)
        x = self.bn2(x)
        x = F.leaky_relu(x)
        x = self.d2(x)

        x = self.fc3(x)
        x = self.bn3(x)
        x = F.leaky_relu(x)
        x = self.d3(x)

        x = self.fc4(x)
        x = self.bn4(x)
        x = F.leaky_relu(x)
        x = self.d4(x)

        x = self.fc5(x)
        x = F.sigmoid(x)

        return x


class WaterPotabilityDataset(Dataset):

    def __init__(self, training_data_file):
        super().__init__()
        self.water_potability_data = pd.read_csv(training_data_file)

    def __len__(self):
        return len(self.water_potability_data)

    def __getitem__(self, index):
        np_array = self.water_potability_data.iloc[index].to_numpy()
        return np_array[:-1], np_array[-1]


def train_epoch(dataloader_train, optimizer, net, criterion):
    running_loss = 0.0

    for inputs, targets in dataloader_train:
        optimizer.zero_grad()

        inputs = inputs.float()
        targets = targets.float().unsqueeze(1)

        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)

    epoch_loss = running_loss / len(dataloader_train.dataset)

    return epoch_loss


def plot_losses(losses):
    plt.figure()
    plt.plot(losses)
    plt.title("Loss per epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.show()


def train_model(dataloader_train,
                optimizer,
                net,
                num_epochs,
                create_plot=False):
    net.train()
    criterion = nn.BCELoss()

    losses = []
    for _ in tqdm(range(num_epochs)):
        epoch_loss = train_epoch(dataloader_train, optimizer, net, criterion)
        losses.append(epoch_loss)

    average_loss = np.mean(losses)
    print(f"Average loss: {average_loss:.16f}")

    if create_plot:
        plot_losses(losses)


def evaluate_model(dataloader_test, net, metric):
    net.eval()

    with torch.no_grad():
        for inputs, targets in dataloader_test:
            inputs = inputs.float()
            targets = targets.float().unsqueeze(1)

            outputs = net(inputs)
            metric.update(outputs, targets)

    metric_value = metric.compute().item()
    metric.reset()
    return metric_value


def load_data():
    train_dataset = WaterPotabilityDataset("../DATA/water_train.csv")
    train_dataloader = DataLoader(train_dataset,
                                  batch_size=BATCH_SIZE,
                                  shuffle=True)

    test_dataset = WaterPotabilityDataset("../DATA/water_test.csv")
    test_dataloader = DataLoader(test_dataset,
                                 batch_size=BATCH_SIZE,
                                 shuffle=False)

    return train_dataloader, test_dataloader


def compare_optimizers(train_dataloader):
    optimizers = {
        "SGD": optim.SGD,
        "RMSprop": optim.RMSprop,
        "Adam": optim.Adam,
        "AdamW": optim.AdamW,
    }

    for optimizer_name, optimizer_class in optimizers.items():
        net = Net()
        print(f"Training with {optimizer_name} optimizer:")
        optimizer = optimizer_class(net.parameters(), lr=LEARNING_RATE)
        train_model(train_dataloader, optimizer, net, num_epochs=10)


def main():
    train_dataloader, test_dataloader = load_data()

    compare_optimizers(train_dataloader)

    net = Net()
    optimizer = optim.AdamW(net.parameters(), lr=LEARNING_RATE)
    train_model(train_dataloader,
                optimizer,
                net,
                num_epochs=500,
                create_plot=True)

    metric = torchmetrics.F1Score(task="binary", average="micro")
    f1_score = evaluate_model(test_dataloader, net, metric)

    print(f"F1 score on test set: {f1_score:.16f}")



if __name__ == '__main__':
    main()
