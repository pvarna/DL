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

# https://pytorch.org/docs/stable/notes/randomness.html
torch.manual_seed(42)

BATCH_SIZE = 64
LEARNING_RATE = 0.001


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(9, 16)
        self.fc2 = nn.Linear(16, 8)
        self.fc3 = nn.Linear(8, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.sigmoid(self.fc3(x))

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
                num_epochs=1000,
                create_plot=True)

    metric = torchmetrics.F1Score(task="binary", average="micro")
    f1_score = evaluate_model(test_dataloader, net, metric)

    print(f"F1 score on test set: {f1_score:.16f}")

    # The model doesn't perform very well (F1 score is ~0.56)
    # https://encord.com/blog/f1-score-in-machine-learning/ --> here is says it should be >0.8


if __name__ == '__main__':
    main()
