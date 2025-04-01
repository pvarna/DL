import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import Dataset
from torchvision import transforms
import glob
import os
from PIL import Image
from tqdm import tqdm
from torch.utils import data
from torch import optim
import torchmetrics

CLASSES_COUNT = 964
ALHABETS_COUNT = 30
BATCH_SIZE = 32
LEARNING_RATE = 0.001
EPOCHS = 5


class OmniglotDataset(Dataset):

    def __init__(self, path_to_images, transformations):
        super().__init__()
        images = []
        alphabets = []
        labels = []
        for file in sorted(
                glob.glob(path_to_images + os.path.sep + "**",
                          recursive=True)):
            if os.path.isfile(file):
                split_filepath = file.split(os.path.sep)
                images.append(file)
                alphabets.append(split_filepath[-3])
                labels.append(f"{split_filepath[-3]}_{split_filepath[-2]}")

        alphabets = torch.tensor(pd.get_dummies(alphabets,
                                                dtype=int).values).float()
        labels = pd.get_dummies(labels,
                                dtype=int).values.argmax(axis=1).tolist()

        self.data = list(zip(images, alphabets, labels))
        self.transformations = transformations

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        image_path, alphabet, label = self.data[index]
        image = Image.open(image_path).convert("L")
        image = self.transformations(image)

        return image, alphabet, label


class ImageNet(nn.Module):

    def __init__(self):
        super(ImageNet, self).__init__()
        self.convolutional_layers = nn.ModuleList([
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.MaxPool2d(kernel_size=2),  # 32x32
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.MaxPool2d(kernel_size=2),  # 16x16
            nn.ReLU()
        ])

        self.linear_layer = nn.Linear(32 * 16 * 16, 128)

    def forward(self, x):
        for layer in self.convolutional_layers:
            x = layer(x)

        x = x.view(x.size(0), -1)

        return self.linear_layer(x)


class AlphabetNet(nn.Module):

    def __init__(self):
        super(AlphabetNet, self).__init__()
        self.layers = nn.ModuleList([
            nn.Linear(ALHABETS_COUNT, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU()
        ])

        self.final_layer = nn.Linear(32, 8)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)

        return self.final_layer(x)


class OmniglotTwoInputNet(nn.Module):

    def __init__(self):
        super(OmniglotTwoInputNet, self).__init__()
        self.image_net = ImageNet()
        self.alphabet_net = AlphabetNet()

        self.classifier = nn.ModuleList([
            nn.Linear(128 + 8, 512),
            nn.ReLU(),
            nn.Linear(512, CLASSES_COUNT)
        ])

    def forward(self, image, alphabet_vector):
        image_output = self.image_net(image)
        aphabet_output = self.alphabet_net(alphabet_vector)

        x = torch.cat((image_output, aphabet_output), dim=1)
        for layer in self.classifier:
            x = layer(x)

        return x


def train_epoch(dataloader_train, optimizer, net, criterion, metric):
    net.train()
    running_loss = 0.0

    for images, alphabets, labels in tqdm(dataloader_train):
        optimizer.zero_grad()

        outputs = net(images, alphabets)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        metric.update(outputs, labels)
        running_loss += loss.item() * images.size(0)

    metric_value = metric.compute().item()
    metric.reset()

    average_loss_per_batch = running_loss / len(dataloader_train.dataset)

    return average_loss_per_batch, metric_value


def validate_epoch(dataloader_validation, net, criterion, metric):
    net.eval()
    running_loss = 0.0

    with torch.no_grad():
        for images, alphabets, labels in tqdm(dataloader_validation):
            outputs = net(images, alphabets)
            loss = criterion(outputs, labels)

            metric.update(outputs, labels)
            running_loss += loss.item() * images.size(0)

    metric_value = metric.compute().item()
    metric.reset()

    average_loss_per_batch = running_loss / len(dataloader_validation.dataset)

    return average_loss_per_batch, metric_value

def plot_learning_curves(train_losses, validation_losses, train_metrics, validation_metrics):
    epochs_range = range(1, len(train_losses) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle("Classify handwritten characters")

    # --- Loss ---
    ax1.plot(epochs_range, train_losses, label='Training loss')
    ax1.plot(epochs_range, validation_losses, label='Validation loss')
    ax1.set_title('Model loss')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.legend()

    # --- Metric ---
    ax2.plot(epochs_range, train_metrics, label='Training metric')
    ax2.plot(epochs_range, validation_metrics, label='Validation metric')
    ax2.set_title('Model performance')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Metric Value')
    ax2.legend()

    plt.show()

def train_model(dataloader_train,
                dataloader_validation,
                optimizer,
                net,
                num_epochs,
                metric,
                create_plot=False):
    net.train()
    criterion = nn.CrossEntropyLoss()

    train_losses, validation_losses = [], []
    train_metrics, validation_metrics = [], []

    for i in range(num_epochs):
        train_loss, train_metric_value = train_epoch(dataloader_train,
                                                     optimizer, net, criterion,
                                                     metric)
        validation_loss, validation_metric_value = validate_epoch(
            dataloader_validation, net, criterion, metric)
        
        train_losses.append(train_loss)
        validation_losses.append(validation_loss)
        train_metrics.append(train_metric_value)
        validation_metrics.append(validation_metric_value)

        print(f"Epoch [{i+1}/{num_epochs}]")
        print(f"Average training loss: {train_loss}")
        print(f"Average validation loss: {validation_loss}")
        print(f"Training metric score: {train_metric_value}")
        print(f"Validation metric score: {validation_metric_value}")

    if create_plot:
        plot_learning_curves(train_losses, validation_losses, train_metrics, validation_metrics)

    return None


def main():
    transformations = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Resize((64, 64))])

    dataset_train = OmniglotDataset("../DATA/omniglot_train", transformations)
    dataset_validation = OmniglotDataset("../DATA/omniglot_test", transformations)

    dataloader_train = data.DataLoader(dataset_train,
                                       batch_size=BATCH_SIZE,
                                       shuffle=True)
    dataloader_validation = data.DataLoader(dataset_validation,
                                            batch_size=BATCH_SIZE,
                                            shuffle=False)

    net = OmniglotTwoInputNet()
    optimizer = optim.AdamW(net.parameters(), lr=LEARNING_RATE)
    f1_metric = torchmetrics.F1Score(task="multiclass",
                                     average="macro",
                                     num_classes=CLASSES_COUNT)
    
    train_model(dataloader_train, dataloader_validation, optimizer, net, EPOCHS, f1_metric, create_plot=True)


if __name__ == '__main__':
    main()
