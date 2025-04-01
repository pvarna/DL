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

CHARACTERS_COUNT = 964
ALHABETS_COUNT = 30
BATCH_SIZE = 32
LEARNING_RATE = 0.001
EPOCHS = 5


class OmniglotDataset(Dataset):

    def __init__(self, path_to_images, transformations):
        super().__init__()
        images = []
        alphabet_names = []
        character_names = []
        for file in sorted(
                glob.glob(path_to_images + os.path.sep + "**",
                          recursive=True)):
            if os.path.isfile(file):
                split_filepath = file.split(os.path.sep)
                images.append(file)
                alphabet_names.append(split_filepath[-3])
                character_names.append(
                    f"{split_filepath[-3]}_{split_filepath[-2]}")

        alphabet_names = pd.get_dummies(
            alphabet_names, dtype=int).values.argmax(axis=1).tolist()
        character_names = pd.get_dummies(
            character_names, dtype=int).values.argmax(axis=1).tolist()

        self.data = list(zip(images, alphabet_names, character_names))
        self.transformations = transformations

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        image_path, alphabet_label, character_label = self.data[index]
        image = Image.open(image_path).convert("L")
        image = self.transformations(image)

        return image, alphabet_label, character_label


class OmniglotTwoOutputNet(nn.Module):

    def __init__(self):
        super(OmniglotTwoOutputNet, self).__init__()

        self.convolutional_layers = nn.ModuleList([
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.MaxPool2d(kernel_size=2),  # 32x32
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.MaxPool2d(kernel_size=2),  # 16x16
            nn.ReLU()
        ])

        self.shared_linear_layer = nn.ModuleList(
            [nn.Linear(32 * 16 * 16, 256),
             nn.ReLU()])

        self.characters_linear_layer = nn.Linear(256, CHARACTERS_COUNT)
        self.alphabets_linear_layer = nn.Linear(256, ALHABETS_COUNT)

    def forward(self, x):
        for layer in self.convolutional_layers:
            x = layer(x)

        x = x.view(x.size(0), -1)

        for layer in self.shared_linear_layer:
            x = layer(x)

        characters_output = self.characters_linear_layer(x)
        alphabets_output = self.alphabets_linear_layer(x)

        return characters_output, alphabets_output


def train_epoch(dataloader_train, optimizer, net, characters_criterion,
                alphabets_criterion, characters_metric, alphabets_metric):
    net.train()
    running_loss = 0.0

    for images, alphabet_labels, character_labels in tqdm(dataloader_train):
        optimizer.zero_grad()

        characters_outputs, alphabets_outputs = net(images)
        characters_loss = characters_criterion(characters_outputs,
                                               character_labels)
        alphabets_loss = alphabets_criterion(alphabets_outputs,
                                             alphabet_labels)

        loss = characters_loss + alphabets_loss

        loss.backward()

        optimizer.step()

        characters_metric.update(characters_outputs, character_labels)
        alphabets_metric.update(alphabets_outputs, alphabet_labels)
        running_loss += loss.item() * images.size(0)

    characters_metric_value = characters_metric.compute().item()
    characters_metric.reset()
    alphabets_metric_value = alphabets_metric.compute().item()
    alphabets_metric.reset()

    average_loss_per_batch = running_loss / len(dataloader_train.dataset)

    return average_loss_per_batch, characters_metric_value, alphabets_metric_value


def validate_epoch(dataloader_validation, net, characters_criterion,
                   alphabets_criterion, characters_metric, alphabets_metric):
    net.eval()
    running_loss = 0.0

    with torch.no_grad():
        for images, alphabet_labels, character_labels in tqdm(
                dataloader_validation):
            characters_outputs, alphabets_outputs = net(images)
            characters_loss = characters_criterion(characters_outputs,
                                                   character_labels)
            alphabets_loss = alphabets_criterion(alphabets_outputs,
                                                 alphabet_labels)

            loss = characters_loss + alphabets_loss

            characters_metric.update(characters_outputs, character_labels)
            alphabets_metric.update(alphabets_outputs, alphabet_labels)
            running_loss += loss.item() * images.size(0)

    characters_metric_value = characters_metric.compute().item()
    characters_metric.reset()
    alphabets_metric_value = alphabets_metric.compute().item()
    alphabets_metric.reset()

    average_loss_per_batch = running_loss / len(dataloader_validation.dataset)

    return average_loss_per_batch, characters_metric_value, alphabets_metric_value


def plot_learning_curves(train_losses, validation_losses,
                         train_characters_metrics,
                         validation_characters_metrics,
                         train_alphabets_metrics, validation_alhabets_metrics):
    epochs_range = range(1, len(train_losses) + 1)
    print(f"Train losses: {train_losses}")
    print(f"Validation losses: {validation_losses}")
    print(f"Train char m: {train_characters_metrics}")
    print(f"V ch m: {validation_characters_metrics}")
    print(f"Tr a m: {train_alphabets_metrics}")
    print(f"V a m: {validation_alhabets_metrics}")

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 4))
    fig.suptitle("Classify handwritten characters")

    # --- Loss ---
    ax1.plot(epochs_range, train_losses, label='Training loss')
    ax1.plot(epochs_range, validation_losses, label='Validation loss')
    ax1.set_title('Model loss')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.legend()

    # --- Characters Metric ---
    ax2.plot(epochs_range, train_characters_metrics, label='Training metric')
    ax2.plot(epochs_range,
             validation_characters_metrics,
             label='Validation metric')
    ax2.set_title('Model performance on characters')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Metric Value')
    ax2.legend()

    # --- Alphabets Metric ---
    ax3.plot(epochs_range, train_alphabets_metrics, label='Training metric')
    ax3.plot(epochs_range,
             validation_alhabets_metrics,
             label='Validation metric')
    ax3.set_title('Model performance on alphabets')
    ax3.set_xlabel('Epochs')
    ax3.set_ylabel('Metric Value')
    ax3.legend()

    plt.tight_layout()
    plt.show()


def train_model(dataloader_train,
                dataloader_validation,
                optimizer,
                net,
                num_epochs,
                characters_metric,
                alphabets_metric,
                create_plot=False):
    net.train()
    characters_criterion = nn.CrossEntropyLoss()
    alphabets_criterion = nn.CrossEntropyLoss()

    train_losses, validation_losses = [], []
    train_characters_metrics, validation_characters_metrics = [], []
    train_alphabets_metrics, validation_alphabets_metrics = [], []

    for i in range(num_epochs):
        train_loss, train_characters_metric_value, train_alphabets_metric_value = train_epoch(
            dataloader_train, optimizer, net, characters_criterion,
            alphabets_criterion, characters_metric, alphabets_metric)
        validation_loss, validation_characters_metric_value, validation_alphabets_metric_value = validate_epoch(
            dataloader_validation, net, characters_criterion,
            alphabets_criterion, characters_metric, alphabets_metric)

        train_losses.append(train_loss)
        validation_losses.append(validation_loss)
        train_characters_metrics.append(train_characters_metric_value)
        train_alphabets_metrics.append(train_alphabets_metric_value)
        validation_characters_metrics.append(
            validation_characters_metric_value)
        validation_alphabets_metrics.append(validation_alphabets_metric_value)

        print(f"Epoch [{i+1}/{num_epochs}]")
        print(f"Average training loss: {train_loss}")
        print(f"Average validation loss: {validation_loss}")
        print(
            f"Training metric score characters: {train_characters_metric_value}"
        )
        print(
            f"Validation metric score characters: {validation_characters_metric_value}"
        )
        print(
            f"Training metric score alphabets: {train_alphabets_metric_value}")
        print(
            f"Validation metric score alphabets: {validation_alphabets_metric_value}"
        )

    if create_plot:
        plot_learning_curves(train_losses, validation_losses,
                             train_characters_metrics,
                             validation_characters_metrics,
                             train_alphabets_metrics,
                             validation_alphabets_metrics)

    return None


def main():
    transformations = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Resize((64, 64))])

    dataset_train = OmniglotDataset("../DATA/omniglot_train", transformations)
    dataset_validation = OmniglotDataset("../DATA/omniglot_test",
                                         transformations)

    dataloader_train = data.DataLoader(dataset_train,
                                       batch_size=BATCH_SIZE,
                                       shuffle=True)
    dataloader_validation = data.DataLoader(dataset_validation,
                                            batch_size=BATCH_SIZE,
                                            shuffle=False)

    net = OmniglotTwoOutputNet()
    optimizer = optim.AdamW(net.parameters(), lr=LEARNING_RATE)
    characters_f1_metric = torchmetrics.F1Score(task="multiclass",
                                                average="micro",
                                                num_classes=CHARACTERS_COUNT)
    alphabets_f1_metric = torchmetrics.F1Score(task="multiclass",
                                               average="micro",
                                               num_classes=ALHABETS_COUNT)

    train_model(dataloader_train,
                dataloader_validation,
                optimizer,
                net,
                EPOCHS,
                characters_f1_metric,
                alphabets_f1_metric,
                create_plot=True)


if __name__ == '__main__':
    main()
