from torchvision.datasets import ImageFolder
from torchvision import transforms
from torch.utils import data
import torch.nn as nn
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import optim
import time
import torchmetrics
from pprint import pprint

NUMBER_CLASSES = 7
LEARNING_RATE = 0.001
BATCH_SIZE = 16
EPOCHS = 20

TRAIN_TRANSFORMS = transforms.Compose([
    transforms.RandomResizedCrop(size=64, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(45),
    transforms.ColorJitter(brightness=0.2,
                           contrast=0.2,
                           saturation=0.2,
                           hue=0.1),
    transforms.RandomAutocontrast(),
    transforms.ToTensor(),
    transforms.Resize((64, 64))
])
TEST_TRANSFORMS = transforms.Compose(
    [transforms.ToTensor(), transforms.Resize((64, 64))])


class MyCNN(nn.Module):

    def __init__(self, input_size=64):
        super().__init__()
        self.feature_extractor = nn.ModuleList([
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3,
                      padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=32,
                      out_channels=64,
                      kernel_size=3,
                      padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=64,
                      out_channels=128,
                      kernel_size=3,
                      padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        ])

        self.classifier = nn.ModuleList([
            nn.Flatten(),
            nn.Linear(128 * (input_size // 8) * (input_size // 8), 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, NUMBER_CLASSES)
        ])

    def forward(self, x):
        for layer in self.feature_extractor:
            x = layer(x)

        for layer in self.classifier:
            x = layer(x)

        return x


def train_epoch(dataloader_train, optimizer, net, criterion, i):
    net.train()
    running_loss = 0.0

    for inputs, targets in tqdm(dataloader_train, desc=f"Epoch {i+1}"):
        optimizer.zero_grad()

        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)

    average_loss_per_batch = running_loss / len(dataloader_train.dataset)
    print(f"Average training loss per batch: {average_loss_per_batch}")

    return average_loss_per_batch


def validate_epoch(dataloader_val, net, criterion):
    net.eval()
    running_loss = 0.0

    with torch.no_grad():
        for inputs, targets in dataloader_val:
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            running_loss += loss.item() * inputs.size(0)

    average_loss = running_loss / len(dataloader_val.dataset)
    return average_loss


def plot_losses(train_losses, val_losses):
    plt.figure()
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title("Training and Validation Loss per Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()


def train_model(dataloader_train,
                dataloader_val,
                optimizer,
                net,
                num_epochs,
                create_plot=False):
    criterion = nn.CrossEntropyLoss()

    losses_train = []
    losses_val = []
    best_val_loss = float('inf')
    patience = 5
    epochs_no_improve = 0

    start_time = time.time()
    for i in range(num_epochs):
        epoch_loss = train_epoch(dataloader_train, optimizer, net, criterion,
                                 i)
        losses_train.append(epoch_loss)

        val_loss = validate_epoch(dataloader_val, net, criterion)
        losses_val.append(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            # Save the best model so far
            torch.save(net.state_dict(), 'best_model.pth')
        else:
            epochs_no_improve += 1
            if epochs_no_improve == patience:
                print('Early stopping triggered')
                break

    end_time = time.time()

    average_loss_per_epoch = np.mean(losses_train)

    if create_plot:
        plot_losses(losses_train, losses_val)

    net.load_state_dict(torch.load('best_model.pth'))

    return average_loss_per_epoch, end_time - start_time


def evaluate_model(dataloader_test, net, precision_metric, recall_metric,
                   f1_metric, f1_per_class_metric):
    net.eval()

    with torch.no_grad():
        for inputs, targets in dataloader_test:
            outputs = net(inputs)

            precision_metric.update(outputs, targets)
            recall_metric.update(outputs, targets)
            f1_metric.update(outputs, targets)
            f1_per_class_metric.update(outputs, targets)

    precision_metric_value = precision_metric.compute().item()
    recall_metric_value = recall_metric.compute().item()
    f1_metric_value = f1_metric.compute().item()
    f1_per_class_metric_values = f1_per_class_metric.compute()

    precision_metric.reset()
    recall_metric.reset()
    f1_metric.reset()
    f1_per_class_metric.reset()

    return precision_metric_value, recall_metric_value, f1_metric_value, f1_per_class_metric_values


def get_data_loaders():

    dataset_train_val = ImageFolder("../Data/clouds/clouds_train",
                                    transform=TRAIN_TRANSFORMS)
    train_size = int(0.8 * len(dataset_train_val))
    val_size = len(dataset_train_val) - train_size
    dataset_train, dataset_val = data.random_split(dataset_train_val,
                                                   [train_size, val_size])

    dataset_test = ImageFolder("../Data/clouds/clouds_test",
                               transform=TEST_TRANSFORMS)

    dataloader_train = data.DataLoader(dataset_train,
                                       batch_size=BATCH_SIZE,
                                       shuffle=True)
    dataloader_val = data.DataLoader(dataset_val,
                                     batch_size=BATCH_SIZE,
                                     shuffle=False)
    dataloader_test = data.DataLoader(dataset_test,
                                      batch_size=BATCH_SIZE,
                                      shuffle=False)

    return dataloader_train, dataloader_val, dataloader_test, dataset_train_val.classes


def main():

    dataloader_train, dataloader_val, dataloader_test, class_names = get_data_loaders(
    )
    net = MyCNN()
    optimizer = optim.AdamW(net.parameters(), lr=LEARNING_RATE)

    average_loss, training_time = train_model(dataloader_train,
                                              dataloader_val,
                                              optimizer,
                                              net,
                                              EPOCHS,
                                              create_plot=True)

    precision_metric = torchmetrics.Precision(task="multiclass",
                                              average="macro",
                                              num_classes=NUMBER_CLASSES)
    recall_metric = torchmetrics.Recall(task="multiclass",
                                        average="macro",
                                        num_classes=NUMBER_CLASSES)
    f1_metric = torchmetrics.F1Score(task="multiclass",
                                     average="macro",
                                     num_classes=NUMBER_CLASSES)
    f1_per_class_metric = torchmetrics.F1Score(task="multiclass",
                                               average="none",
                                               num_classes=NUMBER_CLASSES)

    precision_score, recall_score, f1_score, f1_per_class_scores = evaluate_model(
        dataloader_test, net, precision_metric, recall_metric, f1_metric,
        f1_per_class_metric)

    per_class_f1_dict = {
        class_names[i]: round(f1_per_class_scores[i].item(), 4)
        for i in range(NUMBER_CLASSES)
    }

    print("Summary statistics:")
    print(f"Average training loss per epoch: {average_loss:.16f}")
    print(f"Precision: {precision_score:.16f}")
    print(f"Recall: {recall_score:.16f}")
    print(f"F1: {f1_score:.16f}")
    print(
        f"Total time taken to train the model in seconds: {training_time:.16f}"
    )
    print("Per class F1 score")
    pprint(per_class_f1_dict)


if __name__ == '__main__':
    main()
