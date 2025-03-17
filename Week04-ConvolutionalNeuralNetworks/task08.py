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
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(45),
    transforms.RandomAutocontrast(),
    transforms.ToTensor(),
    transforms.Resize((64, 64))
])
TEST_TRANSFORMS = transforms.Compose([ 
    transforms.ToTensor(),
    transforms.Resize((64, 64))
])


class MyCNN(nn.Module):
    def __init__(self): 
        super(MyCNN, self).__init__()

        self.features = nn.ModuleList([
            # Block 1
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 2
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 3
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 4
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 5
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        ])

        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))

        self.classifier = nn.ModuleList([
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, NUMBER_CLASSES)
        ])

    def forward(self, x):
        for layer in self.features:
            x = layer(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        for layer in self.classifier:
            x = layer(x)

        return x

def train_epoch(dataloader_train, optimizer, net, criterion, i):
    running_loss = 0.0

    for inputs, targets in tqdm(dataloader_train, desc=f"Epoch {i+1}"):
        optimizer.zero_grad()

        # inputs = inputs.float()
        # targets = targets.float().unsqueeze(1)

        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)

    average_loss_per_batch = running_loss / len(dataloader_train.dataset)
    print(f"Average training loss per batch: {average_loss_per_batch}")

    return average_loss_per_batch


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
    criterion = nn.CrossEntropyLoss()

    losses = []
    start_time = time.time()
    for i in range(num_epochs):
        epoch_loss = train_epoch(dataloader_train, optimizer, net, criterion,
                                 i)
        losses.append(epoch_loss)
    end_time = time.time()

    average_loss_per_epoch = np.mean(losses)

    if create_plot:
        plot_losses(losses)

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

    dataset_train = ImageFolder("../Data/clouds/clouds_train", transform=TRAIN_TRANSFORMS)
    dataset_test = ImageFolder("../Data/clouds/clouds_test", transform=TEST_TRANSFORMS)

    dataloader_train = data.DataLoader(dataset_train, batch_size=BATCH_SIZE, shuffle=True)
    dataloader_test = data.DataLoader(dataset_test, batch_size=BATCH_SIZE, shuffle=False)

    return dataloader_train, dataloader_test, dataset_train.classes

def main():

    dataloader_train, dataloader_test, class_names = get_data_loaders()
    net = MyCNN()
    optimizer = optim.AdamW(net.parameters(), lr=LEARNING_RATE)

    average_loss, training_time = train_model(dataloader_train, optimizer, net, EPOCHS, create_plot=True)

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
    
    per_class_f1_dict = {class_names[i]: round(f1_per_class_scores[i].item(), 4) for i in range(NUMBER_CLASSES)}

    print("Summary statistics:")
    print(
        f"Average training loss per epoch: {average_loss:.16f}"
    )
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
