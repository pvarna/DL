from torchvision.datasets import ImageFolder
from torchvision import transforms
from torch.utils import data
import torch.nn as nn
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from torch import optim

NUMBER_CLASSES = 7
LEARNING_RATE = 0.001
BATCH_SIZE = 16
EPOCHS = 20


class MyCNN(nn.Module):

    def __init__(self, input_size=64):
        super().__init__()
        self.feature_extractor = nn.ModuleList([
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3,
                      padding=1),
            nn.ELU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=32,
                      out_channels=64,
                      kernel_size=3,
                      padding=1),
            nn.ELU(),
            nn.MaxPool2d(kernel_size=2)
        ])

        self.classifier = nn.Linear(
            64 * (input_size // 4) * (input_size // 4),  # two MaxPool2d layers
            NUMBER_CLASSES)
        
    def forward(self, x):
        for layer in self.feature_extractor:
            x = layer(x)

        x = x.view(x.size(0), -1)  # Flatten
        x = self.classifier(x)
        return x

def show_first_image(dataset_train):
    dataloader_train = data.DataLoader(
        dataset_train,
        shuffle=True,
        batch_size=1,
    )

    image_tensor, _ = next(iter(dataloader_train))

    to_pil = transforms.ToPILImage()

    image = to_pil(image_tensor.squeeze(0))
    image.show()

def train_epoch(dataloader_train, optimizer, net, criterion, i):
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
    for i in range(num_epochs):
        epoch_loss = train_epoch(dataloader_train, optimizer, net, criterion, i)
        losses.append(epoch_loss)

    average_loss_per_epoch = np.mean(losses)
    print(f"Average training loss per epoch: {average_loss_per_epoch:.16f}")

    if create_plot:
        plot_losses(losses)

def main():
    train_transforms = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(45),
        transforms.RandomAutocontrast(),
        transforms.ToTensor(),
        transforms.Resize((64, 64))
    ])

    dataset_train = ImageFolder("../Data/clouds/clouds_train",
                                transform=train_transforms)
    
    show_first_image(dataset_train)

    dataloader_train = data.DataLoader(
        dataset_train,
        shuffle=True,
        batch_size=BATCH_SIZE,
    )

    net = MyCNN()
    train_model(dataloader_train, optim.AdamW(net.parameters(), lr=LEARNING_RATE), net, EPOCHS, create_plot=True)



if __name__ == '__main__':
    main()
