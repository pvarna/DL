import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image
from torchvision.transforms import functional
from tqdm import tqdm
from torchmetrics.classification import Accuracy, JaccardIndex

IMAGE_DIR = "../DATA/segmentation_cats_dogs/images"
MASK_DIR = "../DATA/segmentation_cats_dogs/annotations"
SEED = 42
BATCH_SIZE = 16
LEARNING_RATE = 0.001
SIZE = (256, 256)


class MyUNet(nn.Module):

    def __init__(self, input_channels, output_channels):
        super().__init__()
        self.initial_conv_layer = nn.Conv2d(input_channels,
                                            64,
                                            kernel_size=3,
                                            padding=1)

        self.encoder = nn.ModuleList([
            self.my_convolutional_block(64, 128),
            self.my_convolutional_block(128, 256),
            self.my_convolutional_block(256, 512)
        ])

        self.pooling_layers = nn.ModuleList([
            nn.MaxPool2d(kernel_size=2),
            nn.MaxPool2d(kernel_size=2),
            nn.MaxPool2d(kernel_size=2)
        ])

        self.upconv_layers = nn.ModuleList([
            nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2),
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        ])

        self.decoder = nn.ModuleList([
            # Input: 256 (from upconv[0]) + 512 (from encoder[2] skip) = 768
            self.my_convolutional_block(256 + 512, 256),
            # Input: 128 (from upconv[1]) + 256 (from encoder[1] skip) = 384
            self.my_convolutional_block(128 + 256, 128),
            # Input: 64 (from upconv[2]) + 128 (from encoder[0] skip) = 192
            self.my_convolutional_block(64 + 128, 64)
        ])

        self.final_conv_layer = nn.Conv2d(64, output_channels, kernel_size=1)

    def forward(self, x):
        skip_connections = []

        x = self.initial_conv_layer(x)
        skip_connections.append(x)

        for encoder_block, pool in zip(self.encoder, self.pooling_layers):
            x = encoder_block(x)
            skip_connections.append(x)
            x = pool(x)

        for upconv, decoder_block in zip(self.upconv_layers, self.decoder):
            x = upconv(x)
            skip_x = skip_connections.pop()
            x = torch.cat((x, skip_x), dim=1)
            x = decoder_block(x)

        x = self.final_conv_layer(x)
        return x

    def my_convolutional_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU())


class CatsDogsSegmentationDataset(Dataset):

    def __init__(self, image_dir, mask_dir):
        self.image_paths = sorted(glob.glob(os.path.join(image_dir, "*.jpg")))
        self.mask_paths = sorted(glob.glob(os.path.join(mask_dir, "*.png")))

        assert (len(self.image_paths) == len(self.mask_paths))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert("RGB").resize(SIZE)
        mask = Image.open(self.mask_paths[idx]).convert("L").resize(SIZE)

        image = functional.to_tensor(image)
        mask = functional.pil_to_tensor(mask)[0]
        mask = mask.long() - 1  # Shift mask labels from [1, 2, 3] to [0, 1, 2]

        return image, mask.long()


def split_dataset(dataset):
    torch.manual_seed(SEED)

    total_size = len(dataset)

    # 80/10/10
    train_size = int(0.8 * total_size)
    val_size = int(0.1 * total_size)
    test_size = total_size - train_size - val_size

    train_set, val_set, test_set = random_split(
        dataset, [train_size, val_size, test_size])

    print(f"Train size: {len(train_set)}")
    print(f"Validation size: {len(val_set)}")
    print(f"Test size: {len(test_set)}")

    return train_set, val_set, test_set


def train_epoch(train_loader, net, optimizer, criterion, i):
    net.train()
    total_loss = 0.0

    for images, masks in tqdm(train_loader, desc=f"Training epoch {i}..."):
        optimizer.zero_grad()
        outputs = net(images)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)

    average_loss = total_loss / len(train_loader.dataset)
    print(f"Average training loss during epoch {i}: {average_loss:.4f}")

    return average_loss


def evaluate_model(loader, net, criterion):
    net.eval()
    total_loss = 0.0

    accuracy_metric = Accuracy(num_classes=3, task="multiclass")
    iou_metric = JaccardIndex(num_classes=3, task="multiclass")

    with torch.no_grad():
        for images, masks in tqdm(loader, desc="Evaluating..."):
            outputs = net(images)
            loss = criterion(outputs, masks)
            total_loss += loss.item() * images.size(0)

            preds = outputs.argmax(dim=1)
            accuracy_metric.update(preds, masks)
            iou_metric.update(preds, masks)

    average_loss = total_loss / len(loader.dataset)
    acc = accuracy_metric.compute().item()
    iou = iou_metric.compute().item()

    print(f"Average evaluation loss: {average_loss:.4f}")
    print(f"Pixel Accuracy: {acc:.4f}")
    print(f"Mean IoU: {iou:.4f}")

    return average_loss


def train_model(train_loader, val_loader, net, optimizer, criterion,
                num_epochs):
    train_losses = []
    val_losses = []

    for epoch in range(num_epochs):
        print(f"Epoch {epoch}/{num_epochs}")
        train_loss = train_epoch(train_loader, net, optimizer, criterion,
                                 epoch)
        val_loss = evaluate_model(val_loader, net, criterion)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

    plot_losses(train_losses, val_losses, title="Training vs Validation Loss")


def plot_losses(train_losses, val_losses, title):
    plt.figure(figsize=(8, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title(title)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def main():
    dataset = CatsDogsSegmentationDataset(IMAGE_DIR, MASK_DIR)

    train_set, val_set, test_set = split_dataset(dataset)

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False)

    net = MyUNet(input_channels=3, output_channels=3)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(net.parameters(), lr=LEARNING_RATE)

    # train_model(train_loader,
    #             val_loader,
    #             net,
    #             optimizer,
    #             criterion,
    #             num_epochs=10)
    # 2:40 hours for a single epoch 💀
    
    evaluate_model(test_loader, net, criterion)


if __name__ == '__main__':
    main()
