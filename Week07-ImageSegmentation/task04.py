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

IMAGE_DIR = "../DATA/segmentation_cats_dogs/images"
MASK_DIR = "../DATA/segmentation_cats_dogs/annotations"
SEED = 42
BATCH_SIZE = 16


class CatsDogsSegmentationDataset(Dataset):

    def __init__(self, image_dir, mask_dir):
        self.image_paths = sorted(glob.glob(os.path.join(image_dir, "*.jpg")))
        self.mask_paths = sorted(glob.glob(os.path.join(mask_dir, "*.png")))

        assert (len(self.image_paths) == len(self.mask_paths))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert("RGB")
        mask = Image.open(self.mask_paths[idx]).convert("L")

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

def evaluate_model(val_loader, net, criterion):
    net.eval()
    total_loss = 0.0

    with torch.no_grad():
        for images, masks in tqdm(val_loader, desc="Evaluating..."):
            outputs = net(images)
            loss = criterion(outputs, masks)
            total_loss += loss.item() * images.size(0)

    average_loss = total_loss / len(val_loader.dataset)
    print(f"Average evaluation loss: {average_loss:.4f}")

    return average_loss


def main():
    dataset = CatsDogsSegmentationDataset(IMAGE_DIR, MASK_DIR)

    train_set, val_set, test_set = split_dataset(dataset)

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=True)


if __name__ == '__main__':
    main()
