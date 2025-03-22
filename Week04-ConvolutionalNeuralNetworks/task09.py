import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms


class OmniglotDataset(Dataset):

    def __init__(self, path_to_images, transformations):
        super().__init__()
        self.image_folder = ImageFolder(path_to_images, transformations)

    def __len__(self):
        return len(self.image_folder)

    def __getitem__(self, index):
        image_as_tensor, label = self.image_folder[index]

        one_hot_encoding_tensor = torch.zeros(len(self.image_folder.classes))
        one_hot_encoding_tensor[label] = 1

        return image_as_tensor, one_hot_encoding_tensor, label


def main():
    train_transforms = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Resize((64, 64))])

    dataset = OmniglotDataset("../DATA/omniglot_train", train_transforms)
    size = dataset.__len__()
    image_tensor, _, _ = last_element = dataset.__getitem__(size - 1)
    print(f"Number of instances: {size}")
    print(f"Last item: {last_element}")

    print(f"Shape of the last image: {image_tensor.shape}")


if __name__ == '__main__':
    main()
