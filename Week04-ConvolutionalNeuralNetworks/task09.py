import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms
import glob
import os
from PIL import Image


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


def main():
    train_transforms = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Resize((64, 64))])

    dataset = OmniglotDataset("../DATA/omniglot_train", train_transforms)
    size = len(dataset)
    image_tensor, _, _ = last_element = dataset[-1]
    print(f"Number of instances: {size}")
    print(f"Last item: {last_element}")

    print(f"Shape of the last image: {image_tensor.shape}")


if __name__ == '__main__':
    main()
