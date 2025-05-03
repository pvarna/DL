import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import os
import random
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms

PATH = os.path.join("..", "DATA", "pokemon_sprites")
NUM_SAMPLES = 64


class PokemonDataset(Dataset):

    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.image_paths = []

        for generation in range(1, 10):
            gen_dir = os.path.join(root_dir, f"Generation_{generation}")
            for fname in os.listdir(gen_dir):
                if fname.lower().endswith(".png"):
                    self.image_paths.append(os.path.join(gen_dir, fname))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert("RGBA")

        image = transforms.ToTensor()(image)

        return image, image_path

def display_images(rows, cols, images, subtitles, title):
    fig, axes = plt.subplots(rows, cols)
    axes = axes.flat

    for ax, image, subtitle in zip(axes, images, subtitles):
        ax.imshow(image)
        ax.set_title(subtitle)
        ax.axis("off")

    fig.suptitle(title)
    plt.tight_layout()
    plt.show()

def main():
    dataset = PokemonDataset(PATH)

    indices = random.sample(range(len(dataset)), NUM_SAMPLES)

    images = []
    image_paths = []

    for idx in indices:
        image, path = dataset[idx]
        images.append(transforms.ToPILImage()(image))
        image_paths.append(os.path.basename(path))

    display_images(8, 8, images, image_paths, title="Exploring the \"pokemon_sprites\" dataset")


if __name__ == '__main__':
    main()
