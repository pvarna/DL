import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import os
from tqdm import tqdm
from torch.optim import AdamW
from torchmetrics.image.fid import FrechetInceptionDistance

ROOT_DIR = os.path.join("..", "DATA", "pokemon_sprites")
Z_DIM = 64
IMAGE_SIZE = 64
NUM_EPOCHS = 20
BATCH_SIZE = 32
LEARNING_RATE = 0.0001
NUM_CHANNELS = 3


def main():
    pass


if __name__ == '__main__':
    main()
