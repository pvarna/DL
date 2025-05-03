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


class LinearGenerator(nn.Module):

    def __init__(self, input_dimensions, output_dimensions):
        super().__init__()

        self.generator = nn.Sequential(self.gen_block(input_dimensions, 256),
                                       self.gen_block(256, 512),
                                       self.gen_block(512, 1024),
                                       nn.Linear(1024, output_dimensions),
                                       nn.Sigmoid())

    def gen_block(self, input_dimensions, output_dimensions):
        return nn.Sequential(nn.Linear(input_dimensions, output_dimensions),
                             nn.BatchNorm1d(output_dimensions), nn.ReLU())

    def forward(self, x):
        return self.generator(x)


class LinearDiscriminator(nn.Module):

    def __init__(self, input_dimensions):
        super().__init__()

        self.discriminator = nn.Sequential(
            self.disc_block(input_dimensions, 1024),
            self.disc_block(1024, 512), self.disc_block(512, 256),
            nn.Linear(256, 1))

    def disc_block(self, input_dimensions, output_dimensions):
        return nn.Sequential(nn.Linear(input_dimensions, output_dimensions),
                             nn.LeakyReLU(0.2))

    def forward(self, x):
        return self.discriminator(x)


class PokemonDataset(Dataset):

    def __init__(self, root_dir, image_size):
        self.root_dir = root_dir
        self.image_paths = []

        for generation in range(1, 10):
            gen_dir = os.path.join(root_dir, f"Generation_{generation}")
            for fname in os.listdir(gen_dir):
                if fname.lower().endswith(".png"):
                    self.image_paths.append(os.path.join(gen_dir, fname))

        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert("RGB")

        image = self.transform(image)

        return image.view(-1)


def gen_loss(gen, disc, num_images, z_dim):
    noise = torch.randn(num_images, z_dim)
    fake = gen(noise)
    disc_pred = disc(fake)
    criterion = nn.BCEWithLogitsLoss()
    gen_loss = criterion(disc_pred, torch.ones_like(disc_pred))

    return gen_loss


def disc_loss(gen, disc, real, num_images, z_dim):
    criterion = nn.BCEWithLogitsLoss()
    noise = torch.randn(num_images, z_dim)
    fake = gen(noise)
    disc_pred_fake = disc(fake)
    fake_loss = criterion(disc_pred_fake, torch.zeros_like(disc_pred_fake))
    disc_pred_real = disc(real)
    real_loss = criterion(disc_pred_real, torch.ones_like(disc_pred_real))
    disc_loss = (real_loss + fake_loss) / 2

    return disc_loss


def train_epoch(dataloader, gen, disc, gen_opt, disc_opt, i):
    for real in tqdm(dataloader, desc=f"Training epoch {i}"):
        cur_batch_size = len(real)

        disc_opt.zero_grad()
        disc_loss_out = disc_loss(gen, disc, real, cur_batch_size, z_dim=Z_DIM)
        disc_loss_out.backward()
        disc_opt.step()

        gen_opt.zero_grad()
        gen_loss_out = gen_loss(gen, disc, cur_batch_size, z_dim=Z_DIM)
        gen_loss_out.backward()
        gen_opt.step()


def train_models(dataloader, gen, disc, gen_opt, disc_opt, num_epochs):
    for epoch in range(num_epochs):
        print(f"Training epoch {epoch+1}/{num_epochs}")
        train_epoch(dataloader, gen, disc, gen_opt, disc_opt, epoch + 1)


def evaluate_model(gen):
    num_images_to_generate = 9
    rows, cols = 3, 3

    noise = torch.randn(num_images_to_generate, Z_DIM)
    with torch.no_grad():
        fake = gen(noise)
    print(f'Generated shape: {fake.shape}')

    images = []
    for i in range(num_images_to_generate):
        flat_image_tensor = fake[i]
        image_tensor_3d = flat_image_tensor.view(3, IMAGE_SIZE, IMAGE_SIZE)
        image_permuted = image_tensor_3d.permute(1, 2, 0)
        images.append(image_permuted)

    display_images(rows, cols, images, "Generated Pokemons")


def display_images(rows, cols, images, title):
    fig, axes = plt.subplots(rows, cols)
    axes = axes.flat

    for ax, image in zip(axes, images):
        ax.imshow(image, cmap="gray")
        ax.axis("off")

    fig.suptitle(title)
    plt.tight_layout()
    plt.show()

def calculate_fid(generator, dataloader, z_dim, image_size, num_channels):
    fid = FrechetInceptionDistance(feature=64)

    generator.eval()

    for real_flat in tqdm(dataloader, desc="Processing batches for FID"):
        cur_batch_size = real_flat.shape[0]
        real_images = real_flat.view(cur_batch_size, num_channels, image_size, image_size)
        real_images_uint8 = (real_images.clamp(0, 1) * 255).to(torch.uint8)

        with torch.no_grad():
            noise = torch.randn(cur_batch_size, z_dim)
            fake_flat = generator(noise)

        fake_images = fake_flat.view(cur_batch_size, num_channels, image_size, image_size)
        fake_images_uint8 = (fake_images.clamp(0, 1) * 255).to(torch.uint8)

        fid.update(real_images_uint8, real=True)
        fid.update(fake_images_uint8, real=False)

    print("Computing final FID score...")
    fid_score = fid.compute()
    print(f"FID Score (feature=64): {fid_score.item():.4f}")
    return fid_score.item()

def main():
    dataset = PokemonDataset(ROOT_DIR, IMAGE_SIZE)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    image_dimension = NUM_CHANNELS * IMAGE_SIZE * IMAGE_SIZE

    gen = LinearGenerator(Z_DIM, image_dimension)
    disc = LinearDiscriminator(image_dimension)

    gen_opt = AdamW(gen.parameters(), lr=LEARNING_RATE)
    disc_opt = AdamW(disc.parameters(), lr=LEARNING_RATE)

    # train_models(dataloader, gen, disc, gen_opt, disc_opt, NUM_EPOCHS)

    evaluate_model(gen)
    calculate_fid(gen, dataloader, Z_DIM, IMAGE_SIZE, NUM_CHANNELS)


if __name__ == '__main__':
    main()
