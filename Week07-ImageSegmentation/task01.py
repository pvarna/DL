import os

from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms
import torch
import random


def choose_random_images(folder_path, n=5):
    all_images = os.listdir(folder_path)

    selected_images = random.sample(all_images, n)
    selected_paths = [
        os.path.join(folder_path, img) for img in selected_images
    ]

    return selected_paths


def to_pil_image(image_path, mask_path):
    image = Image.open(image_path)
    mask = Image.open(mask_path)

    transform = transforms.Compose([transforms.ToTensor()])
    image_tensor = transform(image)
    mask_tensor = transform(mask)

    binary_mask = torch.where(
        mask_tensor == 1 / 255,
        torch.tensor(1.0),
        torch.tensor(0.0),
    )

    mask = transforms.ToPILImage()(binary_mask)

    return transforms.ToPILImage()(image_tensor * binary_mask)


def display_images(rows, cols, pil_images, segmented_pil_images, subtitles,
                   title):
    assert (rows == len(pil_images))
    fig, axes = plt.subplots(rows, cols)

    for idx in range(rows):
        axes[idx][0].imshow(pil_images[idx], cmap="gray")
        axes[idx][0].set_title(subtitles[idx])
        axes[idx][0].axis("off")

        axes[idx][1].imshow(segmented_pil_images[idx], cmap="gray")
        axes[idx][1].set_title(subtitles[idx] + " segmented")
        axes[idx][1].axis("off")

    fig.suptitle(title)
    plt.tight_layout()
    plt.show()


def main():
    image_folder = os.path.join('..', 'DATA', 'segmentation_cats_dogs',
                                'images')
    image_paths = choose_random_images(image_folder, 5)
    mask_paths = [
        path.replace("/images/", "/annotations/").replace(".jpg", ".png")
        for path in image_paths
    ]

    pil_images = [Image.open(image_path) for image_path in image_paths]
    segmented_pil_images = [
        to_pil_image(image_path, mask_path)
        for (image_path, mask_path) in zip(image_paths, mask_paths)
    ]
    subtitles = [os.path.basename(path).split('.')[0] for path in image_paths]
    display_images(5, 2, pil_images, segmented_pil_images, subtitles,
                   "Exploring the dataset")


if __name__ == '__main__':
    main()
