import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from skimage import color, filters, restoration


def denoise_bilateral(original_img):
    np_img = np.array(original_img)

    if np_img.ndim == 3:
        grayscale_img = color.rgb2gray(np_img)
    else:
        grayscale_img = np_img

    return restoration.denoise_bilateral(grayscale_img)


def binarize_image(original_img):
    threshold_value = filters.threshold_otsu(original_img)
    binary_image = original_img > threshold_value
    return binary_image


def display_images(images, subtitles, title):
    assert (len(images) == 4 and len(subtitles) == 4)

    fig, axes = plt.subplots(2, 2)
    axes = axes.flat

    for ax, image, subtitle in zip(axes, images, subtitles):
        ax.imshow(image, cmap="gray")
        ax.set_title(subtitle)
        ax.axis("off")

    fig.suptitle(title)
    plt.tight_layout()
    plt.show()


def main():
    capital_case_r_img = Image.open("../DATA/w05_r5.png")
    world_map_img = Image.open("../DATA/w05_continents.jpg")

    binarized_denoised_capiral_case_r_img = binarize_image(
        denoise_bilateral(capital_case_r_img))
    binarized_denoised_world_map_img = binarize_image(
        denoise_bilateral(world_map_img))

    images = [
        capital_case_r_img, binarized_denoised_capiral_case_r_img,
        world_map_img, binarized_denoised_world_map_img
    ]

    subtitles = ["Original", "Transformed", "Original", "Transformed"]

    display_images(images, subtitles, "Noise removal")


if __name__ == '__main__':
    main()
