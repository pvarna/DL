import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from skimage import restoration


def restore_black_areas(damaged_img):
    np_img = np.array(damaged_img)
    mask = np.all(np_img == [0, 0, 0], axis=-1)

    return restoration.inpaint_biharmonic(np_img, mask, channel_axis=-1)


def remove_logo(img_with_logo):
    np_img = np.array(img_with_logo)
    mask = np.zeros(np_img.shape[:2], dtype=bool)
    y_min, y_max = 220, 275
    x_min, x_max = 360, 420
    mask[y_min:y_max, x_min:x_max] = True

    return restoration.inpaint_biharmonic(np_img, mask, channel_axis=-1)


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
    damaged_astonaut_img = Image.open("../DATA/w05_damaged_astro.png")
    logo_img = Image.open("../DATA/w05_logo_image.png")

    fixed_astronaut_img = restore_black_areas(damaged_astonaut_img)
    logoless_img = remove_logo(logo_img)

    images = [
        damaged_astonaut_img, fixed_astronaut_img, logo_img, logoless_img
    ]
    subtitles = ["Original", "Image restored", "Original", "Image restored"]

    display_images(images, subtitles, "Image restoration")


if __name__ == '__main__':
    main()
