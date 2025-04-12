import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from skimage import restoration, util, filters, segmentation, color


def add_random_noise(original_img):
    np_img = np.array(original_img)

    return util.random_noise(np_img)


def blur_with_gaussian_filter(original_img):
    np_img = np.array(original_img)

    return filters.gaussian(np_img, channel_axis=-1)


def denoise_bilateral(original_img):
    np_img = np.array(original_img)

    return restoration.denoise_bilateral(np_img, channel_axis=-1)


def segment_image(original_img, n):
    np_img = np.array(original_img)

    segments = segmentation.slic(np_img, n, channel_axis=-1)
    return color.label2rgb(segments, np_img, kind='avg')


def display_images(images, subtitles, title):
    assert (len(images) == 8 and len(subtitles) == 8)

    fig, axes = plt.subplots(4, 2)
    axes = axes.flat

    for ax, image, subtitle in zip(axes, images, subtitles):
        ax.imshow(image, cmap="gray")
        ax.set_title(subtitle)
        ax.axis("off")

    fig.suptitle(title)
    plt.tight_layout()
    plt.show()


def main():
    fruits_img = Image.open("../DATA/w05_fruits_generic.png")
    dog_img = Image.open("../DATA/w05_miny.png")
    landscape_img = Image.open("../DATA/w05_landscape.jpg")
    lady_img = Image.open("../DATA/w05_lady.jpg")

    fruits_noise_img = add_random_noise(fruits_img)
    blurred_dog_img = blur_with_gaussian_filter(dog_img)
    bilateral_landscape_img = denoise_bilateral(landscape_img)
    segmented_lady_img = segment_image(lady_img, 400)

    images = [
        fruits_img, fruits_noise_img, dog_img, blurred_dog_img, landscape_img,
        bilateral_landscape_img, lady_img, segmented_lady_img
    ]

    subtitles = [
        "Original", "Noisy image", "Noisy", "Denoised image", "Noisy",
        "Denoised image", "Original", "Segmented image, 400 superpixels"
    ]

    title = "Adding and removing noise | Segmenting images"

    display_images(images, subtitles, title)


if __name__ == '__main__':
    main()
