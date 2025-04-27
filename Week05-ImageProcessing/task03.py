import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from skimage import color, filters, exposure, data


def get_edge_detected_image(original_img):
    grayscale_img = color.rgb2gray(original_img)

    return filters.sobel(grayscale_img)


def get_gaussian_blurred_image(original_img):
    np_img = np.array(original_img)

    return filters.gaussian(np_img)


def get_histogram_equalized_image(original_img):
    np_img = np.array(original_img)

    return exposure.equalize_hist(np_img)


def get_clahe_image(original_img):
    np_img = np.array(original_img)

    return exposure.equalize_adapthist(np_img, clip_limit=0.03)


def display_images(images, subtitles, title):
    assert (len(images) == 10 and len(subtitles) == 10)

    fig, axes = plt.subplots(5, 2)
    axes = axes.flat

    for ax, image, subtitle in zip(axes, images, subtitles):
        ax.imshow(image, cmap="gray")
        ax.set_title(subtitle)
        ax.axis("off")

    fig.suptitle(title)
    plt.tight_layout()
    plt.show()


def main():
    soaps_img = Image.open("../DATA/w05_soaps.jpg")
    buildings_img = Image.open("../DATA/w05_building_image.jpg")
    chest_xray_img = Image.open("../DATA/w05_xray.png")
    aerial_img = Image.open("../DATA/w05_aerial.png")
    coffee_img = data.coffee()

    edge_detected_soaps_img = get_edge_detected_image(soaps_img)
    gaussian_blurred_buildings_img = get_gaussian_blurred_image(buildings_img)
    histogram_equalized_chest_xray_img = get_histogram_equalized_image(
        chest_xray_img)
    histogram_equalized_aerial_img = get_histogram_equalized_image(aerial_img)
    clahe_coffee_img = get_clahe_image(coffee_img)

    images = [
        soaps_img, edge_detected_soaps_img, buildings_img,
        gaussian_blurred_buildings_img, chest_xray_img,
        histogram_equalized_chest_xray_img, aerial_img,
        histogram_equalized_aerial_img, coffee_img, clahe_coffee_img
    ]

    subtitles = [
        "Original", "Edge detection", "Original", "Image blurring", "Original",
        "Standart histogram equalization", "Original",
        "Standart histogram equalization", "Original", "CLAHE"
    ]

    display_images(images, subtitles, "Filtering images")


if __name__ == '__main__':
    main()
