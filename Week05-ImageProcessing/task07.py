import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from skimage import filters, color, data, measure


def get_binary_image_and_contours(original_img, invert=False):
    np_img = np.array(original_img)

    if np_img.ndim == 3:
        grayscale_img = color.rgb2gray(np_img)
    else:
        grayscale_img = np_img

    threshold_value = filters.threshold_otsu(grayscale_img)

    binary_img = grayscale_img > threshold_value
    binary_img = np.invert(binary_img) if invert else binary_img

    contours = measure.find_contours(binary_img)

    return binary_img, contours


def display_images_with_contours(images, contours, subtitles, title):
    assert (len(images) == 4 and len(subtitles) == 4)

    fig, axes = plt.subplots(2, 2)
    axes = axes.flat

    for ax, image, image_contours, subtitle in zip(axes, images, contours,
                                                   subtitles):
        ax.imshow(image, cmap="gray")

        for contour in image_contours:
            ax.plot(contour[:, 1],
                    contour[:, 0],
                    color=np.random.rand(3, ),
                    lw=2)
        ax.set_title(subtitle)
        ax.axis("off")

    fig.suptitle(title)
    plt.tight_layout()
    plt.show()


def main():
    horse_img = data.horse()
    dice_img = Image.open("../DATA/w05_dice.png")

    horse_binary_img, horse_contours = get_binary_image_and_contours(
        horse_img, invert=True)
    dice_binary_img, dice_contours = get_binary_image_and_contours(dice_img,
                                                                   invert=True)

    images = [horse_img, horse_binary_img, dice_img, dice_binary_img]
    contours = [[], horse_contours, [], dice_contours]
    subtitles = ["Original", "Contours", "Original", "Contours"]
    title = "Finding contours"

    display_images_with_contours(images, contours, subtitles, title)

    dots = 0
    for c in dice_contours:
        # print(c.shape)
        x, _ = c.shape
        if x > 40 and x < 60:
            dots += 1

    print(f"Number of dots in dice: {dots}.")


if __name__ == '__main__':
    main()
