import numpy as np
import matplotlib.pyplot as plt
from skimage import data, color


def display_images(rows, cols, images, subtitles, title):
    fig, axes = plt.subplots(rows, cols)
    axes = axes.flat

    for ax, image, subtitle in zip(axes, images, subtitles):
        ax.imshow(image, cmap="gray")
        ax.set_title(subtitle)
        ax.axis("off")

    fig.suptitle(title)
    plt.tight_layout()
    plt.show()


def first_plot(img):
    grayscale_img = color.rgb2gray(img)
    flipped_horizontally_img = np.fliplr(img)
    flipped_vertically_img = np.flipud(img)

    imgs = [
        img, grayscale_img, flipped_horizontally_img, flipped_vertically_img
    ]
    subtitles = ["Original", "Grayscaled", "Horizontal flip", "Vertical flip"]
    title = "Playing with images"

    display_images(2, 2, imgs, subtitles, title)


def second_plot(img):
    channels = ['Red', 'Green', 'Blue']
    channel_data = [img[:, :, i] for i in range(3)]

    fig, axs = plt.subplots(3, 3)

    for i in range(3):
        axs[0, i].imshow(channel_data[i])
        axs[0, i].set_title(channels[i])
        axs[0, i].axis('off')

        axs[1, i].imshow(channel_data[i], cmap="gray")
        axs[1, i].set_title(channels[i])
        axs[1, i].axis('off')

        axs[2, i].hist(channel_data[i].ravel(), bins=256)
        axs[2, i].set_title(f"{channels[i]} pixel distribution")
        axs[2, i].set_xlabel("Intensity")
        axs[2, i].set_ylabel("Number of pixels")

    fig.suptitle("Playing with color intensities", fontsize=16)
    plt.tight_layout()
    plt.show()


def main():
    rocket_img = data.rocket()

    first_plot(rocket_img)
    second_plot(rocket_img)


if __name__ == '__main__':
    main()
