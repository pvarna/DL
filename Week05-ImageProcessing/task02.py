import matplotlib.pyplot as plt
from PIL import Image
from skimage import color, filters


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


def plot_otsu_global_thresholding(image, title):
    gray = color.rgb2gray(image)

    fig, axes = plt.subplots(1, 2)

    axes[0].imshow(image, cmap='gray')
    axes[0].set_title("Original")
    axes[0].axis('off')

    thresh = filters.threshold_otsu(gray)
    binary = gray > thresh

    axes[1].imshow(binary, cmap='gray')
    axes[1].set_title("Binarized image")
    axes[1].axis('off')

    fig.suptitle(title)
    plt.tight_layout()
    plt.show()


def first_plot(img1, img2):

    def get_binary_images(img):
        grayscale_img = color.rgb2gray(img)

        global_threshold = filters.threshold_otsu(grayscale_img)
        global_binary_img = grayscale_img > global_threshold

        local_threshold = filters.threshold_local(grayscale_img,
                                                  block_size=21,
                                                  offset=0)
        local_binary_img = grayscale_img > local_threshold

        return global_binary_img, local_binary_img

    global_binary_img1, local_binary_img1 = get_binary_images(img1)
    global_binary_img2, local_binary_img2 = get_binary_images(img2)

    imgs = [
        img1, global_binary_img1, local_binary_img1, img2, global_binary_img2,
        local_binary_img2
    ]
    subtitles = [
        "Original", "Global Thresholding", "Local Thresholding", "Original",
        "Global Thresholding", "Local Thresholding"
    ]
    title = "Thresholding images"

    display_images(2, 3, imgs, subtitles, title)


def second_plot(img, thresholding_methods):
    grayscale_img = color.rgb2gray(img)

    binary_imgs = []
    for method in thresholding_methods.values():
        threshold = method(grayscale_img)
        binary_img = grayscale_img > threshold
        binary_imgs.append(binary_img)

    imgs = [grayscale_img] + binary_imgs
    subtitles = ["Original"] + list(thresholding_methods.keys())
    title = "Fruits"

    display_images(4, 2, imgs, subtitles, title)


def third_plot(img):
    grayscale_img = color.rgb2gray(img)

    threshold = filters.threshold_otsu(grayscale_img)
    binary_img = grayscale_img > threshold

    imgs = [img, binary_img]
    subtitles = ["Original", "Binarized image"]
    title = "Shapes"

    display_images(1, 2, imgs, subtitles, title)


methods = {
    "Isodata": filters.threshold_isodata,
    "Li": filters.threshold_li,
    "Mean": filters.threshold_mean,
    "Minimum": filters.threshold_minimum,
    "Otsu": filters.threshold_otsu,
    "Triangle": filters.threshold_triangle,
    "Yen": filters.threshold_yen
}


def main():
    chess_pieces_img = Image.open("../DATA/w05_chess_pieces.png")
    text_page_img = Image.open("../DATA/w05_text_page.png")
    fruits_img = Image.open("../DATA/w05_fruits.png")
    shapes_img = Image.open("../DATA/w05_shapes.png")

    first_plot(chess_pieces_img, text_page_img)
    second_plot(fruits_img, methods) # I would recomment the Li thresholding
    third_plot(shapes_img)


if __name__ == '__main__':
    main()
