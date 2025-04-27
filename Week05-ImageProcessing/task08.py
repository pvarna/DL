import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from skimage import color, feature


def get_image_with_detected_edges(original_img, sigma):
    grayscale_img = color.rgb2gray(original_img)
    np_img = np.array(grayscale_img)

    return feature.canny(np_img, sigma)


def get_corners(building_img, min_distance, threshold_rel):
    np_img = np.array(building_img)

    if np_img.ndim == 3:
        grayscale_img = color.rgb2gray(np_img)
    else:
        grayscale_img = np_img

    coords = feature.corner_peaks(feature.corner_harris(grayscale_img),
                                  min_distance=min_distance,
                                  threshold_rel=threshold_rel)
    print(
        f"With {min_distance=} and {threshold_rel=} we detect a total of {len(coords)} corners in the image."
    )
    return coords


def display_images(images, corner_coords, subtitles, title):
    assert (len(images) == 10 and len(subtitles) == 10
            and len(corner_coords) == 10)

    fig, axes = plt.subplots(2, 5)
    axes = axes.flat

    for ax, image, coords, subtitle in zip(axes, images, corner_coords,
                                           subtitles):
        ax.imshow(image, cmap="gray")
        ax.set_title(subtitle)
        ax.axis("off")

        if coords is not None:
            ax.plot(coords[:, 1], coords[:, 0], '+r', markersize=15)

    fig.suptitle(title)
    plt.tight_layout()
    plt.show()


def main():
    grapefruits_img = Image.open("../DATA/w05_grapefruit.jpg")
    building_img = Image.open("../DATA/w05_single_building.jpg")

    sigma1, sigma2, sigma3, sigma4 = 0.8, 1, 1.8, 2.2
    grapefruits_edges1_img = get_image_with_detected_edges(
        grapefruits_img, sigma1)
    grapefruits_edges2_img = get_image_with_detected_edges(
        grapefruits_img, sigma2)
    grapefruits_edges3_img = get_image_with_detected_edges(
        grapefruits_img, sigma3)
    grapefruits_edges4_img = get_image_with_detected_edges(
        grapefruits_img, sigma4)

    # display_images(images, subtitles, title)

    min_distance1, min_distance2, min_distance3, min_distance4 = 10, 10, 20, 60
    threshold_rel1, threshold_rel2, threshold_rel3, threshold_rel4 = 0.01, 0.02, 0.03, 0.02

    corner_coords1 = get_corners(building_img, min_distance1, threshold_rel1)
    corner_coords2 = get_corners(building_img, min_distance2, threshold_rel2)
    corner_coords3 = get_corners(building_img, min_distance3, threshold_rel3)
    corner_coords4 = get_corners(building_img, min_distance4, threshold_rel4)

    images = [
        grapefruits_img, grapefruits_edges1_img, grapefruits_edges2_img,
        grapefruits_edges3_img, grapefruits_edges4_img, building_img,
        building_img, building_img, building_img, building_img
    ]

    subtitles = [
        "Original", f"sigma={sigma1}", f"sigma={sigma2}", f"sigma={sigma3}",
        f"sigma={sigma4}", "Original",
        f"min_distance={min_distance1} | threshold_rel={threshold_rel1}",
        f"min_distance={min_distance2} | threshold_rel={threshold_rel2}",
        f"min_distance={min_distance3} | threshold_rel={threshold_rel3}",
        f"min_distance={min_distance4} | threshold_rel={threshold_rel4}"
    ]

    coords = [
        None, None, None, None, None, None, corner_coords1, corner_coords2,
        corner_coords3, corner_coords4
    ]

    title = "Edge detection with the Canny algorithm"

    display_images(images, coords, subtitles, title)


if __name__ == '__main__':
    main()
