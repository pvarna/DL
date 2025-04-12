import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from skimage import restoration, data, filters, util, transform
from skimage.feature import Cascade


def detect_faces(img, detector):
    np_img = np.array(img)

    detected_faces_coords = detector.detect_multi_scale(img=np_img,
                                                        scale_factor=1.25,
                                                        step_ratio=1.5,
                                                        min_size=(10, 10),
                                                        max_size=(100, 100))

    return detected_faces_coords


def blur_faces(img, face_coords, sigma):
    np_img = np.array(img)

    blurred_img = np_img.copy()

    for face in face_coords:
        r, c, h, w = face['r'], face['c'], face['height'], face['width']
        face_region = np_img[r:r + h, c:c + w]

        blurred_face = filters.gaussian(face_region,
                                        sigma=sigma,
                                        channel_axis=-1)
        blurred_face = util.img_as_ubyte(blurred_face)

        blurred_img[r:r + h, c:c + w] = blurred_face

    return blurred_img


def fix_image(img):
    np_img = np.array(img)

    rotated_img = transform.rotate(np_img, 20)
    gaussian_blurred_img = filters.gaussian(rotated_img,
                                            sigma=1,
                                            channel_axis=-1)

    mask = np.zeros(gaussian_blurred_img.shape[:2], dtype=bool)
    y_min1, y_max1 = 315, 360
    x_min1, x_max1 = 140, 175

    y_min2, y_max2 = 450, 480
    x_min2, x_max2 = 470, 495

    y_min3, y_max3 = 130, 160
    x_min3, x_max3 = 350, 370
    mask[y_min1:y_max1, x_min1:x_max1] = True
    mask[y_min2:y_max2, x_min2:x_max2] = True
    mask[y_min3:y_max3, x_min3:x_max3] = True
    restored_img = restoration.inpaint_biharmonic(gaussian_blurred_img,
                                                  mask,
                                                  channel_axis=-1)

    return restored_img


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
    music_group_img = Image.open("../DATA/w05_music_group.jpg")
    graduation_img = Image.open("../DATA/w05_graduation.jpg")

    trained_file = data.lbp_frontal_face_cascade_filename()
    detector = Cascade(xml_file=trained_file)

    coords = detect_faces(music_group_img, detector)
    blurred_img = blur_faces(music_group_img, coords, sigma=20)

    fixed_img = fix_image(graduation_img)

    images = [music_group_img, blurred_img, graduation_img, fixed_img]
    subtitles = ["Original", "Blurred faces", "Original", "Restored"]
    title = "Real-world applications"

    display_images(images, subtitles, title)


if __name__ == '__main__':
    main()
