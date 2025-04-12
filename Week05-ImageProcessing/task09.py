import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
from skimage import color, segmentation, data
from skimage.feature import Cascade


def show_image(image, rectangles, title='Image', cmap_type='gray'):
    fig, ax = plt.subplots()
    ax.imshow(image, cmap=cmap_type)
    ax.set_title(title)
    ax.axis('off')

    for rect in rectangles:
        r = rect["r"]
        c = rect["c"]
        width = rect["width"]
        height = rect["height"]

        rect_patch = patches.Rectangle((c, r),
                                       width,
                                       height,
                                       edgecolor="red",
                                       facecolor="none",
                                       linewidth=2)
        ax.add_patch(rect_patch)

    plt.show()


def detect_faces(img, detector, min_size, max_size):
    np_img = np.array(img)

    detected_faces_coords = detector.detect_multi_scale(img=np_img,
                                                        scale_factor=1.2,
                                                        step_ratio=1,
                                                        min_size=(min_size,
                                                                  min_size),
                                                        max_size=(max_size,
                                                                  max_size))

    face_images = []
    for face in detected_faces_coords:
        r = face['r']
        c = face['c']
        h = face['height']
        w = face['width']

        cropped = np_img[r:r + h, c:c + w]
        face_images.append(Image.fromarray(cropped))

    return detected_faces_coords, face_images


def segment_image(original_img, n):
    np_img = np.array(original_img)

    segments = segmentation.slic(np_img, n, channel_axis=-1)
    return color.label2rgb(segments, np_img, kind='avg')


def display_images(imgs, segmented_imgs, detected_faces_per_img, title):
    rows = len(imgs)
    cols = max([len(coords) for coords, _ in detected_faces_per_img]) + 2

    fig, axes = plt.subplots(rows, cols)

    for i, (img, segmented_img, (faces_coords, cropped_faces)) in enumerate(
            zip(imgs, segmented_imgs, detected_faces_per_img)):
        axes[i][0].imshow(img, cmap="gray")
        axes[i][0].set_title("Original")
        axes[i][0].axis("off")

        segmented_img = img if segmented_img is None else segmented_img
        axes[i][1].imshow(segmented_img, cmap="gray")
        axes[i][1].set_title("Face image")
        axes[i][1].axis("off")

        for face in faces_coords:
            r = face["r"]
            c = face["c"]
            width = face["width"]
            height = face["height"]

            rect_patch = patches.Rectangle((c, r),
                                           width,
                                           height,
                                           edgecolor="red",
                                           facecolor="none",
                                           linewidth=2)
            axes[i][1].add_patch(rect_patch)

        for j, face in enumerate(cropped_faces):
            axes[i][j + 2].imshow(face, cmap="gray")
            axes[i][j + 2].set_title("Face detected")
            axes[i][j + 2].axis("off")

        for j in range(len(faces_coords) + 2, cols):
            axes[i][j].set_visible(False)

    fig.suptitle(title)
    plt.tight_layout()
    plt.show()


def main():
    person_at_night_img = Image.open("../DATA/w05_person_at_night.jpg")
    friends_img = Image.open("../DATA/w05_friends.jpg")
    profile_img = Image.open("../DATA/w05_profile.jpg")

    profile_segmented_img = segment_image(profile_img, 100)

    trained_file = data.lbp_frontal_face_cascade_filename()
    detector = Cascade(xml_file=trained_file)

    person_at_night_detected_faces = detect_faces(person_at_night_img,
                                                  detector,
                                                  min_size=10,
                                                  max_size=50)
    friends_detected_faces = detect_faces(friends_img,
                                          detector,
                                          min_size=10,
                                          max_size=50)
    profile_detected_faces = detect_faces(profile_segmented_img,
                                          detector,
                                          min_size=200,
                                          max_size=400)

    imgs = [person_at_night_img, friends_img, profile_img]
    segmented_imgs = [None, None, profile_segmented_img]
    detected_faces_per_img = [
        person_at_night_detected_faces, friends_detected_faces,
        profile_detected_faces
    ]
    title = "Edge detection with the Canny algorithm | Face detection with KMeans"
    display_images(imgs, segmented_imgs, detected_faces_per_img, title)


if __name__ == '__main__':
    main()
