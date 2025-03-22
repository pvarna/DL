import pathlib
import random
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

def get_all_cloud_images():
    # https://realpython.com/get-all-files-in-directory-python/
    clouds = pathlib.Path("../DATA/clouds")

    return list(clouds.rglob("*.jpg"))

def main():
    clouds = get_all_cloud_images()

    # https://stackoverflow.com/questions/46615554/how-to-display-multiple-images-in-one-figure
    selected_clouds = random.sample(clouds, 6)
    nrows, ncols = 2, 3
    figsize = [10, 8]

    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
    for i, axi in enumerate(ax.flat):
        img_path = selected_clouds[i]
        img = Image.open(img_path)

        label = img_path.parent.name

        axi.imshow(img)
        axi.set_title(label)
        axi.axis('off')

    fig.suptitle("The Clouds dataset")

    plt.tight_layout()
    plt.show()





if __name__ == '__main__':
    main()
