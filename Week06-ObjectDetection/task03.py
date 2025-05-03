from PIL import Image
import torch
from torchvision import transforms, utils
import torchvision.transforms.functional as TF
import matplotlib.pyplot as plt

def main():
    img = Image.open("../DATA/w06_task03_start.jpeg")

    img = TF.resize(img, (224, 224))

    img_tensor = TF.to_tensor(img) * 255
    img_tensor = img_tensor.type(torch.uint8)

    bbox = torch.tensor([[65, 20, 180, 185]])

    boxed_img = utils.draw_bounding_boxes(img_tensor, bbox, width=3, colors='red')

    boxed_pil = TF.to_pil_image(boxed_img)
    plt.imshow(boxed_pil)
    plt.title("Espresso Shot with Bounding Box")
    plt.axis("off")
    plt.show()

if __name__ == '__main__':
    main()
