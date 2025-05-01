import os
from PIL import Image
import torch
from torchvision import transforms
from torchvision.models import detection
import matplotlib.pyplot as plt

def get_class_names(coco_labels_path):
    with open(coco_labels_path) as file:
        class_names = [line.strip() for line in file]
    return class_names

def main():
    model = detection.maskrcnn_resnet50_fpn(pretrained=True)
    model.eval()

    image = Image.open(os.path.join('..', 'DATA', 'w07_task02.jpg')).convert('RGB')
    image_tensor = transforms.ToTensor()(image).unsqueeze(0)

    with torch.no_grad():
        prediction = model(image_tensor)

    masks = prediction[0]['masks']
    labels = prediction[0]['labels']
    scores = prediction[0]['scores']

    class_names = get_class_names("coco-labels-91.txt")

    fig, axs = plt.subplots(1, 2)

    print(f"Object with highest confidence ({scores[0]:.4f}) is \"{class_names[labels[0] - 1]}\".")
    print(f"Second object with highest confidence ({scores[1]:.4f}) is \"{class_names[labels[1] - 1]}\"")

    for i in range(2):
        axs[i].axis('off')
        axs[i].imshow(image)
        axs[i].imshow(
            masks[i, 0],
            cmap='jet',
            alpha=0.5,
        )
        axs[i].set_title(f'Object: {class_names[labels[i] - 1]}') # class_names is the list with the COCO labels that you saw earlier (without enumeration)

    plt.show()


if __name__ == '__main__':
    main()
