import matplotlib.pyplot as plt
from torchvision.utils import draw_bounding_boxes
from torchvision.datasets import CocoDetection
from torchvision import models, transforms
from torchvision.transforms import functional
from collections import Counter
import random
import os
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import Dataset
from torchvision.transforms import functional
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torch.utils.data import DataLoader
from ultralytics import YOLO

BASE_DATA_DIR = "../DATA/car_bee_detection"
TRAIN_DIR = os.path.join(BASE_DATA_DIR, "train")
VAL_DIR = os.path.join(BASE_DATA_DIR, "valid")
TEST_DIR = os.path.join(BASE_DATA_DIR, "test")
COCO_FILE = "_annotations.coco.json"
LEARNING_RATE = 0.0001
YOLO_YAML = "/Users/petar.kolev/Documents/FMI/DL/DATA/car_bee_detection_yolov5/dataset.yaml"


class DetectionCropsDataset(Dataset):

    def __init__(self, img_folder, ann_file, transform=None):
        self.coco_dataset = CocoDetection(img_folder,
                                          ann_file,
                                          transform=transform)
        self.transform = transform

        self.car_class_id = self.find_class_id('car')
        self.bee_class_id = self.find_class_id('bee')
        self.hornet_class_id = self.find_class_id('hornet')

        self.crops = []
        self.labels = []
        self.boxes = []

        self.prepare_crops()

    def find_class_id(self, class_name):
        category_ids = self.coco_dataset.coco.getCatIds()
        categories = self.coco_dataset.coco.loadCats(category_ids)
        for cat in categories:
            if cat['name'] == class_name:
                return cat['id']

    def prepare_crops(self):
        for idx in range(len(self.coco_dataset)):
            img, target = self.coco_dataset[idx]

            for obj in target:
                bbox = obj['bbox']
                x_min, y_min, width, height = bbox
                cropped_img = functional.crop(img, int(y_min), int(x_min),
                                              int(height), int(width))
                cropped_img = functional.resize(cropped_img, (224, 224))

                if obj['category_id'] == self.car_class_id:
                    label = 1 
                elif obj['category_id'] in [
                        self.bee_class_id, self.hornet_class_id
                ]:
                    label = 0 
                else:
                    assert (False)

                self.crops.append(cropped_img)
                self.labels.append(label)
                self.boxes.append(torch.tensor([0.0, 0.0, 1.0, 1.0]))

    def __len__(self):
        return len(self.crops)

    def __getitem__(self, idx):
        img = self.crops[idx]
        label = torch.tensor(self.labels[idx])
        box = self.boxes[idx]

        return img, (label, box)


class DetectionDataset(Dataset):

    def __init__(self, img_folder, ann_file, transforms=None):
        self.coco = CocoDetection(img_folder, ann_file)
        self.transforms = transforms

        self.car_class_id = self.find_car_class_id()

    def find_car_class_id(self):
        category_ids = self.coco.coco.getCatIds()
        categories = self.coco.coco.loadCats(category_ids)
        for cat in categories:
            if cat['name'] == 'car':
                return cat['id']

    def __getitem__(self, idx):
        img, target = self.coco[idx]

        boxes = []
        labels = []

        for obj in target:
            x_min, y_min, width, height = obj['bbox']
            x_max = x_min + width
            y_max = y_min + height

            boxes.append([x_min, y_min, x_max, y_max])

            if obj['category_id'] == self.car_class_id:
                labels.append(1)  
            else:
                labels.append(0)  

        boxes = torch.tensor(boxes, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.int64)

        target = {'boxes': boxes, 'labels': labels}

        if self.transforms:
            img = self.transforms(img)

        return img, target

    def __len__(self):
        return len(self.coco)


def count_class_distribution(dataset):
    counter = Counter(dataset.labels)
    return {
        "background (bee/hornet)": counter.get(0, 0),
        "car": counter.get(1, 0)
    }


def show_images_per_class(dataset, num_images_per_class):
    class_names = {0: "background", 1: "car"}
    fig, axes = plt.subplots(nrows=2,
                             ncols=num_images_per_class,
                             figsize=(num_images_per_class * 3, 6))
    axes = axes.reshape(2, num_images_per_class)

    for label in [0, 1]:  
        indices = [i for i, l in enumerate(dataset.labels) if l == label]
        sampled = random.sample(indices, min(num_images_per_class,
                                             len(indices)))

        for j, idx in enumerate(sampled):
            image, _ = dataset[idx]
            axes[label, j].imshow(functional.to_pil_image(image))
            axes[label, j].axis('off')
            axes[label, j].set_title(class_names[label])

        for j in range(len(sampled), num_images_per_class):
            axes[label, j].axis('off')

    plt.tight_layout()
    plt.show()


def explore_dataset(train_dataset, val_dataset, test_dataset):
    print("--- Train Dataset ---")
    show_images_per_class(train_dataset, 5)
    print("Train Distribution:", count_class_distribution(train_dataset))

    print("\n--- Validation Dataset ---")
    show_images_per_class(val_dataset, 5)
    print("Validation Distribution:", count_class_distribution(val_dataset))

    print("\n--- Test Dataset ---")
    show_images_per_class(test_dataset, 5)
    print("Test Distribution:", count_class_distribution(test_dataset))


class SimpleRCNN(nn.Module):

    def __init__(self, num_classes):
        super(SimpleRCNN, self).__init__()

        vgg = models.vgg16(weights=models.VGG16_Weights.DEFAULT)
        self.backbone = nn.Sequential(*(list(vgg.features.children())))

        self.channels, self.height, self.width = 512, 7, 7
        input_dimension = self.channels * self.height * self.width

        self.classifier = nn.Sequential(nn.Linear(input_dimension, 512),
                                        nn.ReLU(), nn.Linear(512, num_classes))

        self.bbox_regressor = nn.Sequential(
            nn.Linear(input_dimension, 32),
            nn.ReLU(),
            nn.Linear(32, 4)  
        )

    def forward(self, images):
        features = self.backbone(images)
        features = torch.flatten(features, start_dim=1)

        class_logits = self.classifier(features)
        bbox_preds = self.bbox_regressor(features)

        return class_logits, bbox_preds


def train_epoch(train_dataloader, net, optimizer, cls_criterion,
                bbox_criterion, epoch_i):
    net.train()
    running_loss = 0.0

    for imgs, (labels, boxes) in tqdm(train_dataloader,
                                      desc=f"Epoch {epoch_i+1}"):
        optimizer.zero_grad()

        pred_labels, pred_boxes = net(imgs)

        cls_loss = cls_criterion(pred_labels, labels)
        bbox_loss = bbox_criterion(pred_boxes, boxes)

        loss = cls_loss + bbox_loss

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * imgs.size(0)

    average_loss_per_batch = running_loss / len(train_dataloader.dataset)

    return average_loss_per_batch


def evaluate_model(dataloader, net, cls_criterion, bbox_criterion):
    net.eval()
    running_loss = 0.0

    with torch.no_grad():
        for imgs, (labels, boxes) in tqdm(dataloader, desc="Evaluating"):
            pred_labels, pred_boxes = net(imgs)

            cls_loss = cls_criterion(pred_labels, labels)
            bbox_loss = bbox_criterion(pred_boxes, boxes)

            loss = cls_loss + bbox_loss
            running_loss += loss.item() * imgs.size(0)

    average_loss = running_loss / len(dataloader.dataset)
    print(f"Average evaluation loss: {average_loss:.4f}")

    return average_loss


def train_model(train_dataloader, val_dataloader, net, optimizer,
                cls_criterion, bbox_criterion, num_epochs):
    train_losses = []
    val_losses = []

    for i in range(num_epochs):
        train_loss = train_epoch(train_dataloader, net, optimizer,
                                 cls_criterion, bbox_criterion, i)
        train_losses.append(train_loss)

        val_loss = evaluate_model(val_dataloader, net, cls_criterion,
                                  bbox_criterion)
        val_losses.append(val_loss)

    return train_losses, val_losses


def train_epoch_faster_rcnn(train_loader, model, optimizer, epoch_i):
    model.train()
    running_loss = 0.0

    for images, targets in tqdm(train_loader, desc=f"Epoch {epoch_i+1}"):
        optimizer.zero_grad()

        loss_dict = model(images, targets)
        total_loss = sum(loss for loss in loss_dict.values())

        total_loss.backward()
        optimizer.step()

        running_loss += total_loss.item()

    average_loss = running_loss / len(train_loader)
    return average_loss


def evaluate_model_faster_rcnn(val_loader, model):
    model.eval()
    running_loss = 0.0

    with torch.no_grad():
        for images, targets in tqdm(val_loader, desc="Evaluating"):
            model.train()
            loss_dict = model(images, targets)
            model.eval()
            total_loss = sum(loss for loss in loss_dict.values())
            running_loss += total_loss.item()

    average_loss = running_loss / len(val_loader)
    print(f"Validation Loss: {average_loss:.4f}")
    return average_loss


def train_model_faster_rcnn(train_loader, val_loader, model, optimizer,
                            num_epochs):
    train_losses = []
    val_losses = []

    for epoch in range(num_epochs):
        train_loss = train_epoch_faster_rcnn(train_loader, model, optimizer,
                                             epoch)
        train_losses.append(train_loss)

        val_loss = evaluate_model_faster_rcnn(val_loader, model)
        val_losses.append(val_loss)

    return train_losses, val_losses


def plot_losses(train_losses, val_losses, title):
    plt.figure(figsize=(8, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title(title)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def train_model_yolov5(dataset_yaml_path,
                       model_name='yolov5s.pt',
                       epochs=3,
                       imgsz=640,
                       batch_size=8):
    model = YOLO(model_name)

    results = model.train(data=dataset_yaml_path,
                          epochs=epochs,
                          imgsz=imgsz,
                          batch=batch_size,
                          project='runs',
                          name='car_bee_yolov5_training',
                          exist_ok=True)

    return model, results


def evaluate_model_yolov5(model, dataset_yaml_path):
    metrics = model.val(data=dataset_yaml_path)
    print(metrics)


def load_test_images(test_folder, num_images=5):
    image_paths = [
        os.path.join(test_folder, f) for f in os.listdir(test_folder)
        if f.endswith(('.jpg', '.png', '.jpeg'))
    ]
    sampled_paths = random.sample(image_paths, num_images)
    images = [functional.to_tensor(plt.imread(path)) for path in sampled_paths]
    return sampled_paths, images


def predict_simple_rcnn(model, image):
    model.eval()
    with torch.no_grad():
        resized_img = functional.resize(image, (224, 224))
        resized_img = resized_img.unsqueeze(0)
        class_logits, bbox_preds = model(resized_img)
        probs = torch.softmax(class_logits, dim=1)
        label = torch.argmax(probs, dim=1).item()
        return label, bbox_preds.squeeze().tolist()


def predict_faster_rcnn(model, image):
    model.eval()
    with torch.no_grad():
        image = image.unsqueeze(0)
        outputs = model(image)
        boxes = outputs[0]['boxes']
        labels = outputs[0]['labels']
        scores = outputs[0]['scores']
        return labels[0].item(), boxes[0].tolist(), scores[0].item()


def predict_yolov5(model, image_path):
    results = model.predict(image_path, imgsz=640, conf=0.25, verbose=False)
    pred = results[0]
    if pred.boxes.shape[0] > 0:
        box = pred.boxes.xyxy[0].cpu().numpy()
        label = int(pred.boxes.cls[0].cpu().numpy())
        score = float(pred.boxes.conf[0].cpu().numpy())
        return label, box, score
    else:
        return None, None, None


def plot_predictions(image_paths, simple_rcnn, faster_rcnn, yolo_model):
    fig, axes = plt.subplots(len(image_paths),
                             4,
                             figsize=(20, 5 * len(image_paths)))

    for idx, image_path in enumerate(image_paths):
        img = functional.to_tensor(plt.imread(image_path))

        label1, box1 = predict_simple_rcnn(simple_rcnn, img)
        label2, box2, score2 = predict_faster_rcnn(faster_rcnn, img)
        label3, box3, score3 = predict_yolov5(yolo_model, image_path)

        axes[idx, 0].imshow(img.permute(1, 2, 0))
        axes[idx, 0].set_title(f"Original")
        axes[idx, 0].axis('off')

        axes[idx, 1].imshow(img.permute(1, 2, 0))
        axes[idx, 1].set_title(f"Simple R-CNN: {label1}")
        draw_box(axes[idx, 1], box1, img)
        axes[idx, 1].axis('off')

        axes[idx, 2].imshow(img.permute(1, 2, 0))
        axes[idx, 2].set_title(f"Faster R-CNN: {label2} (score={score2:.2f})")
        draw_box(axes[idx, 2], box2, img)
        axes[idx, 2].axis('off')

        axes[idx, 3].imshow(img.permute(1, 2, 0))
        if box3 is not None:
            axes[idx, 3].set_title(f"YOLOv5: {label3} (score={score3:.2f})")
            draw_box(axes[idx, 3], box3, img)
        else:
            axes[idx, 3].set_title(f"YOLOv5: No Detection")
        axes[idx, 3].axis('off')

    plt.tight_layout()
    plt.show()


def draw_box(ax, box, image_tensor):
    if box is None:
        return
    if isinstance(box, list) or isinstance(box, torch.Tensor):
        x1, y1, x2, y2 = box
    else:
        x1, y1, x2, y2 = box

    img_with_box = draw_bounding_boxes(
        (image_tensor * 255).to(torch.uint8),
        boxes=torch.tensor([[x1, y1, x2, y2]]),
        colors="red",
        width=2
    )

    ax.imshow(img_with_box.permute(1, 2, 0))



def main():
    train_transforms = transforms.Compose([
        transforms.ColorJitter(brightness=0.2,
                               contrast=0.2,
                               saturation=0.2,
                               hue=0.1),
        transforms.ToTensor()
    ])

    test_transforms = transforms.Compose([transforms.ToTensor()])

    # simple_rcnn_train_dataset = DetectionCropsDataset(
    #     TRAIN_DIR,
    #     os.path.join(TRAIN_DIR, COCO_FILE),
    #     transform=train_transforms)
    # simple_rcnn_val_dataset = DetectionCropsDataset(VAL_DIR,
    #                                                 os.path.join(
    #                                                     VAL_DIR, COCO_FILE),
    #                                                 transform=test_transforms)
    # simple_rcnn_test_dataset = DetectionCropsDataset(TEST_DIR,
    #                                                  os.path.join(
    #                                                      TEST_DIR, COCO_FILE),
    #                                                  transform=test_transforms)

    # simple_rcnn_train_loader = DataLoader(simple_rcnn_train_dataset,
    #                                       batch_size=32,
    #                                       shuffle=True)
    # simple_rcnn_val_loader = DataLoader(simple_rcnn_val_dataset,
    #                                     batch_size=32,
    #                                     shuffle=False)

    # explore_dataset(simple_rcnn_train_dataset, simple_rcnn_val_dataset,
    #                 simple_rcnn_test_dataset)

    # model = SimpleRCNN(num_classes=2)
    # cls_criterion = nn.CrossEntropyLoss()
    # bbox_criterion = nn.MSELoss()

    # train_losses, val_losses = train_model(simple_rcnn_train_loader,
    #                                        simple_rcnn_val_loader,
    #                                        model,
    #                                        optimizer,
    #                                        cls_criterion,
    #                                        bbox_criterion,
    #                                        num_epochs=3)
    # plot_losses(train_losses, val_losses, "Simple R-CNN Losses")
    # torch.save(model.state_dict(), "simple_rcnn_car_detection.pt")

    # model = fasterrcnn_resnet50_fpn(weights='DEFAULT')
    # in_features = model.roi_heads.box_predictor.cls_score.in_features
    # model.roi_heads.box_predictor = FastRCNNPredictor(in_features,
    #                                                   num_classes=2)

    # optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    # faster_rcnn_train_dataset = DetectionDataset(TRAIN_DIR,
    #                                              os.path.join(
    #                                                  TRAIN_DIR, COCO_FILE),
    #                                              transforms=train_transforms)
    # faster_rcnn_val_dataset = DetectionDataset(VAL_DIR,
    #                                            os.path.join(
    #                                                VAL_DIR, COCO_FILE),
    #                                            transforms=train_transforms)
    # faster_rcnn_test_dataset = DetectionDataset(TEST_DIR,
    #                                             os.path.join(
    #                                                 TEST_DIR, COCO_FILE),
    #                                             transforms=train_transforms)

    # def collate_fn(batch):
    #     return tuple(zip(*batch))

    # faster_rcnn_train_loader = DataLoader(faster_rcnn_train_dataset,
    #                                       batch_size=4,
    #                                       shuffle=True,
    #                                       collate_fn=collate_fn)
    # faster_rcnn_val_loader = DataLoader(faster_rcnn_val_dataset,
    #                                     batch_size=4,
    #                                     shuffle=True,
    #                                     collate_fn=collate_fn)

    # train_losses, val_losses = train_model_faster_rcnn(
    #     faster_rcnn_train_loader,
    #     faster_rcnn_val_loader,
    #     model,
    #     optimizer,
    #     num_epochs=3)
    # plot_losses(train_losses, val_losses, "Faster R-CNN Losses")
    # torch.save(model.state_dict(), "faster_rcnn_car_detection.pt")

    # yolo_model, yolo_results = train_model_yolov5(
    #     dataset_yaml_path=YOLO_YAML,
    #     model_name='yolov5s.pt',
    #     epochs=3,
    #     imgsz=640,
    #     batch_size=8
    # )
    # print(f"{yolo_results=}")

    # evaluate_model_yolov5(yolo_model, YOLO_YAML)
    # yolo_model.save("my_yolo_model.pt")

    simple_rcnn = SimpleRCNN(num_classes=2)
    simple_rcnn.load_state_dict(torch.load("simple_rcnn_car_detection.pt"))
    simple_rcnn.eval()

    faster_rcnn = fasterrcnn_resnet50_fpn(weights='DEFAULT')
    in_features = faster_rcnn.roi_heads.box_predictor.cls_score.in_features
    faster_rcnn.roi_heads.box_predictor = FastRCNNPredictor(in_features,
                                                            num_classes=2)
    faster_rcnn.load_state_dict(torch.load("faster_rcnn_car_detection.pt"))
    faster_rcnn.eval()

    yolo_model = YOLO("my_yolo_model.pt")


    image_paths, images = load_test_images(TEST_DIR, num_images=5)
    plot_predictions(image_paths, simple_rcnn, faster_rcnn, yolo_model)


if __name__ == '__main__':
    main()
