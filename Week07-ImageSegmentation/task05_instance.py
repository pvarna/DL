import torch
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image
import numpy as np
import os
import json
from torchvision.transforms import functional
from tqdm import tqdm
import matplotlib.pyplot as plt
import random

import torchvision
from torchvision.models.detection import maskrcnn_resnet50_fpn_v2, MaskRCNN_ResNet50_FPN_V2_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
import torch.optim as optim
import torchmetrics
import time

SIZE = (256, 256)
IMAGE_DIR = "../DATA/relabelled_coco/images"
MASK_DIR = "../DATA/relabelled_coco/masks"
JSON_PATH = "../DATA/relabelled_coco/relabeled_coco_val.json"
BACKGROUND_CAT_ID = 0
INSTANCE_BEST_MODEL_PATH = "best_maskrcnn_resnet50_v2.pt"
INSTANCE_LEARNING_RATE = 0.001
INSTANCE_BATCH_SIZE = 8
WEIGHT_DECAY = 0.0005
NUM_EPOCHS = 10


def get_bounding_box(mask_np):
    rows = np.any(mask_np, axis=1)
    cols = np.any(mask_np, axis=0)

    if not np.any(rows) or not np.any(cols): return None

    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]

    x_min, y_min = cmin, rmin
    x_max, y_max = cmax + 1, rmax + 1

    if x_max <= x_min or y_max <= y_min: return None
    return [x_min, y_min, x_max, y_max]


class InstanceSegmentationDataset(Dataset):

    def __init__(self, image_dir, mask_dir, json_path):
        super().__init__()
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.size = SIZE

        with open(json_path, "r") as file:
            self.json_data = json.load(file)
        self.images_info = self.json_data["images"]
        self.categories = self.json_data["categories"]

        self.annotations_map = {}
        for ann in tqdm(self.json_data["annotations"],
                        desc="Mapping annotations"):
            self.annotations_map[ann["image_id"]] = ann

        self.category_id_info = {cat["id"]: cat for cat in self.categories}
        self.thing_cat_ids = sorted([
            cat_id for cat_id, info in self.category_id_info.items()
            if info['isthing'] == 1
        ])

        self.num_thing_classes = len(self.thing_cat_ids)

        self.cat_id_to_label = {
            cat_id: i + 1
            for i, cat_id in enumerate(self.thing_cat_ids)
        }
        self.label_to_cat_id = {
            i + 1: cat_id
            for i, cat_id in enumerate(self.thing_cat_ids)
        }

        self.vis_category_id_to_name = {
            cat["id"]: cat["name"]
            for cat in self.categories
        }
        print(f"'Thing' classes identified: {self.num_thing_classes}")
        print(
            f"Instance segmentation dataset initialized with {len(self.images_info)} images."
        )

    def __len__(self):
        return len(self.images_info)

    def __getitem__(self, idx):
        image_info = self.images_info[idx]
        image_id = image_info["id"]
        image_filename = image_info["file_name"]
        image_path = os.path.join(self.image_dir, image_filename)
        image_pil = Image.open(image_path).convert("RGB").resize(self.size)

        mask_filename = image_filename.replace(".jpg", ".png")
        mask_path = os.path.join(self.mask_dir, mask_filename)

        segment_mask_pil = Image.open(mask_path).convert('L').resize(
            self.size, resample=Image.Resampling.NEAREST)
        segment_mask_np = np.array(segment_mask_pil, dtype=np.int32)

        annotation_info = self.annotations_map[image_id]
        segments_info = annotation_info["segments_info"]

        instance_masks, labels, boxes = [], [], []

        for seg_info in segments_info:
            category_id, segment_id = seg_info["category_id"], seg_info["id"]
            if category_id in self.cat_id_to_label:
                binary_mask_np = (segment_mask_np == segment_id)
                if np.any(binary_mask_np):
                    bbox = get_bounding_box(binary_mask_np)
                    if bbox is not None:
                        instance_masks.append(
                            torch.from_numpy(binary_mask_np).to(torch.uint8))
                        labels.append(self.cat_id_to_label[category_id])
                        boxes.append(bbox)

        target = {}
        num_instances = len(boxes)
        if num_instances > 0:
            target["boxes"] = torch.as_tensor(boxes, dtype=torch.float32)
            target["labels"] = torch.as_tensor(labels, dtype=torch.int64)
            target["masks"] = torch.stack(instance_masks)

            if num_instances > 0:
                target["area"] = (target["boxes"][:, 3] - target["boxes"][:, 1]
                                  ) * (target["boxes"][:, 2] -
                                       target["boxes"][:, 0])
                target["iscrowd"] = torch.zeros((num_instances, ),
                                                dtype=torch.int64)

        if num_instances == 0:
            target["boxes"] = torch.empty((0, 4), dtype=torch.float32)
            target["labels"] = torch.empty((0, ), dtype=torch.int64)
            target["masks"] = torch.empty((0, self.size[0], self.size[1]),
                                          dtype=torch.uint8)
            target["area"] = torch.empty((0, ), dtype=torch.float32)
            target["iscrowd"] = torch.empty((0, ), dtype=torch.int64)

        target["image_id"] = torch.tensor([image_id])
        image_tensor = functional.to_tensor(image_pil)

        return image_tensor, target


def split_dataset(dataset):
    total_size = len(dataset)
    train_size = int(0.8 * total_size)
    val_size = int(0.1 * total_size)
    test_size = total_size - train_size - val_size

    train_set, val_set, test_set = random_split(
        dataset, [train_size, val_size, test_size])

    print(f"Train size: {len(train_set)}")
    print(f"Validation size: {len(val_set)}")
    print(f"Test size: {len(test_set)}")

    return train_set, val_set, test_set


def collate_fn(batch):
    return tuple(zip(*batch))


def get_instance_segmentation_model(num_classes):
    weights = MaskRCNN_ResNet50_FPN_V2_Weights.DEFAULT
    model = maskrcnn_resnet50_fpn_v2(weights=weights)

    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask,
                                                       hidden_layer,
                                                       num_classes)

    return model


def train_one_epoch_instance(model, optimizer, data_loader, epoch):
    model.train()
    total_loss = 0.0
    processed_batches = 0

    for images, targets in tqdm(data_loader, desc=f"Training epoch {epoch+1}"):
        loss_dict = model(images, targets)

        losses = sum(loss for loss in loss_dict.values())
        loss_value = losses.item()
        total_loss += loss_value
        processed_batches += 1

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

    average_loss = total_loss / processed_batches if processed_batches > 0 else 0.0
    print(f"Average training loss epoch {epoch+1}: {average_loss:.4f}")

    return average_loss


def evaluate_instance_model(model, data_loader):
    model.eval()

    metric = torchmetrics.detection.MeanAveragePrecision(iou_type="segm")

    with torch.no_grad():
        for images, targets in tqdm(data_loader, desc="Evaluating"):
            targets_metric = [{k: v for k, v in t.items()} for t in targets]

            predictions = model(images)

            preds_metric = []
            for p in predictions:
                processed_masks = (p['masks'] > 0.5).squeeze(1).to(torch.uint8)

                pred_dict = {
                    'boxes': p['boxes'],
                    'scores': p['scores'],
                    'labels': p['labels'],
                    'masks': processed_masks
                }
                preds_metric.append(pred_dict)

            metric.update(preds_metric, targets_metric)

    results = metric.compute()
    metric.reset()

    map_val = results['map'].item()
    map_50_val = results['map_50'].item()
    print(f"Validation mAP @[0.5:0.95]: {map_val:.4f}")
    print(f"Validation mAP @[0.5]:      {map_50_val:.4f}")
    return map_val


def train_instance_model(model, train_loader, val_loader, optimizer,
                         num_epochs):
    best_mAP = -1.0
    history = {'train_loss': [], 'val_mAP': []}

    for epoch in range(num_epochs):
        train_loss = train_one_epoch_instance(model, optimizer, train_loader,
                                              epoch)
        history['train_loss'].append(train_loss)

        val_mAP = evaluate_instance_model(model, val_loader)
        history['val_mAP'].append(val_mAP)

        print(f"Epoch {epoch+1}/{num_epochs}, Val mAP: {val_mAP:.4f}")

        if val_mAP > best_mAP:
            best_mAP = val_mAP
            torch.save(model.state_dict(), INSTANCE_BEST_MODEL_PATH)

    plot_training_history(history)
    return history


def plot_training_history(history):
    epochs = range(1, len(history['train_loss']) + 1)
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history['train_loss'], label='Train Loss')
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(epochs, history['val_mAP'], label='Validation mAP')
    plt.title('Validation mAP')
    plt.xlabel('Epoch')
    plt.ylabel('mAP')
    plt.grid(True)

    plt.tight_layout()
    plt.show()


def main():
    instance_dataset = InstanceSegmentationDataset(IMAGE_DIR, MASK_DIR,
                                                   JSON_PATH)
    num_classes_model = instance_dataset.num_thing_classes + 1

    train_set, val_set, test_set = split_dataset(instance_dataset)

    train_loader = DataLoader(train_set,
                              batch_size=INSTANCE_BATCH_SIZE,
                              shuffle=True,
                              collate_fn=collate_fn)
    val_loader = DataLoader(val_set,
                            batch_size=INSTANCE_BATCH_SIZE,
                            shuffle=False,
                            collate_fn=collate_fn)
    test_loader = DataLoader(test_set,
                             batch_size=INSTANCE_BATCH_SIZE,
                             shuffle=False,
                             collate_fn=collate_fn)

    model = get_instance_segmentation_model(num_classes_model)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(params,
                            lr=INSTANCE_LEARNING_RATE,
                            weight_decay=WEIGHT_DECAY)

    train_instance_model(model, train_loader, val_loader, optimizer,
                         NUM_EPOCHS)

    torch.save(model.state_dict(), INSTANCE_BEST_MODEL_PATH)

    model.load_state_dict(torch.load(INSTANCE_BEST_MODEL_PATH))

    test_mAP = evaluate_instance_model(model, test_loader)
    print(f"Test mAP @[0.5:0.95]: {test_mAP:.4f}")


if __name__ == '__main__':
    main()
