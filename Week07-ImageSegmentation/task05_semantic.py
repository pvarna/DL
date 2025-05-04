import os
import json
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision.transforms import functional
from torchvision.models.segmentation import fcn_resnet50, FCN_ResNet50_Weights
from torchvision.models.segmentation.fcn import FCNHead
import torch.optim as optim
import torchmetrics

SIZE = (256, 256)
IMAGE_DIR = "../DATA/relabelled_coco/images"
MASK_DIR = "../DATA/relabelled_coco/masks"
JSON_PATH = "../DATA/relabelled_coco/relabeled_coco_val.json"
BEST_MODEL_PATH = "best_fcn_resnet50.pt"
LEARNING_RATE = 0.0005
BATCH_SIZE = 32
BACKGROUND_CAT_ID = 0


class SemanticSegmentationDataset(Dataset):

    def __init__(self, image_dir, mask_dir, json_path):
        super().__init__()
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.size = SIZE

        with open(json_path, "r") as file:
            self.json_data = json.load(file)

        self.images = self.json_data["images"]
        self.categories = self.json_data["categories"]

        all_cat_ids = set(cat['id'] for cat in self.categories)
        self.num_classes = max(all_cat_ids) + 1

        self.annotations_map = {}
        for ann in self.json_data["annotations"]:
            self.annotations_map[ann["image_id"]] = ann

        self.category_id_to_name = {
            cat["id"]: cat["name"]
            for cat in self.categories
        }
        self.category_id_to_isthing = {
            cat['id']: cat['isthing']
            for cat in self.categories
        }

        print(f"Dataset initialized with {len(self.images)} images.")
        print(f"Number of categories found: {self.num_classes}")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_info = self.images[idx]
        image_id = image_info["id"]
        image_filename = image_info["file_name"]  # 000000000139.jpg

        image_path = os.path.join(self.image_dir, image_filename)
        image = Image.open(image_path).convert("RGB").resize(self.size)
        image_tensor = functional.to_tensor(image)

        mask_filename = image_filename.replace(".jpg", ".png")
        mask_path = os.path.join(self.mask_dir, mask_filename)
        segment_mask_pil = Image.open(mask_path).convert("L").resize(
            SIZE, resample=Image.Resampling.NEAREST)
        segment_mask_np = np.array(segment_mask_pil, dtype=np.int32)

        annotation_info = self.annotations_map[image_id]
        segments_info = annotation_info["segments_info"]

        seg_id_to_cat_id = {BACKGROUND_CAT_ID: BACKGROUND_CAT_ID}
        for seg_info in segments_info:
            seg_id_to_cat_id[seg_info["id"]] = seg_info["category_id"]

        semantic_mask_np = np.full(segment_mask_np.shape,
                                   BACKGROUND_CAT_ID,
                                   dtype=np.int64)

        unique_segment_ids = np.unique(segment_mask_np)

        for segment_id in unique_segment_ids:
            if segment_id == 0:
                continue

            category_id = seg_id_to_cat_id[segment_id]
            semantic_mask_np[segment_mask_np == segment_id] = category_id

        semantic_mask_tensor = torch.from_numpy(semantic_mask_np).long()

        return image_tensor, semantic_mask_tensor


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


def train_epoch(train_loader, net, optimizer, criterion, i):
    net.train()
    total_loss = 0.0

    for images, masks in tqdm(train_loader, desc=f"Training epoch {i+1}..."):
        optimizer.zero_grad()
        outputs = net(images)["out"]
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)

    average_loss = total_loss / len(train_loader.dataset)
    print(f"Average training loss during epoch {i+1}: {average_loss:.4f}")

    return average_loss


def evaluate_model(loader, net, criterion, num_classes):
    net.eval()
    total_loss = 0.0
    num_samples = 0

    iou_metric = torchmetrics.JaccardIndex(task="multiclass",
                                           num_classes=num_classes,
                                           average='macro',
                                           ignore_index=None)

    acc_metric = torchmetrics.Accuracy(task="multiclass",
                                       num_classes=num_classes,
                                       average='micro')

    with torch.no_grad():
        for images, masks in tqdm(loader, desc="Evaluating..."):
            outputs = net(images)["out"]
            loss = criterion(outputs, masks)
            total_loss += loss.item() * images.size(0)
            num_samples += images.size(0)

            preds = torch.argmax(outputs, dim=1)

            iou_metric.update(preds, masks)
            acc_metric.update(preds, masks)

    average_loss = total_loss / len(loader.dataset)
    final_miou = iou_metric.compute()
    final_accuracy = acc_metric.compute()

    final_miou_float = final_miou.item()
    final_accuracy_float = final_accuracy.item()

    iou_metric.reset()
    acc_metric.reset()

    print(f"Average evaluation loss: {average_loss:.4f}")
    print(f"Pixel Accuracy: {final_accuracy_float:.4f}")
    print(f"Mean IoU (mIoU): {final_miou_float:.4f}")

    return average_loss, final_accuracy_float, final_miou_float


def train_model(train_loader, val_loader, net, optimizer, criterion,
                num_epochs, num_classes):
    train_losses = []
    val_losses = []
    val_accs = []
    val_mIoUs = [] 
    best_mIoU = -1.0 

    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        train_loss = train_epoch(train_loader, net, optimizer, criterion,
                                 epoch)
        val_loss, val_acc, val_mIoU = evaluate_model(val_loader, net,
                                                     criterion, num_classes)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        val_mIoUs.append(val_mIoU)

        if val_mIoU > best_mIoU:
            best_mIoU = val_mIoU
            torch.save(net.state_dict(), BEST_MODEL_PATH)

    plot_losses(train_losses,
                val_losses,
                val_accs,
                val_mIoUs,
                title="Training Metrics")


def plot_losses(train_losses,
                val_losses,
                val_accs,
                val_mIoUs,
                title="Training Metrics"):
    epochs = range(1, len(train_losses) + 1)
    num_plots = 3
    plt.figure(figsize=(6 * num_plots, 5))

    plot_index = 1

    plt.subplot(1, num_plots, plot_index)
    plt.plot(epochs, train_losses, 'bo-', label='Training Loss')
    plt.plot(epochs, val_losses, 'ro-', label='Validation Loss')
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plot_index += 1

    plt.subplot(1, num_plots, plot_index)
    plt.plot(epochs, val_accs, 'yo-', label='Validation Accuracy')
    plt.title('Validation Pixel Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plot_index += 1

    plt.subplot(1, num_plots, plot_index)
    plt.plot(epochs, val_mIoUs, 'go-', label='Validation mIoU')
    plt.title('Validation Mean IoU')
    plt.xlabel('Epoch')
    plt.ylabel('mIoU')
    plt.legend()
    plt.grid(True)
    plot_index += 1

    plt.suptitle(title)
    plt.tight_layout()
    plt.show()


def main():
    dataset = SemanticSegmentationDataset(IMAGE_DIR, MASK_DIR, JSON_PATH)

    train_set, val_set, test_set = split_dataset(dataset)

    train_loader = DataLoader(train_set,
                              batch_size=BATCH_SIZE,
                              shuffle=True)
    val_loader = DataLoader(val_set,
                            batch_size=BATCH_SIZE,
                            shuffle=False)
    test_loader = DataLoader(test_set,
                             batch_size=BATCH_SIZE,
                             shuffle=False)

    weights = FCN_ResNet50_Weights.DEFAULT
    model = fcn_resnet50(weights=weights)

    model.classifier = FCNHead(2048, dataset.num_classes)
    model.aux_classifier = None

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(),
                            lr=LEARNING_RATE,
                            weight_decay=1e-4)

    train_model(train_loader, val_loader, model, optimizer, criterion, 5, dataset.num_classes)

    model.load_state_dict(torch.load(BEST_MODEL_PATH))
    test_loss, test_acc, test_mIoU = evaluate_model(test_loader, model,
                                                    criterion,
                                                    dataset.num_classes)
    
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Pixel Accuracy: {test_acc:.4f}")
    print(f"Test Mean IoU (mIoU): {test_mIoU:.4f}")


if __name__ == '__main__':
    main()
