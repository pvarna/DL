from torchvision.datasets import ImageFolder
from torchvision import transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import os
from PIL import Image
from tqdm import tqdm
from torchmetrics.classification import Accuracy, Precision, Recall, F1Score
from torch import optim

BATCH_SIZE = 32
IMG_SIZE = 224
TRAIN_DIR = os.path.join("DATA", "train")
VAL_DIR = os.path.join("DATA", "val")
TEST_DIR = os.path.join("DATA", "test")
NUM_CLASSES = 15
LEARNING_RATE = 0.001
NUM_EPOCHS = 5


class MultiClassCNN(nn.Module):

    def __init__(self, num_classes):
        super(MultiClassCNN, self).__init__()

        self.feature_extractor = nn.ModuleList([
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        ])

        self.classifier = nn.ModuleList([
            nn.Flatten(),
            # 224 -> 112 -> 56 -> 28 -> 14
            nn.Linear(128 * 14 * 14, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        ])

    def forward(self, input_image):
        for layer in self.feature_extractor:
            input_image = layer(input_image)

        for layer in self.classifier:
            input_image = layer(input_image)

        return input_image


def train_epoch(loader, net, optimizer, criterion, epoch_i):
    net.train()
    run_loss, run_correct, run_total = 0.0, 0, 0

    for x, y in tqdm(loader, desc=f"Epoch {epoch_i+1}"):
        optimizer.zero_grad()
        out = net(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()

        run_loss += loss.item() * x.size(0)
        preds = out.argmax(1)
        run_correct += (preds == y).sum().item()
        run_total += y.size(0)

    avg_loss = run_loss / run_total
    avg_acc = run_correct / run_total
    return avg_loss, avg_acc


def evaluate_model(loader, net, criterion):
    net.eval()
    run_loss, run_correct, run_total = 0.0, 0, 0
    f1_metric = F1Score(task='multiclass',
                        num_classes=NUM_CLASSES,
                        average='macro')

    per_class_f1_metric = F1Score(task="multiclass",
                                  num_classes=NUM_CLASSES,
                                  average="none")

    with torch.no_grad():
        for x, y in tqdm(loader, "Evaluating", leave=False):
            out = net(x)
            loss = criterion(out, y)
            run_loss += loss.item() * x.size(0)

            preds = out.argmax(1)
            run_correct += (preds == y).sum().item()
            run_total += y.size(0)

            f1_metric.update(out, y)
            per_class_f1_metric.update(out, y)

    avg_loss = run_loss / run_total
    avg_acc = run_correct / run_total
    avg_f1 = f1_metric.compute()

    per_class_f1 = per_class_f1_metric.compute().numpy()
    class_names = loader.dataset.classes
    per_class_dict = {
        cls: float(score)
        for cls, score in zip(class_names, per_class_f1)
    }

    f1_metric.reset()
    per_class_f1_metric.reset()

    return avg_loss, avg_acc, avg_f1, per_class_dict


def train_model(train_loader, val_loader, net, optim_, crit, epochs):
    tr_loss, tr_acc = [], []
    va_loss, va_acc, va_f1 = [], [], []

    for ep in range(epochs):
        tl, ta = train_epoch(train_loader, net, optim_, crit, ep)
        vl, va, vf1, _ = evaluate_model(val_loader, net, crit)

        tr_loss.append(tl)
        tr_acc.append(ta)
        va_loss.append(vl)
        va_acc.append(va)
        va_f1.append(vf1)

        print(f"Epoch {ep+1}: "
              f"train‑loss {tl:.3f}  train‑acc {ta:.3f} | "
              f"val‑loss {vl:.3f}  val‑acc {va:.3f}  val‑F1 {vf1:.3f}")

    plot_curves(tr_loss, va_loss, tr_acc, va_acc, va_f1)
    return tr_loss, va_loss, tr_acc, va_acc, va_f1


def plot_curves(tr_loss,
                va_loss,
                tr_acc,
                va_acc,
                va_f1,
                title="Training progress"):
    epochs = range(1, len(tr_loss) + 1)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # --- Loss ---
    axes[0].plot(epochs, tr_loss, label="Train")
    axes[0].plot(epochs, va_loss, label="Val")
    axes[0].set_title("Cross‑entropy loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()
    axes[0].grid(True)

    # --- Accuracy ---
    axes[1].plot(epochs, tr_acc, label="Train")
    axes[1].plot(epochs, va_acc, label="Val")
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()
    axes[1].grid(True)

    # --- F1 (val only) ---
    axes[2].plot(epochs, va_f1, label="Val F1", color="tab:green")
    axes[2].set_ylim(0, 1)
    axes[2].set_title("Macro‑F1")
    axes[2].set_xlabel("Epoch")
    axes[2].legend()
    axes[2].grid(True)

    fig.suptitle(title)
    plt.tight_layout()
    plt.show()


def main():
    basic_transforms = transforms.Compose(
        [transforms.Resize((IMG_SIZE, IMG_SIZE)),
         transforms.ToTensor()])

    train_set = ImageFolder(TRAIN_DIR, transform=basic_transforms)
    val_set = ImageFolder(VAL_DIR, transform=basic_transforms)
    test_set = ImageFolder(TEST_DIR, transform=basic_transforms)

    train_load = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    val_load = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False)
    test_load = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False)

    criterion = nn.CrossEntropyLoss()
    my_net = MultiClassCNN(num_classes=NUM_CLASSES)

    optimizer = optim.AdamW(my_net.parameters(), lr=LEARNING_RATE)

    train_model(train_load, val_load, my_net, optimizer, criterion, NUM_EPOCHS)
    loss, acc, f1, per_class_dict = evaluate_model(test_load, my_net,
                                                   criterion)
    pretty_dict = {
        cls: f"{score:.3f}"
        for cls, score in per_class_dict.items()
    }
    print(f"Average loss: {loss:.3f}")
    print(f"Accuracy: {acc:.3f}")
    print(f"F1 score: {f1:.3f}")
    print(f"Per class F1 score: {pretty_dict}")


if __name__ == '__main__':
    main()
