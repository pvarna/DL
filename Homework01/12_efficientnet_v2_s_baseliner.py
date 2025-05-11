import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import os
from PIL import Image
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms, models
from torchmetrics.classification import Accuracy, Precision, Recall, F1Score
from torch import optim
from torch.optim import lr_scheduler


BATCH_SIZE = 32
IMG_SIZE = 224
TRAIN_DIR = os.path.join("drive", "MyDrive", "DL_HW1", "Homework01", "DATA", "train")
VAL_DIR = os.path.join("drive", "MyDrive", "DL_HW1", "Homework01", "DATA", "val")
TEST_DIR = os.path.join("drive", "MyDrive", "DL_HW1", "Homework01", "DATA", "test")
NUM_CLASSES = 15
LEARNING_RATE = 1e-3
NUM_EPOCHS = 50
PATIENCE = 10

PRETRAINED_MODEL_SAVE_PATH = "best_effnet_v2_s_finetuned.pt"

def train_epoch(loader, net, optimizer, criterion, epoch_i):
    net.train()
    run_loss, run_correct, run_total = 0.0, 0, 0

    for x, y in tqdm(loader, desc=f"Epoch {epoch_i + 1}"):
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
    f1_metric = F1Score(task="multiclass", num_classes=NUM_CLASSES, average="macro")
    per_class_f1_metric = F1Score(task="multiclass", num_classes=NUM_CLASSES, average="none")

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
    avg_f1 = f1_metric.compute().item()
    per_class_f1 = per_class_f1_metric.compute().cpu().numpy()

    class_names = loader.dataset.classes
    per_class_dict = {cls: float(score) for cls, score in zip(class_names, per_class_f1)}

    f1_metric.reset()
    per_class_f1_metric.reset()

    return avg_loss, avg_acc, avg_f1, per_class_dict


def train_model(train_loader, val_loader, net, optim_, crit, epochs, scheduler, patience, model_save_path):
    tr_loss, tr_acc = [], []
    va_loss, va_acc, va_f1 = [], [], []

    best_val_loss = float("inf")
    epochs_no_improve = 0

    for ep in range(epochs):
        tl, ta = train_epoch(train_loader, net, optim_, crit, ep)
        vl, va, vf1, _ = evaluate_model(val_loader, net, crit)

        tr_loss.append(tl)
        tr_acc.append(ta)
        va_loss.append(vl)
        va_acc.append(va)
        va_f1.append(vf1)

        print(
            f"Epoch {ep + 1}: train-loss {tl:.3f}  train-acc {ta:.3f} | val-loss {vl:.3f}  val-acc {va:.3f}  val-F1 {vf1:.3f}"
        )

        scheduler.step(vl)

        if vl < best_val_loss:
            best_val_loss = vl
            epochs_no_improve = 0
            print(f" Saving model to {model_save_path}")
            torch.save(net.state_dict(), model_save_path)
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            print(f"\nEarly stopping triggered after {patience} epochs without improvement.")
            print(f"Best validation loss achieved: {best_val_loss:.4f}")
            break

    plot_curves(tr_loss, va_loss, tr_acc, va_acc, va_f1)

    print(f"Loading best model weights from {model_save_path}")
    net.load_state_dict(torch.load(model_save_path))
    return tr_loss, va_loss, tr_acc, va_acc, va_f1


def plot_curves(tr_loss, va_loss, tr_acc, va_acc, va_f1, title="Training progress"):
    epochs = range(1, len(tr_loss) + 1)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # --- Loss ---
    axes[0].plot(epochs, tr_loss, label="Train")
    axes[0].plot(epochs, va_loss, label="Val")
    axes[0].set_title("Cross-entropy loss")
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
    axes[2].plot(epochs, va_f1, label="Val F1")
    axes[2].set_ylim(0, 1)
    axes[2].set_title("Macro-F1")
    axes[2].set_xlabel("Epoch")
    axes[2].legend()
    axes[2].grid(True)

    fig.suptitle(title)
    plt.tight_layout()
    plt.show()


def main():
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]

    train_transforms = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
    ])

    basic_transforms = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
    ])

    train_set = ImageFolder(TRAIN_DIR, transform=train_transforms)
    val_set = ImageFolder(VAL_DIR, transform=basic_transforms)
    test_set = ImageFolder(TEST_DIR, transform=basic_transforms)

    train_load = DataLoader(train_set, shuffle=True, batch_size=BATCH_SIZE)
    val_load = DataLoader(val_set, shuffle=False, batch_size=BATCH_SIZE)
    test_load = DataLoader(test_set, shuffle=False, batch_size=BATCH_SIZE)

    weights = models.EfficientNet_V2_S_Weights.IMAGENET1K_V1
    effnet_v2_s_model = models.efficientnet_v2_s(weights=weights)

    for param in effnet_v2_s_model.parameters():
        param.requires_grad = False

    num_ftrs = effnet_v2_s_model.classifier[-1].in_features
    effnet_v2_s_model.classifier[-1] = nn.Linear(num_ftrs, NUM_CLASSES)
    effnet_v2_s_model

    criterion = nn.CrossEntropyLoss()

    optimizer_ft = optim.AdamW(effnet_v2_s_model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)

    scheduler_ft = lr_scheduler.ReduceLROnPlateau(
        optimizer_ft, mode="min", factor=0.1, patience=3, verbose=True
    )

    train_model(
        train_load,
        val_load,
        effnet_v2_s_model,
        optimizer_ft,
        criterion,
        NUM_EPOCHS,
        scheduler_ft,
        PATIENCE,
        PRETRAINED_MODEL_SAVE_PATH,
    )

    loss_ft, acc_ft, f1_ft, per_class_dict_ft = evaluate_model(test_load, effnet_v2_s_model, criterion)

    pretty_dict_ft = {cls: f"{score:.3f}" for cls, score in per_class_dict_ft.items()}
    print(f"EfficientNet-V2-S Test Avg Loss: {loss_ft:.3f}")
    print(f"EfficientNet-V2-S Test Accuracy:  {acc_ft:.3f}")
    print(f"EfficientNet-V2-S Test Macro F1:  {f1_ft:.3f}")
    print(f"EfficientNet-V2-S Test Per-class F1: {pretty_dict_ft}")


if __name__ == "__main__":
    main()
