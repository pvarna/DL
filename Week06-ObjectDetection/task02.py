import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils import data
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torchvision.transforms import functional
from torchmetrics.classification import Accuracy, Precision, Recall, F1Score
from collections import Counter
import random
from tqdm import tqdm
from torch import optim
from torchvision.models import inception_v3, vgg11, resnet18, Inception_V3_Weights, VGG11_Weights, ResNet18_Weights
from torchvision.models.inception import InceptionOutputs

LEARNING_RATE = 0.001
VAL_SPLIT = 0.5
TRAIN_DIR = "../DATA/brain_tumor_dataset/Training"
TEST_DIR = "../DATA/brain_tumor_dataset/Testing"


# works with images 224x224
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
        ])

        self.classifier = nn.ModuleList([
            nn.Flatten(),
            # 224 -> 112 -> 56 -> 28
            nn.Linear(64 * 28 * 28, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        ])

    def forward(self, input_image):
        for layer in self.feature_extractor:
            input_image = layer(input_image)

        for layer in self.classifier:
            input_image = layer(input_image)

        return input_image


def load_val_test_datasets(data_dir, transforms):
    full_dataset = ImageFolder(data_dir, transforms)

    val_len = int(len(full_dataset) * VAL_SPLIT)
    test_len = len(full_dataset) - val_len

    # https://pytorch.org/docs/stable/data.html (random_split)
    generator = torch.Generator()
    val_dataset, test_dataset = data.random_split(full_dataset,
                                                  [val_len, test_len],
                                                  generator)

    return val_dataset, test_dataset


# https://stackoverflow.com/questions/62319228/number-of-instances-per-class-in-pytorch-dataset
def count_class_distribution(dataset):
    if isinstance(dataset, data.Subset):
        class_names = dataset.dataset.classes
        targets = dataset.dataset.targets
        indices = dataset.indices
    else:  # it is a whole Dataset
        class_names = dataset.classes
        targets = dataset.targets
        indices = range(len(dataset))

    classes = [targets[i] for i in indices]
    counter = Counter(classes)
    return {
        class_names[class_idx]: count
        for class_idx, count in counter.items()
    }


def show_images_per_class(dataset, num_images):
    class_names = dataset.classes
    targets = dataset.targets

    class_to_indices = {cls_idx: [] for cls_idx in range(len(class_names))}
    for i in range(len(targets)):
        class_to_indices[targets[i]].append(i)

    fig, axes = plt.subplots(nrows=len(class_names), ncols=num_images)

    for class_idx, indices in class_to_indices.items():
        sampled_indices = random.sample(indices, num_images)
        for j, idx in enumerate(sampled_indices):
            image, label = dataset[idx]
            image = functional.to_pil_image(image)
            axes[class_idx, j].imshow(image)
            axes[class_idx, j].axis('off')
            axes[class_idx, j].set_title(class_names[label])

    plt.tight_layout()
    plt.show()


def explore_dataset(train_dataset, val_dataset, test_dataset):
    show_images_per_class(train_dataset, 5)

    train_dist = count_class_distribution(train_dataset)
    val_dist = count_class_distribution(val_dataset)
    test_dist = count_class_distribution(test_dataset)

    print("Train:", train_dist)
    print("Validation:", val_dist)
    print("Test:", test_dist)


def train_epoch(train_dataloader, net, optimizer, criterion, epoch_i):
    net.train()
    running_loss = 0.0

    for inputs, targets in tqdm(train_dataloader, desc=f"Epoch {epoch_i+1}"):
        optimizer.zero_grad()

        outputs = net(inputs)
        # Handle Inception V3 (https://discuss.pytorch.org/t/inception-v3-pre-trained-model/64608/4)
        if isinstance(outputs, InceptionOutputs):
            outputs = outputs.logits

        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)

    average_loss_per_batch = running_loss / len(train_dataloader.dataset)

    return average_loss_per_batch


def train_model(train_dataloader, val_dataloader, net, optimizer, criterion,
                num_epochs):
    train_losses = []
    val_losses = []

    for i in range(num_epochs):
        train_loss = train_epoch(train_dataloader, net, optimizer, criterion,
                                 i)
        train_losses.append(train_loss)

        val_loss = evaluate_model(val_dataloader,
                                  net,
                                  criterion,
                                  show_metrics=False)
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


def evaluate_model(dataloader, net, criterion, show_metrics=True):
    net.eval()
    running_loss = 0.0

    accuracy = Accuracy(task='multiclass', num_classes=4, average='micro')
    precision = Precision(task='multiclass', num_classes=4, average='micro')
    recall = Recall(task='multiclass', num_classes=4, average='micro')
    f1 = F1Score(task='multiclass', num_classes=4, average='micro')

    with torch.no_grad():
        for inputs, targets in tqdm(dataloader, "Evaluating"):
            outputs = net(inputs)
            # Handle Inception V3 (https://discuss.pytorch.org/t/inception-v3-pre-trained-model/64608/4)
            if isinstance(outputs, InceptionOutputs):
                outputs = outputs.logits

            accuracy.update(outputs, targets)
            precision.update(outputs, targets)
            recall.update(outputs, targets)
            f1.update(outputs, targets)

            loss = criterion(outputs, targets)
            running_loss += loss.item() * inputs.size(0)

    average_loss = running_loss / len(dataloader.dataset)

    accuracy_value = accuracy.compute()
    precision_value = precision.compute()
    recall_value = recall.compute()
    f1_value = f1.compute()

    if show_metrics:
        print(f"Average evaluation loss: {average_loss}")
        print(f"Accuracy: {accuracy_value}")
        print(f"Precision: {precision_value}")
        print(f"Recall: {recall_value}")
        print(f"F1 Score: {f1_value}")

    accuracy.reset()
    precision.reset()
    recall.reset()
    f1.reset()

    return average_loss


def fine_tune(model, model_name, num_classes=4):
    for param in model.parameters():
        param.requires_grad = False

    if model_name.lower() == "inception v3":
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        for param in model.fc.parameters():
            param.requires_grad = True

    elif model_name.lower() == "vgg11":
        model.classifier[6] = nn.Linear(model.classifier[6].in_features,
                                        num_classes)
        for layer in [model.classifier[5], model.classifier[6]]:
            for param in layer.parameters():
                param.requires_grad = True

    elif model_name.lower() == "resnet18":
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        for param in model.fc.parameters():
            param.requires_grad = True

    else:
        raise ValueError(f"Model {model_name} is not supported.")


def predict_random_images(test_dataset_224, test_dataset_299, models_dict,
                          class_names):
    indices = random.sample(range(len(test_dataset_224)), 5)

    fig, axes = plt.subplots(nrows=1, ncols=5, figsize=(20, 6))

    for i, idx in enumerate(indices):
        image_224, label = test_dataset_224[idx]
        image_299, _ = test_dataset_299[idx]

        predictions = {}

        for model_name, model in models_dict.items():
            model.eval()

            input_tensor = image_299.unsqueeze(
                0) if model_name == "Inception V3" else image_224.unsqueeze(0)

            with torch.no_grad():
                output = model(input_tensor)
                if isinstance(output, InceptionOutputs):
                    output = output.logits
                predicted_class = output.argmax(dim=1).item()
                predictions[model_name] = predicted_class

        image_to_show = functional.to_pil_image(image_224)
        axes[i].imshow(image_to_show)
        axes[i].axis('off')

        title = f"True: {class_names[label]}\n"
        for model_name, pred_idx in predictions.items():
            title += f"{model_name}: {class_names[pred_idx]}\n"

        axes[i].set_title(title, fontsize=10)

    plt.tight_layout()
    plt.show()


def main():
    transforms_224 = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Resize((224, 224))])
    transforms_299 = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Resize((299, 299))])
    # train_transforms_with_augmentation = transforms.Compose([
    #     transforms.ToTensor(),
    #     transforms.Resize((299, 299)),
    #     transforms.RandomHorizontalFlip(p=0.5),
    #     transforms.RandomRotation(degrees=15),
    #     transforms.ColorJitter(brightness=0.2)
    # ])

    train_dataset_224 = ImageFolder(TRAIN_DIR, transforms_224)
    val_dataset_224, test_dataset_224 = load_val_test_datasets(
        TEST_DIR, transforms_224)

    train_dataloader_224 = data.DataLoader(train_dataset_224,
                                           shuffle=True,
                                           batch_size=32)
    val_dataloader_224 = data.DataLoader(val_dataset_224,
                                         shuffle=False,
                                         batch_size=32)
    test_dataloader_224 = data.DataLoader(test_dataset_224,
                                          shuffle=False,
                                          batch_size=32)

    train_dataset_299 = ImageFolder(TRAIN_DIR, transforms_299)
    val_dataset_299, test_dataset_299 = load_val_test_datasets(
        TEST_DIR, transforms_299)

    train_dataloader_299 = data.DataLoader(train_dataset_299,
                                           shuffle=True,
                                           batch_size=32)
    val_dataloader_299 = data.DataLoader(val_dataset_299,
                                         shuffle=False,
                                         batch_size=32)
    test_dataloader_299 = data.DataLoader(test_dataset_299,
                                          shuffle=False,
                                          batch_size=32)

    # explore_dataset(train_dataset_224, val_dataset_224, test_dataset_224)

    criterion = nn.CrossEntropyLoss()

    my_net = MultiClassCNN(num_classes=4)

    inception_v3_weights = Inception_V3_Weights.DEFAULT
    inception_v3_net = inception_v3(weights=inception_v3_weights)
    fine_tune(inception_v3_net, "Inception V3")

    vgg11_weights = VGG11_Weights.DEFAULT
    vgg11_net = vgg11(weights=vgg11_weights)
    fine_tune(vgg11_net, "VGG11")

    resnet18_weights = ResNet18_Weights.DEFAULT
    resnet18_net = resnet18(weights=resnet18_weights)
    fine_tune(resnet18_net, "ResNet18")

    nets = {
        "My Multiclass CNN": my_net,
        "Inception V3": inception_v3_net,
        "VGG11": vgg11_net,
        "ResNet18": resnet18_net
    }

    # for name, net in nets.items():
    #     train_dataloader = train_dataloader_299 if name == "Inception V3" else train_dataloader_224
    #     val_dataloader = val_dataloader_299 if name == "Inception V3" else val_dataloader_224

    #     print(f"Training {name}...")
    #     optimizer = optim.AdamW(net.parameters(), lr=LEARNING_RATE)
    #     train_losses, val_losses = train_model(train_dataloader,
    #                 val_dataloader,
    #                 net,
    #                 optimizer,
    #                 criterion,
    #                 num_epochs=5)
    #     plot_losses(train_losses, val_losses, f"{name} Training vs Validation Loss")

    #     torch.save(net.state_dict(), f"{name.replace(" ", "_").lower()}.pt")

    new_my_net = MultiClassCNN(num_classes=4)

    new_inception_v3_net = inception_v3(weights=inception_v3_weights)
    fine_tune(new_inception_v3_net, "Inception V3")

    new_vgg11_net = vgg11(weights=vgg11_weights)
    fine_tune(new_vgg11_net, "VGG11")

    new_resnet18_net = resnet18(weights=resnet18_weights)
    fine_tune(new_resnet18_net, "ResNet18")

    new_my_net.load_state_dict(torch.load("my_multiclass_cnn.pt"))
    new_my_net.eval()
    new_inception_v3_net.load_state_dict(torch.load("inception_v3.pt"))
    new_inception_v3_net.eval()
    new_vgg11_net.load_state_dict(torch.load("vgg11.pt"))
    new_vgg11_net.eval()
    new_resnet18_net.load_state_dict(torch.load("resnet18.pt"))
    new_resnet18_net.eval()

    new_nets = nets = {
        "My Multiclass CNN": new_my_net,
        "Inception V3": new_inception_v3_net,
        "VGG11": new_vgg11_net,
        "ResNet18": new_resnet18_net
    }

    # for name, net in nets.items():
    #     test_dataloader = test_dataloader_299 if name == "Inception V3" else test_dataloader_224
    #     print(f"Evaluating {name}...")
    #     evaluate_model(test_dataloader, net, criterion)

    print("\nPredicting 5 random test images:")
    predict_random_images(test_dataset_224, test_dataset_299, new_nets,
                          train_dataset_224.classes)


if __name__ == '__main__':
    main()
