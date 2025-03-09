import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import TensorDataset, DataLoader
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import torchmetrics

TRAINING_DATA_PATH = "../Data/water_train.csv"
TESTING_DATA_PATH = "../Data/water_test.csv"
FEATURE_LABELS = [
    "ph", "Hardness", "Solids", "Chloramines", "Sulfate", "Conductivity",
    "Organic_carbon", "Trihalomethanes", "Turbidity"
]
TARGET_LABELS = ["Potability"]
BATCH_SIZE = 64
LEARNING_RATE = 0.001
EPOCHS = 30
RANDOM_SEED = 42


def load_data():
    df_training = pd.read_csv(TRAINING_DATA_PATH)
    df_testing = pd.read_csv(TESTING_DATA_PATH)

    training_set_features = df_training[FEATURE_LABELS].values
    training_set_targets = df_training[TARGET_LABELS].values

    testing_set_features = df_testing[FEATURE_LABELS].values
    testing_set_targets = df_testing[TARGET_LABELS].values

    validation_set_features, testing_set_features, validation_set_targets, testing_set_targets = train_test_split(
        testing_set_features,
        testing_set_targets,
        test_size=0.5,
        random_state=RANDOM_SEED,
        stratify=testing_set_targets)

    dataset_training = TensorDataset(
        torch.tensor(training_set_features).float(),
        torch.tensor(training_set_targets).float())
    dataset_validation = TensorDataset(
        torch.tensor(validation_set_features).float(),
        torch.tensor(validation_set_targets).float())
    dataset_testing = TensorDataset(
        torch.tensor(testing_set_features).float(),
        torch.tensor(testing_set_targets).float())

    print_dataset_distributions(dataset_training, dataset_validation,
                                dataset_testing)

    dataloader_training = DataLoader(dataset_training,
                                     batch_size=BATCH_SIZE,
                                     shuffle=True)
    dataloader_validation = DataLoader(dataset_validation,
                                       batch_size=BATCH_SIZE,
                                       shuffle=False)
    dataloader_testing = DataLoader(dataset_testing,
                                    batch_size=BATCH_SIZE,
                                    shuffle=False)

    return dataloader_training, dataloader_validation, dataloader_testing


def print_dataset_distributions(training_dataset, validation_dataset,
                                test_dataset):

    def print_distribution(dataset, dataset_name):
        all_targets = [target.item() for _, target in dataset]

        unique, counts = np.unique(all_targets, return_counts=True)
        total = sum(counts)
        proportions = counts / total

        print(f"Distribution of target values in {dataset_name} set:")
        for value, count, proportion in zip(unique, counts, proportions):
            print(
                f"Potability: {int(value)} | Count: {count} | Proportion: {proportion:.6f}"
            )
        print()

    print_distribution(training_dataset, "training")
    print_distribution(validation_dataset, "validation")
    print_distribution(test_dataset, "testing")


def create_model():
    return nn.Sequential(nn.Linear(len(FEATURE_LABELS), 64), nn.ReLU(),
                         nn.Linear(64, 32), nn.ReLU(),
                         nn.Linear(32, len(TARGET_LABELS)), nn.Sigmoid())


def train_epoch(model, dataloader, optimizer, criterion, metric):
    model.train()
    running_loss = 0.0

    for inputs, targets in tqdm(dataloader, desc="Training"):
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        metric.update(outputs, targets)

    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_metric_value = metric.compute().item()
    metric.reset()
    return epoch_loss, epoch_metric_value


def validate_epoch(model, dataloader, criterion, metric):
    model.eval()
    running_loss = 0.0

    with torch.no_grad():
        for inputs, targets in tqdm(dataloader, desc="Validation"):
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            running_loss += loss.item() * inputs.size(0)
            metric.update(outputs, targets)

    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_metric_value = metric.compute().item()
    metric.reset()
    return epoch_loss, epoch_metric_value


def plot_results(training_losses, validation_losses, training_metrics,
                 validation_metrics):

    plt.subplot(1, 2, 1)
    plt.plot(training_losses, label='Train loss')
    plt.plot(validation_losses, label='Validation loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss per epoch')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(training_metrics, label='Train F1')
    plt.plot(validation_metrics, label='Validation F1')
    plt.xlabel('Epoch')
    plt.ylabel('Metric')
    plt.title('Metric per epoch')
    plt.legend()

    plt.tight_layout()
    plt.show()


def train_model(dataloader_training, dataloader_validation):
    model = create_model()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.BCELoss()

    training_metric = torchmetrics.F1Score(task="binary")
    validation_metric = torchmetrics.F1Score(task="binary")

    training_losses = []
    validation_losses = []
    training_metrics = []
    validation_metrics = []

    for epoch in range(EPOCHS):
        training_loss, training_metric_value = train_epoch(
            model, dataloader_training, optimizer, criterion, training_metric)
        validation_loss, validation_metric_value = validate_epoch(
            model, dataloader_validation, criterion, validation_metric)

        training_losses.append(training_loss)
        validation_losses.append(validation_loss)
        training_metrics.append(training_metric_value)
        validation_metrics.append(validation_metric_value)

        print(f"Epoch [{epoch+1}/{EPOCHS}]:")
        print(f"Average training loss: {training_loss:.16f}")
        print(f"Average validation loss: {validation_loss:.16f}")
        print(f"Training metric: {training_metric_value:.16f}")
        print(f"Validation metric: {validation_metric_value:.16f}")

    plot_results(training_losses, validation_losses, training_metrics,
                 validation_metrics)
    return model


def main():
    dataloader_training, dataloader_validation, _ = load_data()
    _ = train_model(dataloader_training, dataloader_validation)


if __name__ == '__main__':
    main()
