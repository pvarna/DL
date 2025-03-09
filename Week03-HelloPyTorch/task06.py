import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import TensorDataset, DataLoader
import pandas as pd
from sklearn import preprocessing
from tqdm import tqdm
import matplotlib.pyplot as plt

# Constants
DATA_PATH = "../Data/ds_salaries.csv"
BATCH_SIZE = 8
LEARNING_RATE = 0.001
EPOCHS = 20

# Define activation functions
ACTIVATION_FUNCTIONS = {
    "nn_with_sigmoid": nn.Sigmoid(),
    "nn_with_relu": nn.ReLU(),
    "nn_with_leakyrelu": nn.LeakyReLU()
}


def load_and_preprocess_data():
    df = pd.read_csv(DATA_PATH)
    target = "salary_in_usd"
    features = [
        "experience_level", "employment_type", "remote_ratio", "company_size"
    ]

    encoder = preprocessing.OneHotEncoder()
    encoded_features = encoder.fit_transform(df[features]).toarray()

    normalizer = preprocessing.Normalizer()
    normalized_features = normalizer.fit_transform(encoded_features)
    normalized_target = normalizer.fit_transform(df[[target]])

    dataset = TensorDataset(
        torch.tensor(normalized_features).float(),
        torch.tensor(normalized_target).float())

    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    return dataloader, normalized_features.shape[1]


def create_model(activation_function, input_size, output_size):
    return nn.Sequential(nn.Linear(input_size, 5), activation_function,
                         nn.Linear(5, output_size))


def train_model(name, activation_function, dataloader, input_size,
                output_size):
    print(f"Training model: {name}")
    model = create_model(activation_function, input_size, output_size)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.MSELoss()

    average_losses_per_epoch = []
    for epoch in range(EPOCHS):
        batch_losses = []
        for batch_features, batch_target in tqdm(
                dataloader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
            optimizer.zero_grad()
            prediction = model(batch_features)
            loss = criterion(prediction, batch_target)
            loss.backward()
            optimizer.step()
            batch_losses.append(loss.item())

        average_loss = torch.tensor(batch_losses).mean().item()
        average_losses_per_epoch.append(average_loss)
        print(f"Epoch [{epoch+1}/{EPOCHS}]: Average loss: {average_loss}")

    return average_losses_per_epoch


def plot_losses(losses):
    _, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, (name, loss_values) in zip(axes, losses.items()):
        ax.plot(loss_values)
        ax.set_title(name)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Average Training Loss")
    plt.tight_layout()
    plt.show()


def main():
    dataloader, input_size = load_and_preprocess_data()
    output_size = 1

    losses = {}
    best_model = None
    best_loss = float("inf")

    for name, activation_function in ACTIVATION_FUNCTIONS.items():
        average_losses = train_model(name, activation_function, dataloader,
                                     input_size, output_size)
        losses[name] = average_losses

        if average_losses[-1] < best_loss:
            best_loss = average_losses[-1]
            best_model = name

    print(f"Lowest loss of {best_loss} was achieved by model {best_model}.")
    plot_losses(losses)


if __name__ == '__main__':
    main()
