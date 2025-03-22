import pandas as pd
from torch.utils.data import Dataset, DataLoader


class WaterPotabilityDataset(Dataset):

    def __init__(self, training_data_file):
        super().__init__()
        self.water_potability_data = pd.read_csv(training_data_file)

    def __len__(self):
        return len(self.water_potability_data)

    def __getitem__(self, index):
        # https://pandas.pydata.org/docs/reference/api/pandas.Series.to_numpy.html
        np_array = self.water_potability_data.iloc[index].to_numpy()
        return np_array[:-1], np_array[-1]


def main():
    water_potability_dataset = WaterPotabilityDataset(
        "../DATA/water_train.csv")

    print(f"Number of instances: {water_potability_dataset.__len__()}")
    print(f"Fifth item: {water_potability_dataset.__getitem__(4)}")

    water_potability_dataloader = DataLoader(water_potability_dataset,
                                             batch_size=2,
                                             shuffle=True)
    train_features, train_labels = next(iter(water_potability_dataloader))

    print(train_features, train_labels)


if __name__ == '__main__':
    main()
