import torch
import torchvision
import numpy as np

from itertools import chain
from torch.utils.data import Dataset, DataLoader


def get_data(_path: str):
    return np.load(_path, allow_pickle=True)


class ParametersDataset(Dataset):
    def __init__(self, _path_x, _path_y):
        _data_x = get_data(_path_x)
        _data_y = get_data(_path_y)

        _data_x = np.array(np.array([list(chain(*i)) for i in _data_x[::]]))

        self.x = torch.from_numpy(_data_x)
        self.y = torch.from_numpy(_data_y)
        self.n_samples = self.x.shape[0]

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return self.n_samples


if __name__ == "__main__":
    dataset = ParametersDataset("train_ecg_parameters.npy", "train_y.npy")

    first_data = dataset[1]
    features, labels = dataset[0]
    print(features)
