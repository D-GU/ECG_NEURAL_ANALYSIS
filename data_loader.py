import torch
import torchvision
import numpy as np

from itertools import chain
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence


def get_data(_path: str):
    return np.load(_path, allow_pickle=True)


class ParametersDataset(Dataset):
    def __init__(self, _path_x, _path_y):
        _data_x = get_data(_path_x)
        _data_y = get_data(_path_y)

        self.x = np.array([list(chain(*_data_x[i][::])) for i in range(17111)])
        self.y = torch.from_numpy(_data_y)
        self.n_samples = self.x.shape[0]

    def __getitem__(self, index):
        return pad_sequence([torch.tensor(i) for i in self.x[index]], batch_first=True), self.y[index]

    def __len__(self):
        return self.n_samples


if __name__ == "__main__":
    dataset = ParametersDataset("train_ecg_parameters.npy", "train_y.npy")

    first_data = dataset[1]
    features, labels = dataset[0]
    print(features)
