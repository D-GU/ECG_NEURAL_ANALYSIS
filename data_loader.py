import math

import torch
import torchvision
import numpy as np

from math import ceil
from itertools import chain
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

TRAIN_SIZE = 17111
VAL_SIZE = 2156


def get_data(_path: str):
    return np.load(_path, allow_pickle=True)


class ParametersDataset(Dataset):
    def __init__(self, _path_x, _path_y):
        _data_x = get_data(_path_x)
        _data_y = get_data(_path_y)

        self.x = np.array([list(chain(*_data_x[sample][::])) for sample in range(TRAIN_SIZE)])
        self.y = torch.from_numpy(_data_y)
        self.n_samples = _data_x.shape[0]

    def __getitem__(self, index):
        return pad_sequence([torch.tensor(i) for i in self.x[index]], batch_first=True), self.y[index]

    def __len__(self):
        return self.n_samples


if __name__ == "__main__":
    dataset = ParametersDataset("train_ecg_parameters.npy", "train_y.npy")
    dataloader = DataLoader(dataset=dataset, batch_size=4, shuffle=True, num_workers=2)

    # first_data = dataset
    # features, labels = first_data[5]
    # print(features)

    num_epochs = 2
    total_samples = len(dataset)
    n_iterations = math.ceil(total_samples / 4)


