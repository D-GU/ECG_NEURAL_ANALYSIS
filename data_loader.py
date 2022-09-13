import math

import torch
import torch.nn as nn
import numpy as np
import torchvision
import torch.nn.functional as F

from math import ceil
from itertools import chain
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from torchvision.transforms import ToTensor

# Define hyper parameters
VAL_SIZE = 2156
TRAIN_SIZE = 17111
INPUT_SIZE = 360
HIDDEN_SIZE = 100
NUM_CLASSES = 5
NUM_EPOCHS = 2
BATCH_SIZE = 4
LEARNING_RATE = 0.01

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_data(_path: str):
    return np.load(_path, allow_pickle=True)


class ParametersDataset(Dataset):
    def __init__(self, _path_x, _path_y):
        _data_x = get_data(_path_x)
        _data_y = get_data(_path_y)

        self.x = np.array([list(chain(*_data_x[sample][::])) for sample in range(TRAIN_SIZE)])
        self.y = torch.tensor(_data_y)
        self.n_samples = _data_x.shape[0]

    def __getitem__(self, index):
        sample_x = np.array([sample for sample in self.x[index]])
        tensor_sample_x = torch.tensor(sample_x)
        return tensor_sample_x, self.y[index]

    def __len__(self):
        return self.n_samples


class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(NeuralNet, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(hidden_size, 1)
        self.softmax = nn.Softmax()

    def forward(self, x):
        out = self.linear1(x)
        out = self.relu(out)
        out = self.linear2(out)
        out = self.softmax(out)

        return out


if __name__ == "__main__":
    dataset = ParametersDataset("train_ecg_parameters.npy", "train_y.npy")
    dataloader = DataLoader(dataset=dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

    n_iterations = ceil(TRAIN_SIZE / BATCH_SIZE)

    # dummy loop
    for epochs in range(NUM_EPOCHS):
        for i, (inputs, labels) in enumerate(dataloader):
            if (i + 1) % 5 == 0:
                print(f"epoch {epochs + 1} / {NUM_EPOCHS}, step {i + 1}/ {n_iterations}, inputs {inputs.shape}")
