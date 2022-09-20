import os

import numpy
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms

from math import ceil
from itertools import chain
from torch.utils.data import Dataset, DataLoader

# Define hyper parameters
NUM_EPOCHS = 100
BATCH_SIZE = 5
VAL_SIZE = 2156
TRAIN_SIZE = 17111
INPUT_SIZE = 360
HIDDEN_SIZE = 100
NUM_CLASSES = 5
LEARNING_RATE = 0.0001

# Get device
device = torch.device("cpu")


def get_data(_path: str):
    return np.load(_path, allow_pickle=True)


class ParametersDataset(Dataset):
    def __init__(self, _path_x, _path_y):
        _data_x = get_data(_path_x)
        _data_y = get_data(_path_y)

        self.x = np.array([list(chain(*numpy.nan_to_num(_data_x[sample][::], nan=0))) for sample in range(TRAIN_SIZE)])
        self.y = torch.tensor(_data_y)
        self.n_samples = _data_x.shape[0]

    def __getitem__(self, index):
        sample_x = np.array([sample for sample in self.x[index]])
        sample_x = np.array([np.nan_to_num(parameter, nan=0.0) for parameter in sample_x])
        tensor_sample_x = torch.FloatTensor(sample_x)

        return tensor_sample_x, self.y[index]

    def __len__(self):
        return self.n_samples


class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out = self.linear1(x)
        out = self.relu(out)
        out = self.linear2(out)

        return out


class ConvNeuralNet(nn.Module):
    def __init__(self, in_channel, num_classes):
        # It does something with the first layer on the first iteration
        # Applied more convolution layers to check
        super(ConvNeuralNet, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=in_channel, out_channels=180 - 5 + 1, kernel_size=21, stride=3)
        self.pool = nn.MaxPool1d(2)
        self.conv2 = nn.Conv1d(in_channels=180 - 5 + 1, out_channels=64, kernel_size=21, stride=3)
        self.conv3 = nn.Conv1d(in_channels=64, out_channels=32, kernel_size=21, stride=3)
        self.conv4 = nn.Conv1d(in_channels=32, out_channels=16, kernel_size=21, stride=3)
        self.conv5 = nn.Conv1d(in_channels=16, out_channels=8, kernel_size=21, stride=3)
        self.conv6 = nn.Conv1d(in_channels=8, out_channels=5, kernel_size=21, stride=3)
        self.fc1 = nn.Linear(1, num_classes)

    def forward(self, x):
        # print(x)
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        # x = x.reshape(x.shape[0], -1)
        x = self.fc1(x)

        return x

    def backward(self, x):
        pass


class Residual(nn.Module):  # @save
    """The Residual block of ResNet."""

    def __init__(self, num_channels, use_1x1conv=False, strides=1):
        super().__init__()
        self.conv1 = nn.LazyConv1d(num_channels, kernel_size=3, padding=1,
                                   stride=strides)
        self.conv2 = nn.LazyConv1d(num_channels, kernel_size=3, padding=1)
        if use_1x1conv:
            self.conv3 = nn.LazyConv1d(num_channels, kernel_size=1,
                                       stride=strides)
        else:
            self.conv3 = None
        self.bn1 = nn.LazyBatchNorm1d()
        self.bn2 = nn.LazyBatchNorm1d()

    def forward(self, X):
        Y = F.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3:
            X = self.conv3(X)
        Y += X
        return F.relu(Y)


class ResNet(nn.Module):
    def __init__(self, arch, lr=LEARNING_RATE, num_classes=5):
        super(ResNet, self).__init__()
        # self.save_hyperparameters()
        self.net = nn.Sequential(self.b1())
        for i, b in enumerate(arch):
            self.net.add_module(f'b{i + 2}', self.block(*b, first_block=(i == 0)))
        self.net.add_module('last', nn.Sequential(
            nn.AdaptiveAvgPool1d(1), nn.Flatten(),
            nn.LazyLinear(num_classes)))

    def b1(self):
        return nn.Sequential(
            nn.LazyConv1d(2, kernel_size=7, stride=2, padding=3),
            nn.LazyBatchNorm1d(), nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1))

    def block(self, num_residuals, num_channels, first_block=False):
        blk = []
        for i in range(num_residuals):
            if i == 0 and not first_block:
                blk.append(Residual(num_channels, use_1x1conv=True, strides=2))
            else:
                blk.append(Residual(num_channels))
        return nn.Sequential(*blk)


class ResNet18(ResNet):
    def __init__(self, arch, lr=LEARNING_RATE, num_classes=5):
        super().__init__(arch, lr, num_classes)


# Посчитать количество выходных слоев из одного слоя свертки (брать ширину свертки)
# Посчитать количесвто сверточных слоев, чтобы в конце получилось 5 выходных слоев, которые можно трактовать как
# Определенный класс для классификации

if __name__ == "__main__":
    dataset = ParametersDataset("train_ecg_parameters.npy", "train_y.npy")
    dataloader = DataLoader(dataset=dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

    n_iterations = ceil(TRAIN_SIZE / BATCH_SIZE)

    # model = NeuralNet(input_size=INPUT_SIZE, hidden_size=HIDDEN_SIZE, num_classes=NUM_CLASSES)
    model = ResNet18(((2, 180), (2, 32)))

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)
    # model.train()

    for epoch in range(NUM_EPOCHS):
        for i, (inputs, labels) in enumerate(dataloader):

            # Reshaping the input if needed
            # inputs = inputs.reshape(-1, 180 * 2).to(device)
            # inputs = inputs.permute(0, 2, 1)
            # print("Input reshape - {}".format(inputs.shape))

            # fw
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # bw
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i + 1) % 100 == 0:
                print(f'epoch {epoch + 1}/{NUM_EPOCHS}, step {i + 1}/{TRAIN_SIZE}, loss = {loss.item():.4f}')
