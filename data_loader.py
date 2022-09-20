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


class Block(nn.Module):
    def __init__(self, in_channel, identity_downsample=None, stride=1):
        super(Block, self).__init__()
        self.expansion = 4
        self.conv1 = nn.LazyConv1d(in_channel, kernel_size=5, stride=1, padding=0)
        self.bn1 = nn.LazyBatchNorm1d()
        self.conv2 = nn.LazyConv1d(in_channel, kernel_size=5, stride=1, padding=0)
        self.bn2 = nn.LazyBatchNorm1d()
        self.conv3 = nn.LazyConv1d(in_channel, kernel_size=1, stride=1, padding=0)
        self.bn3 = nn.LazyBatchNorm1d()
        self.relu = nn.ReLU()
        self.identity_downsample = identity_downsample

    def forward(self, x):
        identity = x

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.bn3(x)

        if self.identity_downsample:
            identity = self.identity_downsample(identity)

        x += identity

        return self.relu(x)


class Resnet(nn.Module):  # [3, 4, 6, 3] - how many times to use blocks
    def __init__(self, block, layers, channels, num_classes):
        super(Resnet, self).__init__()

        self.in_chanel = 2
        self.conv1 = nn.LazyConv1d(channels, kernel_size=7, stride=1, padding=3)
        self.bn1 = nn.LazyBatchNorm1d()
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool1d(kernel_size=7)

        # ResNet layers
        self.layer1 = self.make_layer(block, layers[0], stride=1)
        self.layer2 = self.make_layer(block, layers[1], stride=2)
        self.layer3 = self.make_layer(block, layers[2], stride=2)
        self.layer4 = self.make_layer(block, layers[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.LazyLinear(num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.reshape(x.shape[0], -1)

        x = self.fc(x)

        return x

    def make_layer(self, block, num_res_blocks, stride):
        identity_downsample = None
        layers = []

        if stride != 1:
            identity_downsample = nn.Sequential(
                nn.LazyConv1d(self.in_chanel, kernel_size=1, stride=stride),
                nn.LazyBatchNorm1d())

        layers.append(block(self.in_chanel, identity_downsample, stride))

        for i in range(num_res_blocks - 1):
            layers.append(block(self.in_chanel))

        return nn.Sequential(*layers)


def ResNet50(channels=2, num_classes=5):
    return Resnet(Block, [3, 4, 6, 3], channels, num_classes)


# Посчитать количество выходных слоев из одного слоя свертки (брать ширину свертки)
# Посчитать количесвто сверточных слоев, чтобы в конце получилось 5 выходных слоев, которые можно трактовать как
# Определенный класс для классификации

if __name__ == "__main__":
    dataset = ParametersDataset("train_ecg_parameters.npy", "train_y.npy")
    dataloader = DataLoader(dataset=dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

    n_iterations = ceil(TRAIN_SIZE / BATCH_SIZE)

    # model = NeuralNet(input_size=INPUT_SIZE, hidden_size=HIDDEN_SIZE, num_classes=NUM_CLASSES)
    model = ResNet50()

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)
    # model.train()

    for epoch in range(NUM_EPOCHS):
        for i, (inputs, labels) in enumerate(dataloader):

            # Reshaping the input if needed
            # inputs = inputs.reshape(-1, 180 * 2).to(device)
            inputs = inputs.permute(0, 2, 1)
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
