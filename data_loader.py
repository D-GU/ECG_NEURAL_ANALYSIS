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
NUM_EPOCHS = 1
BATCH_SIZE = 5
VAL_SIZE = 2156
TRAIN_SIZE = 17111
INPUT_SIZE = 360
HIDDEN_SIZE = 100
NUM_CLASSES = 5
LEARNING_RATE = 0.001

# Get device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_data(_path: str):
    return np.load(_path, allow_pickle=True)


def jackard_loss(set_a, set_b):
    unity = np.union1d(set_b, set_b)
    intersection = np.intersect1d(set_a, set_b)

    subtraction_fwd = np.subtract(set_a, set_b)
    subtraction_bwd = np.subtract(set_b, set_a)

    return np.divide(intersection, np.add(unity, subtraction_fwd, subtraction_bwd))


class ParametersDataset(Dataset):
    def __init__(self, _path_x, _path_y):
        _data_x = get_data(_path_x)
        _data_y = get_data(_path_y)

        self.x = np.array(
            [
                list(chain(*numpy.nan_to_num(_data_x[sample][::], nan=0))) for sample in range(_data_x.shape[0])
            ]
        )
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
    def __init__(self, in_channel, use_1x1conv=False, stride=1):
        super(Block, self).__init__()
        self.conv1 = nn.LazyConv1d(in_channel, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.LazyBatchNorm1d()
        self.conv2 = nn.LazyConv1d(in_channel, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.LazyBatchNorm1d()

        if use_1x1conv:
            self.conv3 = nn.LazyConv1d(in_channel, kernel_size=1, stride=1)
        else:
            self.conv3 = None

    def forward(self, x):
        y = F.relu(self.bn1(self.conv1(x)))
        y = self.bn2(self.conv2(y))

        if self.conv3:
            x = self.conv3(x)

        y += x

        return F.relu(y)


class ResNet(nn.Module):  # [3, 4, 6, 3] - how many times to use blocks
    def __init__(self, block, layers, channels, num_classes):
        super(ResNet, self).__init__()

        self.in_chanel = 2
        self.conv1 = nn.LazyConv1d(channels, kernel_size=7, stride=1, padding=1)
        self.bn1 = nn.LazyBatchNorm1d()
        self.relu = nn.ReLU()

        # ResNet layers
        self.layer1 = self.make_layer(block, layers[0], stride=2)
        self.layer2 = self.make_layer(block, layers[1], stride=2)
        self.layer3 = self.make_layer(block, layers[2], stride=2)
        self.layer4 = self.make_layer(block, layers[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.flatten = nn.Flatten()
        self.fc = nn.LazyLinear(num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.reshape(x.shape[0], -1)

        x = self.flatten(x)
        x = self.fc(x)

        return x

    def make_layer(self, block, num_res_blocks, stride):
        layers = [block(self.in_chanel, stride)]

        # if stride != 1:
        #     identity_downsample = nn.Sequential(
        #         nn.LazyConv1d(self.in_chanel, kernel_size=7, stride=stride),
        #         nn.LazyBatchNorm1d(),
        #         nn.ReLU(),
        #         nn.MaxPool1d(kernel_size=3, stride=2, padding=1))

        for blocks in range(num_res_blocks):
            layers.append(block(self.in_chanel))

        return nn.Sequential(*layers)


def ResNet_init(channels=2, num_classes=5):
    return ResNet(Block, [3, 4, 6, 3], channels, num_classes)


if __name__ == "__main__":
    dataset = ParametersDataset("train_ecg_parameters.npy", "train_y.npy")
    dataloader = DataLoader(dataset=dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

    dataset_val = ParametersDataset("val_ecg_parameters.npy", "val_y.npy")
    dataloader_val = DataLoader(dataset=dataset_val, batch_size=BATCH_SIZE, num_workers=0)

    n_iterations = ceil(TRAIN_SIZE / BATCH_SIZE)

    # model = NeuralNet(input_size=INPUT_SIZE, hidden_size=HIDDEN_SIZE, num_classes=NUM_CLASSES)
    model = ResNet_init()

    criterion = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)
    model.train()

    for epoch in range(NUM_EPOCHS):
        for i, (inputs, labels) in enumerate(dataloader):
            # Reshaping the input if needed

            inputs = inputs.permute(0, 2, 1).to(device)
            labels = labels.to(device)

            # fw
            outputs = model(inputs)
            loss = criterion(outputs, labels.float())

            # bw
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i + 1) % 100 == 0:
                print(f'epoch {epoch + 1}/{NUM_EPOCHS}, step {i + 1}/{TRAIN_SIZE}, loss = {loss.item():.4f}')

    with torch.no_grad():
        n_correct = 0
        n_samples = 0

        for i, (inputs, labels) in enumerate(dataloader_val):
            inputs = inputs.permute(0, 2, 1).to(device)
            labels = labels.to(device)
            outputs = model(inputs)

            prediction = outputs

            argmax_in_tensor = 0

            # Доработать проверку данных, чтобы узнать точность
            for i, tens in enumerate(prediction):
                print(f'sample №{i}: sample - {tens}')
                argmax_in_tensor = np.argmax(tens)
                print(f'label №{i}: label - {labels[i]}')
                print(f"Argmax in tensor: {argmax_in_tensor}")
            print('\n')

            n_samples += labels.shape[0]
            n_correct += (prediction == labels).sum().item()
