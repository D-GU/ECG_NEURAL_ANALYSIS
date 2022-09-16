import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms

from math import ceil
from itertools import chain
from torch.utils.data import Dataset, DataLoader

# Define hyper parameters
NUM_EPOCHS = 2
BATCH_SIZE = 5
VAL_SIZE = 2156
TRAIN_SIZE = 17111
INPUT_SIZE = 360
HIDDEN_SIZE = 100
NUM_CLASSES = 5
LEARNING_RATE = 0.01 / 100000

# Get device
device = torch.device("cpu")


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
        super(ConvNeuralNet, self).__init__()
        self.conv1 = nn.Conv1d(in_channel, out_channels=180 - 5 + 1, kernel_size=2, stride=1)
        self.pool = nn.MaxPool1d(1)
        self.conv2 = nn.Conv1d(180 - 5 + 1, out_channels=5, kernel_size=2, stride=5)
        self.fc1 = nn.Linear(2, num_classes)

    def forward(self, x):
        # print(f'this is x - {x}')
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        # x = x.reshape(x.shape[0], -1)
        x = self.fc1(x)

        return x


# Посчитать количество выходных слоев из одного слоя свертки (брать ширину свертки)
# Посчитать количесвто сверточных слоев, чтобы к конце получилось 5 выходных слоев, которые можно трактовать как
# Определенный класс для классификации

if __name__ == "__main__":
    dataset = ParametersDataset("train_ecg_parameters.npy", "train_y.npy")
    dataloader = DataLoader(dataset=dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

    n_iterations = ceil(TRAIN_SIZE / BATCH_SIZE)

    # model = NeuralNet(input_size=INPUT_SIZE, hidden_size=HIDDEN_SIZE, num_classes=NUM_CLASSES)
    model = ConvNeuralNet(2, NUM_CLASSES)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)
    model.train()
    for epoch in range(NUM_EPOCHS):
        for i, (inputs, labels) in enumerate(dataloader):

            # Reshaping the input if needed
            # inputs = inputs.reshape(-1, 180 * 2).to(device)
            inputs = inputs.permute(1, 2, 0)
            print("Input reshape - {}".format(inputs.shape))

            # fw
            # print(labels)
            outputs = model(inputs)
            loss = criterion(outputs, labels.float())

            # bw
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i + 1) % 100 == 0:
                print(f'epoch {epoch + 1}/{NUM_EPOCHS}, step {i + 1}/{TRAIN_SIZE}, loss = {loss.item():.4f}')
