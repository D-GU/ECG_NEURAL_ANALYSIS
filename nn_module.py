import torch.nn as nn
import torch.nn.functional as F

from data_loader import ParametersDataset


class ConvNeuralNet(nn.Module):
    def __init__(self, in_channel, num_classes):
        super(ConvNeuralNet, self).__init__()
        self.conv1 = nn.Conv1d(in_channel, out_channels=10, kernel_size=1)
        self.pool = nn.MaxPool1d(1)
        self.conv2 = nn.Conv1d(10, out_channels=5, kernel_size=1)
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
