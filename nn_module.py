import torch.nn as nn
import torch.nn.functional as F


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
