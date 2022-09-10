import torch
import torch.nn as nn  # All NN modules
import torch.optim as optim  # All optimisation algos
import torch.nn.functional as func  # Activations functions
import torchvision.transforms as transforms  # Transformations that can be applied to a dataset

import pandas as pd

from data_loader import get_data

# Get the device to operate(train) a nn
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Get input size and number of classes
input_size = 12 * 15
num_classes = 5

num_epochs = 10
batch_size = 10

# Добавить объяснение из книги о learning rate (позже)
learning_rate = 0.001

# Get train dataset
train_x = get_data("train_ecg_parameters.npy")
labels_tr = get_data("train_y.npy")

# Get val dataset
val_x = get_data("val_ecg_parameters.npy")
labels_vl = get_data("val_y.npy")

# Get classes
classes = ([1, 0, 0, 0, 0], [0, 1, 0, 0, 0], [0, 0, 1, 0, 0], [0, 0, 0, 1, 0], [0, 0, 0, 0, 1])

# Get steps
steps = train_x.shape[0]


class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()

    def forward(self):
        pass


model = ConvNet().to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=learning_rate)

for epoch in range(num_epochs):
    for i, (examples, labels) in (steps, (train_x, labels_tr)):
        table = pd.DataFrame(train_x)
