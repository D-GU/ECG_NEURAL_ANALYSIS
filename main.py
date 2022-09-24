import torch
import torch.nn as nn
import numpy as np
import pytorch_lightning as pl

from data_loader import ParametersDataset
from nn_module import ResNet_init
from torch.utils.data import DataLoader

# Define hyper parameters
NUM_EPOCHS = 100
BATCH_SIZE = 5
VAL_SIZE = 2156
TRAIN_SIZE = 17111
HIDDEN_SIZE = 100
NUM_CLASSES = 5
LEARNING_RATE = 0.01

# Get device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# def validation(_model, _dataloader):
#     with torch.no_grad():
#         _n_correct = [0 for i in range(2156)]
#         _n_samples = 0
#
#         for i, (_inputs, _labels) in enumerate(_dataloader):
#             _inputs = _inputs.permute(0, 2, 1).to(device)
#             _labels = _labels.to(device)
#             _outputs = _model(_inputs)
#
#             for out in enumerate(_outputs):
#                 print(f"out: {out}")
#                 _prediction = torch.max(out)
#                 _n_correct[i] += (_prediction == _labels[i]).sum().item()
#
#     return _n_correct


def train(
        _model,
        _dataloader,
        _device,
):
    file_name = "model_100_epochs_2.pth"

    _n_iterations = np.ceil(TRAIN_SIZE / BATCH_SIZE)
    _criterion = nn.MSELoss()
    _optimizer = torch.optim.SGD(_model.parameters(), lr=LEARNING_RATE)

    _model.cuda()

    for epoch in range(NUM_EPOCHS):
        for i, (inputs, labels) in enumerate(_dataloader):
            # Reshaping the input if needed

            inputs = inputs.permute(0, 2, 1).to(_device)
            labels = labels.to(_device)

            # fw
            outputs = _model(inputs)
            loss = _criterion(outputs, labels.float())

            # bw
            _optimizer.zero_grad()
            loss.backward()
            _optimizer.step()

            if (i + 1) % 100 == 0:
                print(f'epoch {epoch + 1}/{NUM_EPOCHS}, step {i + 1}/{TRAIN_SIZE}, loss = {loss.item():.4f}')

    torch.save(model, file_name)


if __name__ == "__main__":
    file_name = "model_100_epochs_1.pth"

    train_dataset = ParametersDataset("train_ecg_parameters.npy", "train_y.npy")
    dataloader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, num_workers=1)

    validation_dataset = ParametersDataset("val_ecg_parameters.npy", "val_y.npy")
    val_dataloader = DataLoader(dataset=validation_dataset, batch_size=BATCH_SIZE, num_workers=1)

    model = ResNet_init()
    train(model, dataloader, device)

    # model = torch.load(file_name)
    # model.eval()
    #
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #
    # with torch.no_grad():
    #     for i, (ins, lbl) in enumerate(val_dataloader):
    #         ins = ins.permute(0, 2, 1).to(device)
    #         lbl = lbl.to(device)
    #
    #         outputs = model(ins)
    #
    #         for j, tens in enumerate(outputs):
    #             print(f'Sample №{j}, sample: {tens}')
    #             print(f'label №{j}, label: {lbl[j]}')
