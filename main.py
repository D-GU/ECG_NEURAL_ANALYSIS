import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from data_loader import ParametersDataset
from nn_module import ResNet_init
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix
from sklearn.metrics import multilabel_confusion_matrix
from sklearn.metrics import classification_report
from pytorch_lightning import Trainer

# Define hyper parameters
NUM_EPOCHS = 150
BATCH_SIZE = 64
NUM_CLASSES = 5
lr = 0.01


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
):
    _file_name = "model_lightning_150_epochs_64_batch.pth"

    trainer = Trainer(gpus=1, max_epochs=NUM_EPOCHS, fast_dev_run=False)
    trainer.fit(_model)

    torch.save(_model, _file_name)

    return


def score_check(_model, _dataloader):
    _correct = 0
    _score = 0

    _n_samples = 0
    _n_correct = 0

    with torch.no_grad():
        for num_in_batch, (_ins, _lbl) in enumerate(_dataloader):
            _ins = _ins.permute(0, 2, 1)
            _outputs = _model(_ins)

            _predicted = []
            _threshold = 0.5

            for num_tensor, tensor in enumerate(_outputs):
                tensor[tensor >= _threshold] = 1
                tensor[tensor < _threshold] = 0

            conf_matrix = multilabel_confusion_matrix(_lbl, _outputs)

            labels_names = ["A", "B", "C", "D", "E"]
            _classification_report = classification_report(_lbl, _outputs, target_names=labels_names)
            print(_classification_report)
    return _score


if __name__ == "__main__":
    file_name = "model_lightning_100_epochs_MLML.pth"

    train_dataset = ParametersDataset("train_ecg_parameters.npy", "train_y.npy")
    dataloader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, num_workers=1)

    validation_dataset = ParametersDataset("val_ecg_parameters.npy", "val_y.npy")
    val_dataloader = DataLoader(dataset=validation_dataset, batch_size=BATCH_SIZE, num_workers=1)

    # model = ResNet_init()
    model = torch.load(file_name)
    model.eval()
    # train(model, dataloader)

    score_check(model, val_dataloader)
