import torch
import numpy as np
import torch.nn.functional as F

from data_loader import ParametersDataset
from nn_module import ResNet_init

from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix
from sklearn.metrics import multilabel_confusion_matrix
from sklearn.metrics import classification_report

from hyperparameters import hyperparameters
from resnet_50_no_lazy import ResNet50


def score_check(_model, _dataloader):
    _correct = 0
    _score = 0

    _n_samples = 0
    _n_correct = 0

    _predicted = []
    _labels = []

    with torch.no_grad():
        for num_in_batch, (_ins, _lbl) in enumerate(_dataloader):
            _ins = _ins.permute(0, 2, 1)
            _outputs = _model(_ins)

            _threshold = 0.5

            for num_tensor, tensor in enumerate(_outputs):
                # tensor = F.sigmoid(tensor)
                tensor[tensor >= _threshold] = 1
                tensor[tensor < _threshold] = 0
                print(f"{tensor} / {_lbl[num_tensor]}")
                _predicted.append(np.asarray(tensor))
                _labels.append(np.asarray(_lbl[num_tensor]))

    print(classification_report(_labels, _predicted))
    return _score


if __name__ == "__main__":
    file_name = "ResNet50_MLML_EPOCHS_32_BATCH_16.pth"

    validation_dataset = ParametersDataset("val_ecg_parameters.npy", "val_y.npy")
    val_dataloader = DataLoader(dataset=validation_dataset, batch_size=hyperparameters["batch_size"], num_workers=4)

    model = torch.load(file_name)
    model.eval()

    score_check(model, val_dataloader)
