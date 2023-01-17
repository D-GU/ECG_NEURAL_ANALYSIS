import torch
import numpy as np

from data_loader import ParametersDataset
from torch.utils.data import DataLoader
from sklearn.metrics import multilabel_confusion_matrix, confusion_matrix
from hyperparameters import hyperparameters
from pandas import DataFrame


def get_confusion_matrix_precision(matrix):
    return np.diag(matrix).sum() / matrix.sum()


def score_check(_model, _dataloader):
    _n_samples = 0
    _threshold = 0.5

    _predicted = []
    _labels = []

    with torch.no_grad():
        for num_in_batch, (_ins, _lbl) in enumerate(_dataloader):
            _ins = _ins.permute(0, 2, 1)
            _outputs = _model(_ins)

            for num_tensor, tensor in enumerate(_outputs):
                _n_samples += 1
                tensor[tensor >= _threshold] = 1
                tensor[tensor < _threshold] = 0

                _predicted.append(np.asarray(tensor))
                _labels.append(np.asarray(_lbl[num_tensor]))

    _precision_per_class = 0
    _precision_overall = 0

    _confusion_matrix_per_sample = multilabel_confusion_matrix(_labels, _predicted, samplewise=True)
    _confusion_matrix_per_class = multilabel_confusion_matrix(_labels, _predicted)

    _class_precision = {f"{class_id}": 0 for class_id in range(hyperparameters["num_classes"])}

    for _id, matrix in enumerate(_confusion_matrix_per_class):
        _precision_per_class = get_confusion_matrix_precision(matrix)
        _class_precision[str(_id)] = _precision_per_class * 100

    for matrix in _confusion_matrix_per_sample:
        _precision_overall += get_confusion_matrix_precision(matrix) * 100

    return _class_precision, _precision_overall / _n_samples


if __name__ == "__main__":
    file_name = "lstm_ecg_features_try_5.pth"

    validation_dataset = ParametersDataset("val_ecg_parameters.npy", "val_y.npy")
    val_dataloader = DataLoader(dataset=validation_dataset, batch_size=hyperparameters["batch_size"], num_workers=4)

    model = torch.load(file_name)
    model.eval()

    class_precision, overall_precision = score_check(model, val_dataloader)
    df = DataFrame([class_precision])

    print(df)
    print(f"Overall precision in percent = {overall_precision:.3f}%")
