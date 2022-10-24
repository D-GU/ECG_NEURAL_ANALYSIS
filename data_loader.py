import torch
import numpy as np

from itertools import chain
from torch.utils.data import Dataset


def get_data(_path: str):
    return np.load(_path, allow_pickle=True)


class ParametersDataset(Dataset):
    def __init__(self, _path_x, _path_y):
        _data_x = get_data(_path_x)
        _data_y = get_data(_path_y)

        self.x = np.array(
            [
                list(chain(*np.nan_to_num(_data_x[sample][::], nan=0))) for sample in range(_data_x.shape[0])
            ]
        )
        self.y = torch.tensor(_data_y)
        self.n_samples = _data_x.shape[0]

    def __getitem__(self, index):
        sample_x = np.array([sample for sample in self.x[index]])
        sample_x = np.array([np.nan_to_num(parameter, nan=0) for parameter in sample_x])
        tensor_sample_x = torch.FloatTensor(sample_x)

        return tensor_sample_x, self.y[index]

    def __len__(self):
        return self.n_samples
