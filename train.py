import torch

from pytorch_lightning import Trainer
from hyperparameters import hyperparameters
from resnet_50_no_lazy import ResNet50
from LSTM_RNN import lstm_init

def train(_model, _filename: str):
    trainer = Trainer(gpus=1, max_epochs=hyperparameters["num_epochs"])
    trainer.tune(_model)

    trainer.fit(_model)
    torch.save(_model, _filename)

    return


if __name__ == "__main__":
    model = lstm_init()
    train(_model=model, _filename="lstm_ecg_features_try_5.pth")
