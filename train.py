import torch

from pytorch_lightning import Trainer
from hyperparameters import hyperparameters
from resnet_50_no_lazy import ResNet50


def train(_model, _filename: str):
    trainer = Trainer(gpus=1, max_epochs=hyperparameters["num_epochs"], auto_lr_find=True)
    trainer.tune(_model)

    trainer.fit(_model)
    torch.save(_model, _filename)

    return


if __name__ == "__main__":
    model = ResNet50()
    train(_model=model, _filename="name.pth")
