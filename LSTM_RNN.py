# Time Series Recurrent Neural Network

import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import torch.optim

from torch.utils.data import DataLoader
from data_loader import ParametersDataset
from hyperparameters import hyperparameters


class LSTM(pl.LightningModule):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(LSTM, self).__init__()
        self.learning_rate = 1e-4
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True, dropout=0.2)
        # x -> (batch size, seq, input size)

        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # self.lstm.flatten_parameters()

        lstm_out, _ = self.lstm(x)
        y_pred = self.fc(lstm_out[:, -1])

        return y_pred

        # h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        # c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        # #
        # out, _ = self.lstm(x, (h0, c0))
        # # out: batch, seq_len, hidden_size
        # # out (N, 180, 128)
        # # out (N, 128) - needed
        # out = out[:, -1, :]
        # out = self.fc(out)
        #
        # return out

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

    def training_step(self, batch, batch_idx):
        inputs, labels = batch
        inputs = inputs.reshape(-1, hyperparameters["seq_length"], hyperparameters["input_size"])
        inputs = inputs.permute(0, 2, 1)
        labels = labels

        # fw
        outputs = self(inputs)
        loss = F.multilabel_soft_margin_loss(outputs, labels)

        return {"loss": loss}

    def train_dataloader(self):
        train_dataset = ParametersDataset("train_ecg_parameters.npy", "train_y.npy")
        return DataLoader(
            dataset=train_dataset,
            batch_size=hyperparameters["batch_size"],
            num_workers=4,
            shuffle=True
        )

    def val_dataloader(self):
        validation_dataset = ParametersDataset("val_ecg_parameters.npy", "val_y.npy")
        return DataLoader(
            dataset=validation_dataset,
            batch_size=hyperparameters["batch_size"],
            num_workers=4,
            shuffle=False
        )

    def validation_step(self, batch, batch_idx):
        inputs, labels = batch
        inputs.reshape(-1, hyperparameters["seq_length"], hyperparameters["input_size"])
        inputs = inputs.permute(0, 2, 1)
        labels = labels

        # fw
        outputs = self(inputs)
        loss = F.multilabel_soft_margin_loss(outputs, labels)
        # loss = nn.MultiLabelMarginLoss()(outputs, labels)

        return {"val_loss": loss}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        return {"val_loss": avg_loss}


def lstm_init(
        input_size=hyperparameters["input_size_lstm"],
        hidden_size=hyperparameters["hidden_size"],
        num_layers=hyperparameters["num_layers"],
        num_classes=hyperparameters["num_classes"],
):
    return LSTM(input_size, hidden_size, num_layers, num_classes)
