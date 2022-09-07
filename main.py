import numpy as np
import torch as tc

# Get the size of datasets
TRAIN_SIZE = 17111
VAL_SIZE = 2156

# Parameters labels
LABELS = [
    "Q_Dur", "R_Dur", "S_Dur", "T_Dur", "P_Dur",
    "Q_Amp", "R_Amp", "S_Amp", "T_Amp", "P_Amp",
    "QT_Interval", "PQ_Interval", "RR_Interval",
    "PR_Interval", "QRS_Interval"
]

# Get train datasets
train_x = np.load("train_ecg_parameters.npy", allow_pickle=True)
train_y = np.load("train_y.npy")

# Get validation datasets
val_x = np.load("val_ecg_parameters.npy", allow_pickle=True)
val_y = np.load("val_y.npy")

