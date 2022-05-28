
import torch
import torch.nn.functional as F
from torch import nn, Tensor


model = nn.Sequential(
    nn.Conv2d(
        in_channels=1,
        out_channels=32,
        kernel_size=(3, 3),
    ),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=2),
    nn.Conv2d(
        in_channels=32,
        out_channels=64,
        kernel_size=(3, 3),
    ),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=2),
    nn.Conv2d(
        in_channels=64,
        out_channels=64,
        kernel_size=(3, 3),
    ),
    nn.Flatten(),
    nn.Linear(in_features=576, out_features=64),
    nn.ReLU(),
    nn.Linear(in_features=64, out_features=10)
)




class BasicCNN:
    base = CNNBase
    curve = CNNCurve
    kwargs = {
        'depth': 16,
        'batch_norm': False
    }