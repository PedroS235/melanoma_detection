import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from melanoma_detection.early_stopping import EarlyStopping
from melanoma_detection.metrics_utils import plot_metrics, compute_metrics
from melanoma_detection.models.base import BaseNetwork

PKG_DIR = os.path.dirname(os.path.abspath(sys.argv[0]))


class MelanomaNetwork(BaseNetwork):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.5)  # Dropout layer with 50% drop rate
        self.fc1 = nn.Linear(128 * 28 * 28, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 1)
        self.to(self.device)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool2(F.relu(self.conv3(x)))
        x = x.flatten(1)  # Flatten starting from the 1st dimension
        x = F.relu(self.fc1(x))
        x = self.dropout(x)  # Apply dropout during training
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
