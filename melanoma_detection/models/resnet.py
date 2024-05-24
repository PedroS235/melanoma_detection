import os
import sys
import torch.nn as nn
import torchvision.models as models
from melanoma_detection.models.base import BaseNetwork

PKG_DIR = os.path.dirname(os.path.abspath(sys.argv[0]))


class ResNet(BaseNetwork):
    def __init__(self):
        super().__init__()

        self.resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

        num_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_features, 1)

        self.resnet.to(self.device)

    def forward(self, x):
        return self.resnet(x)
