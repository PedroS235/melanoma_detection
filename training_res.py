import sys
import datetime
import torch
from melanoma_detection.preprocess_dataset import (
    create_train_dataset,
    create_test_dataset,
    MelanomaDataset,
)
from torch.utils.data import DataLoader

import torch.optim as optim
from melanoma_detection.models import ResNet
from melanoma_detection.regularization import EarlyStopping
import melanoma_detection.params as params
from melanoma_detection.transforms import TRANSFORM_TRAIN, TRANSFORM_VALIDATION


torch.manual_seed(params.SEED)


train_loader = DataLoader(
    MelanomaDataset(create_train_dataset(), transform=TRANSFORM_TRAIN),
    params.BATCH_SIZE_RESNET,
    shuffle=True,
    num_workers=5,
)

test_loader = DataLoader(
    MelanomaDataset(create_test_dataset(), transform=TRANSFORM_VALIDATION),
    params.BATCH_SIZE_RESNET,
    shuffle=False,
    num_workers=5,
)

net = ResNet()

criterion = torch.nn.BCEWithLogitsLoss()

optimizer = optim.Adam(
    net.parameters(),
    lr=0.0002288372831567034,
    betas=(0.8378331684737104, 0.9034144582679383),
    weight_decay=3.358130934902445e-09,
)

net.fit(
    train_loader,
    test_loader,
    params.EPOCHS,
    optimizer,
    criterion,
    EarlyStopping(5, 0.001),
    True,
)

DEFAULT_PATH = f"./models/{datetime.datetime.now().time()}.pth"
net.save(DEFAULT_PATH) if len(sys.argv) == 1 else net.save(sys.argv[1])
