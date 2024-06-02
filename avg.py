import sys
import numpy as np
import datetime
import torch
from melanoma_detection.preprocess_dataset import (
    create_train_dataset,
    create_test_dataset,
    MelanomaDataset,
)
from torch.utils.data import DataLoader

import torch.optim as optim
from melanoma_detection.models import MelanomaNetworkV2
from melanoma_detection.regularization import EarlyStopping
import melanoma_detection.params as params
from melanoma_detection.transforms import TRANSFORM_TRAIN, TRANSFORM_VALIDATION


# torch.manual_seed(params.SEED)

acc_list = []
auc_list = []


for i in range(10):
    train_loader = DataLoader(
        MelanomaDataset(create_train_dataset(), transform=TRANSFORM_TRAIN),
        params.BATCH_SIZE,
        shuffle=True,
        num_workers=5,
    )

    test_loader = DataLoader(
        MelanomaDataset(create_test_dataset(), transform=TRANSFORM_VALIDATION),
        params.BATCH_SIZE,
        shuffle=False,
        num_workers=5,
    )

    net = MelanomaNetworkV2()

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
        False,
        False,
    )

    _, _, _, _, metrics = net.validate(test_loader, criterion, False)

    acc_list.append(metrics["accuracy"])
    auc_list.append(metrics["auc"])
    print(f"Trial {i+1}")


print(acc_list)
print(auc_list)

print("--------------------------------------")

print("Accuracy Average:", sum(acc_list) / len(acc_list))
print("AUC Average:", sum(auc_list) / len(auc_list))
