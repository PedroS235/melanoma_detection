import sys
import datetime
import torch
from melanoma_detection.preprocess_dataset import (
    create_train_dataset,
    create_test_dataset,
    MelanomaDataset,
)
from torch.utils.data import DataLoader, random_split

import torch.optim as optim
from melanoma_detection.models import MelanomaNetworkV2
from melanoma_detection.regularization import EarlyStopping
import melanoma_detection.params as params
from melanoma_detection.transforms import TRANSFORM_TRAIN, TRANSFORM_VALIDATION


torch.manual_seed(params.SEED)


# Load the full training dataset
full_train_dataset = MelanomaDataset(create_train_dataset(), transform=TRANSFORM_TRAIN)

# Define the validation split ratio
validation_ratio = 0.2
train_size = int((1 - validation_ratio) * len(full_train_dataset))
val_size = len(full_train_dataset) - train_size

# Split the dataset into training and validation sets
train_dataset, val_dataset = random_split(full_train_dataset, [train_size, val_size])

# Create DataLoaders for training, validation, and test sets
train_loader = DataLoader(
    train_dataset,
    batch_size=params.BATCH_SIZE,
    shuffle=True,
    num_workers=5,
)

val_loader = DataLoader(
    val_dataset,
    batch_size=params.BATCH_SIZE,
    shuffle=False,
    num_workers=5,
)

test_loader = DataLoader(
    MelanomaDataset(create_test_dataset(), transform=TRANSFORM_VALIDATION),
    batch_size=params.BATCH_SIZE,
    shuffle=False,
    num_workers=5,
)



if __name__ == '__main__':
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
        val_loader,
        params.EPOCHS,
        optimizer,
        criterion,
        EarlyStopping(5, 0.001),
        True,
    )

    DEFAULT_PATH = f"./models/{datetime.datetime.now().time()}.pth"
    net.save(DEFAULT_PATH) if len(sys.argv) == 1 else net.save(sys.argv[1])
