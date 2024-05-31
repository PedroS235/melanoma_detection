import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torch.optim as optim
from melanoma_detection import params
from melanoma_detection.models.melanoma import MelanomaNetworkV2
from melanoma_detection.regularization import EarlyStopping
from melanoma_detection.preprocess_dataset import (
    create_train_dataset,
    create_test_dataset,
    MelanomaDataset,
)
import melanoma_detection.params as params
from melanoma_detection.transforms import TRANSFORM_TRAIN, TRANSFORM_VALIDATION
import optuna


def objective(trial):
    # Suggest hyperparameters
    lr = trial.suggest_float("lr", 1e-5, 1e-1, log=True)
    # beta1 = trial.suggest_float("beta1", 0.8, 0.999)
    # beta2 = trial.suggest_float("beta2", 0.8, 0.999)
    weight_decay = trial.suggest_float("weight_decay", 1e-10, 1e-3, log=True)

    train_loader = DataLoader(
        MelanomaDataset(create_train_dataset(), transform=TRANSFORM_TRAIN),
        batch_size=params.BATCH_SIZE,
        shuffle=True,
        num_workers=10,
    )

    test_loader = DataLoader(
        MelanomaDataset(create_test_dataset(), transform=TRANSFORM_VALIDATION),
        batch_size=params.BATCH_SIZE,
        shuffle=False,
        num_workers=10,
    )

    # Create the network and define the optimizer
    net = MelanomaNetworkV2()

    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)

    # Train the network using the fit method
    net.fit(
        train_loader,
        test_loader,
        params.EPOCHS,
        optimizer,
        criterion,
        EarlyStopping(5, 0.001),
        True,
        False,
    )
    _, _, _, acc, _ = net.validate(train_loader, criterion, verbose=False)

    return acc


# Create a study object and optimize the objective function
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=30)

print("Best trial:")
trial = study.best_trial
print(f"  Value: {trial.value}")
print("  Params: ")
for key, value in trial.params.items():
    print(f"    {key}: {value}")
