import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torch.optim as optim
from melanoma_detection.preprocess_dataset import (
    create_train_dataset,
    create_test_dataset,
    MelanomaDataset,
)
from melanoma_detection.network import MelanomaNetwork, ResNet
import optuna

BATCH_SIZE = 32
EPOCHS = 15

# Imagenet normalization values
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]


def objective(trial):
    # Suggest hyperparameters
    lr = trial.suggest_float("lr", 1e-5, 1e-1, log=True)
    beta1 = trial.suggest_float("beta1", 0.8, 0.999)
    beta2 = trial.suggest_float("beta2", 0.8, 0.999)
    weight_decay = trial.suggest_float("weight_decay", 1e-10, 1e-3, log=True)

    # Suggest parameters for ColorJitter
    brightness = trial.suggest_float("brightness", 0.0, 1.0)
    contrast = trial.suggest_float("contrast", 0.0, 1.0)
    saturation = trial.suggest_float("saturation", 0.0, 1.0)
    hue = trial.suggest_float("hue", 0.0, 0.5)

    transform_train = transforms.Compose(
        [
            transforms.ColorJitter(
                brightness=brightness, contrast=contrast, saturation=saturation, hue=hue
            ),
            transforms.Resize((224, 224)),  # Resize the image to 224x224 pixels
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(20),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )

    transform_validation = transforms.Compose(
        [
            transforms.ColorJitter(
                brightness=brightness, contrast=contrast, saturation=saturation, hue=hue
            ),
            transforms.Resize((224, 224)),  # Resize the image to 224x224 pixels
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )
    train_loader = DataLoader(
        MelanomaDataset(create_train_dataset(), transform=transform_train),
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=5,
    )

    test_loader = DataLoader(
        MelanomaDataset(create_test_dataset(), transform=transform_validation),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=5,
    )

    # Create the network and define the optimizer
    net = MelanomaNetwork()

    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(
        net.parameters(), lr=lr, betas=(beta1, beta2), weight_decay=weight_decay
    )

    # Train the network using the fit method
    net.fit(train_loader, test_loader, EPOCHS, optimizer, criterion, False, 2)
    val_loss, val_accuracy, val_metrics = net.validate(
        train_loader, criterion, verbose=False
    )

    return val_loss


# Create a study object and optimize the objective function
study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=30)

print("Best trial:")
trial = study.best_trial
print(f"  Value: {trial.value}")
print("  Params: ")
for key, value in trial.params.items():
    print(f"    {key}: {value}")
