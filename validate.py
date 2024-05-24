import sys
import torch
from melanoma_detection.preprocess_dataset import (
    create_test_dataset,
    MelanomaDataset,
)
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from melanoma_detection.models.melanoma import MelanomaNetwork

# Imagenet normalization values
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

BATCH_SIZE = 32
EPOCHS = 2

transform = transforms.Compose(
    [
        # transforms.ColorJitter(brightness=0.0, contrast=0.2, saturation=0.5, hue=0.0),
        transforms.Resize((224, 224)),  # Resize the image to 224x224 pixels
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ]
)

test_loader = DataLoader(
    MelanomaDataset(create_test_dataset(), transform=transform),
    BATCH_SIZE,
    shuffle=False,
    num_workers=5,
)

criterion = torch.nn.BCEWithLogitsLoss()

net = MelanomaNetwork()

PATH = sys.argv[1]

net.load(PATH)
net.validate(test_loader, criterion)
