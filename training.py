import torch
from melanoma_detection.preprocess_dataset import (
    create_train_dataset,
    create_test_dataset,
    MelanomaDataset,
)
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torch.optim as optim
from melanoma_detection.models import MelanomaNetwork, ResNet
from melanoma_detection.models.base import StoppingCriteria


BATCH_SIZE = 32
EPOCHS = 20


# Imagenet normalization values
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

transform_train = transforms.Compose(
    [
        # transforms.ColorJitter(
        #     brightness=0.1863956988359896,
        #     contrast=0.3355207249261889,
        #     saturation=0.43541353636866553,
        #     hue=0.07548291582198652,
        # ),
        transforms.Resize((224, 224)),  # Resize the image to 224x224 pixels
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(20),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ]
)

transform_validation = transforms.Compose(
    [
        # transforms.ColorJitter(
        #     brightness=0.1863956988359896,
        #     contrast=0.3355207249261889,
        #     saturation=0.43541353636866553,
        #     hue=0.07548291582198652,
        # ),
        transforms.Resize((224, 224)),  # Resize the image to 224x224 pixels
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ]
)

train_loader = DataLoader(
    MelanomaDataset(create_train_dataset(), transform=transform_train),
    BATCH_SIZE,
    shuffle=True,
    num_workers=5,
)

test_loader = DataLoader(
    MelanomaDataset(create_test_dataset(), transform=transform_validation),
    BATCH_SIZE,
    shuffle=False,
    num_workers=5,
)

net = MelanomaNetwork()
# net = ResNet()

criterion = torch.nn.BCEWithLogitsLoss()

optimizer = optim.Adam(
    net.parameters(),
    # lr=6.051040788116986e-05,
    # betas=(0.8362066397681903, 0.9907655800812818),
    # weight_decay=3.6250408963045035e-10,
    lr=0.0002288372831567034,
    betas=(0.8378331684737104, 0.9034144582679383),
    weight_decay=3.358130934902445e-09,
)

net.fit(
    train_loader,
    test_loader,
    EPOCHS,
    optimizer,
    criterion,
    StoppingCriteria(3),
    True,
)

PATH = "./best_model"
net.save(PATH)
