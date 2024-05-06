import torch
from melanoma_detection.preprocess_dataset import (
    create_train_dataset,
    MelanomaDataset,
)
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torch.optim as optim
from melanoma_detection.network import Net


BATCH_SIZE = 8
EPOCHS = 2

transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
)  # mean, Standard diviation,  channels

train_loader = DataLoader(
    MelanomaDataset(create_train_dataset(), transform=transform),
    BATCH_SIZE,
    shuffle=True,
    num_workers=5,
)

net = Net()

criterion = torch.nn.BCEWithLogitsLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

net.fit(train_loader, EPOCHS, optimizer, criterion)

PATH = "./cifar_net.pth"
net.save(PATH)
