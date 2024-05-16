import torch
from melanoma_detection.preprocess_dataset import (
    create_train_dataset,
    create_test_dataset,
    MelanomaDataset,
)
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torch.optim as optim
from melanoma_detection.network import Net, ResNet
from melanoma_detection.img_utils import ImagePreprocessingPipeline


BATCH_SIZE = 4
EPOCHS = 2


# Imagenet normalization values
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

img_pipeline = ImagePreprocessingPipeline(2, 5, False)

transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize(mean, std)]
)  # mean, Standard diviation,  channels

train_loader = DataLoader(
    MelanomaDataset(create_train_dataset(), transform=transform, pipeline=img_pipeline),
    BATCH_SIZE,
    shuffle=True,
    num_workers=5,
)

test_loader = DataLoader(
    MelanomaDataset(create_test_dataset(), transform=transform, pipeline=img_pipeline),
    BATCH_SIZE,
    shuffle=False,
    num_workers=5,
)

net = Net()
# net = ResNet()

criterion = torch.nn.BCEWithLogitsLoss()
# optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
optimizer = optim.Adam(net.parameters())

net.fit(train_loader, test_loader, EPOCHS, optimizer, criterion)

PATH = "./cifar_net.pth"
net.save(PATH)
