import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from preprocess_dataset import (
    create_test_dataset,
    create_train_dataset,
    MelanomaDataset,
)
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torch.optim as optim
from tqdm import tqdm
import matplotlib.pyplot as plt
import torchvision
import numpy as np


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=5, padding=2)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, padding=2)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=5, padding=2)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(
            128 * 28 * 28, 512
        )  # Corrected to match the actual flattened size
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 1)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool2(F.relu(self.conv3(x)))
        x = x.flatten(1)  # Flatten starting from the 1st dimension
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


BATCH_SIZE = 4
EPOCHS = 2

transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
)  # mean, Standard diviation,  channels

train_set = MelanomaDataset(create_train_dataset(), transform=transform)
test_set = MelanomaDataset(create_test_dataset(), transform=transform)

train_loader = DataLoader(train_set, BATCH_SIZE, shuffle=True, num_workers=2)
test_loader = DataLoader(test_set, BATCH_SIZE, shuffle=True, num_workers=2)

PATH = sys.argv[1]
net = Net()
net.load_state_dict(torch.load(PATH))


# def imshow(img):
#     img = img / 2 + 0.5  # unnormalize
#     npimg = img.numpy()
#     plt.imshow(np.transpose(npimg, (1, 2, 0)))
#     plt.show()
#
#
# dataiter = iter(test_loader)
#
# for _ in range(5):
#     images, labels = dataiter.__next__()
#
#     outputs = net(images)
#
#     predicted = torch.sigmoid(outputs)
#     predicted_labels = (predicted > 0.5).float()
#
#     imshow(torchvision.utils.make_grid(images))
#     print("GroundTruth: ", " ".join(f"{labels[j]}" for j in range(4)))
#     print("Predicted: ", " ".join(f"{predicted_labels[j]}" for j in range(4)))

net.eval()  # Set the model to evaluation mode
correct = 0
total = 0

with torch.no_grad():  # Turn off gradients for validation, saves memory and computations
    for data in test_loader:
        inputs, labels = data
        outputs = net(inputs)

        # Apply sigmoid to convert logits to probabilities
        probabilities = torch.sigmoid(outputs).squeeze()

        # Convert probabilities to predicted class (0 or 1)
        predicted_labels = (probabilities > 0.5).float()  # Using 0.5 as the threshold

        total += labels.size(0)
        correct += (predicted_labels == labels).sum().item()

print(f"Accuracy of the network on the validation set: {100 * correct / total:.2f}%")
