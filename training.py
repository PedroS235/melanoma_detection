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

BATCH_SIZE = 4
EPOCHS = 2

transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
)  # mean, Standard diviation,  channels

train_set = MelanomaDataset(create_train_dataset(), transform=transform)
test_set = MelanomaDataset(create_test_dataset(), transform=transform)

train_loader = DataLoader(train_set, BATCH_SIZE, shuffle=True, num_workers=2)
test_loader = DataLoader(test_set, BATCH_SIZE, shuffle=False, num_workers=2)


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


net = Net()

criterion = nn.BCEWithLogitsLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

from tqdm import tqdm

for epoch in range(EPOCHS):
    running_loss = 0.0
    # Wrap train_loader with tqdm for a progress bar
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}")

    for i, data in enumerate(progress_bar):
        inputs, labels = data
        optimizer.zero_grad()

        # Forward and backward passes
        outputs = net(inputs).squeeze()
        loss = criterion(outputs, labels.float())
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        # Update progress bar description with current loss
        progress_bar.set_description(f"Loss: {running_loss / (i + 1):.6f}")

    # Optional: print summary after each epoch
    print(
        f"Epoch [{epoch + 1}/{EPOCHS}] - Loss: {running_loss / len(train_loader):.6f}"
    )


PATH = "./cifar_net.pth"
torch.save(net.state_dict(), PATH)
