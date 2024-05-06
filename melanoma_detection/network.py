import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

PKG_DIR = os.path.dirname(os.path.abspath(sys.argv[0]))


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

        self.to(self.device)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool2(F.relu(self.conv3(x)))
        x = x.flatten(1)  # Flatten starting from the 1st dimension
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def fit(self, x_loader, epochs, optimizer, criterion):
        self.train()
        for epoch in range(epochs):
            running_loss = 0.0
            progress_bar = tqdm(x_loader, desc=f"Epoch {epoch + 1}")

            for i, data in enumerate(progress_bar):
                inputs, labels = data[0].to(self.device), data[1].to(self.device)
                optimizer.zero_grad()

                # Forward and backward passes
                outputs = self(inputs).squeeze()
                loss = criterion(outputs, labels.float())
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

                progress_bar.set_description(f"Loss: {running_loss / (i + 1):.6f}")

            print(
                f"Epoch [{epoch + 1}/{epochs}] - Loss: {running_loss / len(x_loader):.6f}"
            )

    def validate(self, x_loader):
        self.eval()
        correct = 0
        total = 0

        with torch.no_grad():  # Turn off gradients for validation, saves memory and computations
            progress_bar = tqdm(x_loader)
            for data in progress_bar:
                inputs, labels = data[0].to(self.device), data[1].to(self.device)
                outputs = self(inputs)

                # Apply sigmoid to convert logits to probabilities
                probabilities = torch.sigmoid(outputs).squeeze()

                # Convert probabilities to predicted class (0 or 1)
                predicted_labels = (
                    probabilities > 0.5
                ).float()  # Using 0.5 as the threshold

                total += labels.size(0)
                correct += (predicted_labels == labels).sum().item()

        print(
            f"Accuracy of the network on the validation set: {100 * correct / total:.2f}%"
        )

    def save(self, path: str):
        torch.save(self.state_dict(), path)

    def load(self, path: str):
        self.load_state_dict(torch.load(path))
