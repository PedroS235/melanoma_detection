import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import matplotlib.pyplot as plt

PKG_DIR = os.path.dirname(os.path.abspath(sys.argv[0]))


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
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

    def fit(self, train_loader, val_loader, epochs, optimizer, criterion):
        self.train()
        train_losses = []
        val_accuracies = []

        for epoch in range(epochs):
            running_loss = 0.0
            progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}")

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

            train_losses.append(running_loss / len(train_loader))
            val_accuracy = self.validate(val_loader)
            val_accuracies.append(val_accuracy)

            print(
                f"Epoch [{epoch + 1}/{epochs}] - Loss: {running_loss / len(train_loader):.6f}, Val Accuracy: {val_accuracy:.2f}%"
            )

        self.plot_metrics(train_losses, val_accuracies)

    def validate(self, val_loader):
        self.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            progress_bar = tqdm(val_loader, desc="Validation")
            for data in progress_bar:
                inputs, labels = data[0].to(self.device), data[1].to(self.device)
                outputs = self(inputs)

                probabilities = torch.sigmoid(outputs).squeeze()
                predicted_labels = (probabilities > 0.5).float()

                total += labels.size(0)
                correct += (predicted_labels == labels).sum().item()

        accuracy = 100 * correct / total
        print(f"Accuracy of the network on the validation set: {accuracy:.2f}%")
        return accuracy

    def plot_metrics(self, train_losses, val_accuracies):
        epochs = range(1, len(train_losses) + 1)

        plt.figure(figsize=(12, 5))

        plt.subplot(1, 2, 1)
        plt.plot(epochs, train_losses, label="Training Loss")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.title("Training Loss over Epochs")
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(epochs, val_accuracies, label="Validation Accuracy")
        plt.xlabel("Epochs")
        plt.ylabel("Accuracy (%)")
        plt.title("Validation Accuracy over Epochs")
        plt.legend()

        plt.tight_layout()
        plt.show()

    def save(self, path: str):
        torch.save(self.state_dict(), path)

    def load(self, path: str):
        self.load_state_dict(torch.load(path))


import torch
import torchvision.models as models
from torch import nn
from tqdm import tqdm


class ResNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

        num_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_features, 1)

        self.resnet.to(self.device)

    def forward(self, x):
        return self.resnet(x)

    def fit(self, train_loader, val_loader, epochs, optimizer, criterion):
        self.train()
        train_losses = []
        val_accuracies = []

        for epoch in range(epochs):
            running_loss = 0.0
            progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}")

            for i, data in enumerate(progress_bar):
                inputs, labels = data[0].to(self.device), data[1].to(self.device)
                optimizer.zero_grad()

                outputs = self(inputs).squeeze()
                loss = criterion(outputs, labels.float())
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                progress_bar.set_description(f"Loss: {running_loss / (i + 1):.6f}")

            train_losses.append(running_loss / len(train_loader))
            val_accuracy = self.validate(val_loader)
            val_accuracies.append(val_accuracy)

            print(
                f"Epoch [{epoch + 1}/{epochs}] - Loss: {running_loss / len(train_loader):.6f}, Val Accuracy: {val_accuracy:.2f}%"
            )

        self.plot_metrics(train_losses, val_accuracies)

    def validate(self, val_loader):
        self.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            progress_bar = tqdm(val_loader, desc="Validation")
            for data in progress_bar:
                inputs, labels = data[0].to(self.device), data[1].to(self.device)
                outputs = self(inputs)

                probabilities = torch.sigmoid(outputs).squeeze()
                predicted_labels = (probabilities > 0.5).float()

                total += labels.size(0)
                correct += (predicted_labels == labels).sum().item()

        accuracy = 100 * correct / total
        print(f"Accuracy of the network on the validation set: {accuracy:.2f}%")
        return accuracy

    def plot_metrics(self, train_losses, val_accuracies):
        epochs = range(1, len(train_losses) + 1)

        plt.figure(figsize=(12, 5))

        plt.subplot(1, 2, 1)
        plt.plot(epochs, train_losses, label="Training Loss")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.title("Training Loss over Epochs")
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(epochs, val_accuracies, label="Validation Accuracy")
        plt.xlabel("Epochs")
        plt.ylabel("Accuracy (%)")
        plt.title("Validation Accuracy over Epochs")
        plt.legend()

        plt.tight_layout()
        plt.show()

    def save(self, path: str):
        torch.save(self.state_dict(), path)

    def load(self, path: str):
        self.load_state_dict(torch.load(path))
