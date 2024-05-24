import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from melanoma_detection.early_stopping import EarlyStopping
from melanoma_detection.metrics_utils import plot_metrics, compute_metrics

PKG_DIR = os.path.dirname(os.path.abspath(sys.argv[0]))


class BaseNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

    def forward(self, x):
        raise NotImplementedError

    def fit(
        self,
        train_loader,
        val_loader,
        epochs,
        optimizer,
        criterion,
        verbose=True,
        early_stopping_patience=None,
    ):
        assert train_loader is not None, "Train data cannot be empty"
        self.train()
        train_losses = []
        val_losses = []
        train_accuracies = []
        val_accuracies = []

        if early_stopping_patience:
            early_stopping = EarlyStopping(patience=early_stopping_patience)

        for epoch in range(epochs):
            running_loss = 0.0
            correct = 0
            total = 0
            progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}")

            for i, data in enumerate(progress_bar):
                inputs, labels = data[0].to(self.device), data[1].to(self.device)
                optimizer.zero_grad()

                # Forward and backward passes
                outputs = self(inputs).squeeze()
                l1_lambda = 1e-5
                l1_norm = sum(p.abs().sum() for p in self.parameters())
                loss = criterion(outputs, labels.float()) + l1_lambda * l1_norm
                # loss = criterion(outputs, labels.float())
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                progress_bar.set_description(f"Loss: {running_loss / (i + 1):.6f}")

                # Compute training accuracy
                probabilities = torch.sigmoid(outputs).squeeze()
                predicted_labels = (probabilities > 0.5).float()
                correct += (predicted_labels == labels).sum().item()
                total += labels.size(0)

            train_losses.append(running_loss / len(train_loader))
            train_accuracy = 100 * correct / total
            train_accuracies.append(train_accuracy)

            if val_loader:
                val_loss, val_accuracy, val_metrics = self.validate(
                    val_loader, criterion, verbose
                )
                val_losses.append(val_loss)
                val_accuracies.append(val_accuracy)
                if verbose:
                    print(
                        f"Epoch [{epoch + 1}/{epochs}] - "
                        f"Train Loss: {running_loss / len(train_loader):.6f}, "
                        f"Train Accuracy: {train_accuracy:.2f}%, "
                        f"Val Loss: {val_loss:.6f}, "
                        f"Val Accuracy: {val_accuracy:.2f}%, "
                        f"Val Metrics: {val_metrics}"
                    )

                # Early Stopping
                if early_stopping_patience:
                    early_stopping(val_loss, self)
                    if early_stopping.early_stop:
                        print("Early stopping triggered.")
                        early_stopping.load_checkpoint(self)
                        break
            else:
                if verbose:
                    print(
                        f"Epoch [{epoch + 1}/{epochs}] - Loss: {running_loss / len(train_loader):.6f}"
                    )

        if verbose:
            plot_metrics(
                train_losses, val_losses, train_accuracies, val_accuracies, val_metrics
            )

    def validate(self, val_loader, criterion, verbose=True):
        self.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        all_labels = []
        all_outputs = []

        with torch.no_grad():
            progress_bar = tqdm(val_loader, desc="Validation")
            for data in progress_bar:
                inputs, labels = data[0].to(self.device), data[1].to(self.device)
                outputs = self(inputs).squeeze()

                loss = criterion(outputs, labels.float())
                running_loss += loss.item()

                probabilities = torch.sigmoid(outputs).squeeze()
                predicted_labels = (probabilities > 0.5).float()
                correct += (predicted_labels == labels).sum().item()
                total += labels.size(0)

                all_labels.append(labels)
                all_outputs.append(outputs)

        accuracy = 100 * correct / total
        all_labels = torch.cat(all_labels)
        all_outputs = torch.cat(all_outputs)
        metrics = compute_metrics(all_labels, all_outputs)

        if verbose:
            print(
                f"Validation Loss: {running_loss / len(val_loader):.6f}, Accuracy: {accuracy:.2f}%"
            )

        return running_loss / len(val_loader), accuracy, metrics

    def save(self, path: str):
        torch.save(self.state_dict(), path)

    def load(self, path: str):
        self.load_state_dict(torch.load(path))
