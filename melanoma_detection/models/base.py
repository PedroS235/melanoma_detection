import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from melanoma_detection.regularization import L1Regularization
from melanoma_detection.utils.metrics_utils import plot_metrics, compute_metrics
from torch.optim import Optimizer
from typing import Tuple, Dict

PKG_DIR = os.path.dirname(os.path.abspath(sys.argv[0]))


class BaseNetwork(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.l1_reg = L1Regularization()

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int,
        optimizer: Optimizer,
        criterion: nn.Module,
        early_stopping=None,
        verbose: bool = True,
        show_plot: bool = True,
    ) -> None:
        assert train_loader is not None, "Train data cannot be empty"
        self.train()
        train_losses = []
        val_losses = []
        train_accuracies = []
        val_accuracies = []
        val_metrics = []
        y_true = []
        y_pred = []

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
                loss = criterion(outputs, labels.float()) + self.l1_reg(
                    self.parameters()
                )
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
                true, pred, val_loss, val_accuracy, val_metrics = self.validate(
                    val_loader, criterion, False
                )
                val_losses.append(val_loss)
                val_accuracies.append(val_accuracy)
                y_true.append(true)
                y_pred.append(pred)
                if verbose:
                    print(
                        f"Epoch [{epoch + 1}/{epochs}] - \n"
                        f"Train Loss: {running_loss / len(train_loader):.6f} | "
                        f"Train Accuracy: {train_accuracy:.2f}%\n"
                        f"Val Loss: {val_loss:.6f} | "
                        f"Val Accuracy: {val_accuracy:.2f}%"
                    )

                # Early Stopping
                if early_stopping:
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

        if verbose and show_plot:
            y_true = torch.cat(y_true)
            y_pred = torch.cat(y_pred)
            plot_metrics(
                train_losses,
                val_losses,
                train_accuracies,
                val_accuracies,
                val_metrics,
                y_true,
                y_pred,
            )

    def validate(
        self, val_loader: DataLoader, criterion: nn.Module, verbose: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor, float, float, Dict[str, float]]:
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
                all_outputs.append(probabilities)

        accuracy = 100 * correct / total
        all_labels = torch.cat(all_labels)
        all_outputs = torch.cat(all_outputs)
        metrics = compute_metrics(all_labels, all_outputs)

        if verbose:
            print(
                f"Validation Loss: {running_loss / len(val_loader):.6f}, Accuracy: {accuracy:.2f}%"
            )

        return (
            all_labels,
            all_outputs,
            running_loss / len(val_loader),
            accuracy,
            metrics,
        )

    def save(self, path: str) -> None:
        torch.save(self.state_dict(), path)

    def load(self, path: str) -> None:
        self.load_state_dict(torch.load(path))
