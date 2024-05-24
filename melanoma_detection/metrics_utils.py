import matplotlib.pyplot as plt
import torch
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)


def compute_metrics(y_true, y_pred):
    y_pred = torch.sigmoid(y_pred).cpu().numpy() > 0.5
    y_true = y_true.cpu().numpy()

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_pred)

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "auc": auc,
    }


def plot_metrics(train_losses, val_losses, train_accuracies, val_accuracies, metrics):
    epochs = range(1, len(train_losses) + 1)

    plt.figure(figsize=(14, 10))

    # Plot losses
    plt.subplot(2, 1, 1)
    plt.plot(epochs, train_losses, "bo-", label="Training loss")
    plt.plot(epochs, val_losses, "ro-", label="Validation loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Losses")

    # Plot accuracies
    plt.subplot(2, 1, 2)
    plt.plot(epochs, train_accuracies, "bo-", label="Training accuracy")
    plt.plot(epochs, val_accuracies, "ro-", label="Validation accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.title("Accuracies")

    plt.tight_layout()
    plt.show()

    # Print other metrics
    print("Validation Metrics:")
    for metric, values in metrics.items():
        print(f"{metric}: {values:.4f}")
