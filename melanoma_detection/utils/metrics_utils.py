from typing import Dict
import matplotlib.pyplot as plt
import torch
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    confusion_matrix,
    precision_recall_curve,
)
import pandas as pd


def generate_plots_paper(y_true: torch.Tensor, y_pred: torch.Tensor) -> None:
    y_pred_prob = y_pred.cpu().numpy()
    y_pred = y_pred_prob > 0.5
    y_true = y_true.cpu().numpy() > 0.5

    # Set the seaborn color palette
    sns.set_palette("hls")

    # Compute metrics
    metrics = compute_metrics(torch.tensor(y_true), torch.tensor(y_pred))

    # 1. AUC-ROC Curve
    fpr, tpr, _ = roc_curve(y_true, y_pred_prob)
    plt.figure(figsize=(10, 6))
    plt.plot(fpr, tpr, color="blue", lw=2, label=f'AUC = {metrics["auc"]:.2f}')
    plt.plot([0, 1], [0, 1], color="grey", lw=2, linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic (ROC) Curve")
    plt.legend(loc="lower right")
    plt.show()

    # Plot Precision-Recall Curve
    plt.figure(figsize=(10, 6))
    precision, recall, _ = precision_recall_curve(y_true, y_pred_prob)
    plt.plot(recall, precision, marker=".", label="Precision-Recall Curve")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend()
    plt.show()

    # 2. Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix")
    plt.show()

    # 3. Metrics Plot
    plt.figure(figsize=(10, 6))
    metric_names = ["accuracy", "precision", "recall", "f1_score", "auc"]
    metric_values = [metrics[name] for name in metric_names]
    # Create a DataFrame for the metrics
    data = pd.DataFrame({"Metric": metric_names, "Value": metric_values})
    # Use hue and set legend to False
    sns.barplot(
        x="Metric",
        y="Value",
        hue="Metric",
        data=data,
        palette="hls",
        dodge=False,
        legend=False,
    )
    plt.ylim(0, 1)
    for i, v in enumerate(metric_values):
        plt.text(i, v + 0.02, f"{v:.2f}", ha="center", va="bottom")
    plt.title("Model Performance Metrics")
    plt.show()


def compute_metrics(y_true: torch.Tensor, y_pred: torch.Tensor) -> Dict[str, float]:
    y_pred = y_pred.cpu().numpy() > 0.5
    y_true = y_true.cpu().numpy() > 0.5
    # y_pred = torch.sigmoid(y_pred).cpu().numpy() > 0.5
    # y_true = y_true.cpu().numpy()

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    auc = float(roc_auc_score(y_true, y_pred))

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "auc": auc,
    }


def plot_metrics(
    train_losses: torch.Tensor,
    val_losses: torch.Tensor,
    train_accuracies: torch.Tensor,
    val_accuracies: torch.Tensor,
    metrics: Dict[str, float],
    y_true: torch.Tensor,
    y_pred_prob: torch.Tensor,
) -> None:
    sns.set_palette("hls")
    epochs = range(1, len(train_losses) + 1)

    # Plot losses
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_losses, "bo-", label="Training loss")
    plt.plot(epochs, val_losses, "ro-", label="Validation loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Losses")
    plt.show()

    # Plot accuracies
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_accuracies, "bo-", label="Training accuracy")
    plt.plot(epochs, val_accuracies, "ro-", label="Validation accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.title("Accuracies")
    plt.show()

    # Plot Precision-Recall Curve
    plt.figure(figsize=(10, 6))
    y_true_np = y_true.cpu().numpy()
    y_pred_prob_np = y_pred_prob.cpu().numpy()
    precision, recall, _ = precision_recall_curve(y_true_np, y_pred_prob_np)
    plt.plot(recall, precision, marker=".", label="Precision-Recall Curve")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend()
    plt.show()

    # Plot Confusion Matrix
    plt.figure(figsize=(10, 6))
    y_pred_np = y_pred_prob_np > 0.5
    cm = confusion_matrix(y_true_np, y_pred_np)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix")
    plt.show()

    # Plot ROC Curve
    fpr, tpr, _ = roc_curve(y_true_np, y_pred_prob_np)
    plt.figure(figsize=(10, 6))
    plt.plot(fpr, tpr, color="blue", lw=2, label=f'AUC = {metrics["auc"]:.2f}')
    plt.plot([0, 1], [0, 1], color="grey", lw=2, linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic (ROC) Curve")
    plt.legend(loc="lower right")
    plt.show()

    # Print other metrics
    print("Validation Metrics:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")


def plot_metrics_comparison(ours, other):
    # Define the metrics data
    metrics = {
        "Model": [
            "Ours",
            "Ours",
            "Ours",
            "Ours",
            "Ours",
            "ResNet50",
            "ResNet50",
            "ResNet50",
            "ResNet50",
            "ResNet50",
        ],
        "Metric": [
            "Accuracy",
            "Precision",
            "Recall",
            "F1 Score",
            "AUC",
            "Accuracy",
            "Precision",
            "Recall",
            "F1 Score",
            "AUC",
        ],
        "Value": [
            ours["accuracy"],
            ours["precision"],
            ours["recall"],
            ours["f1_score"],
            ours["auc"],
            other["accuracy"],
            other["precision"],
            other["recall"],
            other["f1_score"],
            other["auc"],
        ],
    }

    # Convert the dictionary to a DataFrame
    df = pd.DataFrame(metrics)

    plt.figure(figsize=(10, 6))
    ax = sns.barplot(x="Metric", y="Value", hue="Model", data=df, palette="hls")

    # Add the numbers on top of each bar
    for p in ax.patches:
        ax.annotate(
            format(p.get_height(), ".2f"),
            (p.get_x() + p.get_width() / 2.0, p.get_height()),
            ha="center",
            va="center",
            xytext=(0, 9),
            textcoords="offset points",
        )

    plt.legend(title="Model", loc="lower left")

    plt.title("Model Performance Comparison")
    plt.ylim(0, 1)
    plt.show()
