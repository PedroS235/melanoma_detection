from melanoma_detection.utils.metrics_utils import plot_metrics_comparison


ours = {
    "accuracy": 0.93,
    "precision": 0.93,
    "recall": 0.92,
    "f1_score": 0.92,
    "auc": 0.93,
}

other = {
    "accuracy": 0.95,
    "precision": 0.93,
    "recall": 0.96,
    "f1_score": 0.95,
    "auc": 0.95,
}

plot_metrics_comparison(ours, other)
