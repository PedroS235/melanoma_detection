import sys
import torch
from melanoma_detection.preprocess_dataset import (
    create_test_dataset,
    MelanomaDataset,
)
from torch.utils.data import DataLoader
from melanoma_detection.models import ResNet
from melanoma_detection.utils.metrics_utils import generate_plots_paper
import melanoma_detection.params as params
from melanoma_detection.transforms import TRANSFORM_VALIDATION


test_loader = DataLoader(
    MelanomaDataset(create_test_dataset(), transform=TRANSFORM_VALIDATION),
    params.BATCH_SIZE_RESNET,
    shuffle=False,
    num_workers=5,
)

criterion = torch.nn.BCEWithLogitsLoss()

net = ResNet()

PATH = sys.argv[1]

net.load(PATH)

y_true, y_pred, _, _, metrics = net.validate(test_loader, criterion)

for key, value in metrics.items():
    print(f"    {key}: {value}")


generate_plots_paper(y_true, y_pred)
