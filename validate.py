import sys
import torch
from melanoma_detection.preprocess_dataset import (
    create_test_dataset,
    MelanomaDataset,
)
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from melanoma_detection.transforms import AdjustSharpness
from melanoma_detection.models.melanoma import MelanomaNetwork, MelanomaNetworkV2
from melanoma_detection.utils.metrics_utils import generate_plots_paper

# Set the seed for reproducibility
# seed = 42
# torch.manual_seed(seed)


BATCH_SIZE = 42


# Imagenet normalization values
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

transform_validation = transforms.Compose(
    [
        transforms.Resize((224, 224)),  # Resize the image to 224x224 pixels
        AdjustSharpness(3),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ]
)

test_loader = DataLoader(
    MelanomaDataset(create_test_dataset(), transform=transform_validation),
    BATCH_SIZE,
    shuffle=False,
    num_workers=5,
)

criterion = torch.nn.BCEWithLogitsLoss()

net = MelanomaNetworkV2()

PATH = sys.argv[1]

net.load(PATH)
y_true, y_pred, _, _, metrics = net.validate(test_loader, criterion)

for key, value in metrics.items():
    print(f"    {key}: {value}")


generate_plots_paper(y_true, y_pred)
