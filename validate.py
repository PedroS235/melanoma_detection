import sys
from melanoma_detection.preprocess_dataset import (
    create_test_dataset,
    MelanomaDataset,
)
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from melanoma_detection.network import Net, ResNet
from melanoma_detection.img_utils import ImagePreprocessingPipeline

# Imagenet normalization values
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

BATCH_SIZE = 4
EPOCHS = 2

img_pipeline = ImagePreprocessingPipeline(2, 5, False)

transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize(mean, std)]
)  # mean, Standard diviation,  channels

test_loader = DataLoader(
    MelanomaDataset(create_test_dataset(), transform=transform, pipeline=img_pipeline),
    BATCH_SIZE,
    shuffle=False,
    num_workers=5,
)

net = Net()
# net = ResNet()

PATH = sys.argv[1]

net.load(PATH)
net.validate(test_loader)
