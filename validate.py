import sys
from melanoma_detection.preprocess_dataset import (
    create_test_dataset,
    MelanomaDataset,
)
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from melanoma_detection.network import Net, ResNet
from melanoma_detection.img_utils import ImagePreprocessingPipeline


BATCH_SIZE = 4
EPOCHS = 2

img_pipeline = ImagePreprocessingPipeline(1, 1, False)

transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
)  # mean, Standard diviation,  channels

test_loader = DataLoader(
    MelanomaDataset(create_test_dataset(), transform=transform, pipeline=img_pipeline),
    BATCH_SIZE,
    shuffle=False,
    num_workers=5,
)

# net = Net()
net = ResNet()

PATH = sys.argv[1]

net.load(PATH)
net.validate(test_loader)
