"""This module contains functions to create a dataset from the images in the training and testing
directories. The dataset is a list of dictionaries, where each dictionary contains the path to an
image and the label of the image (0 for benign, 1 for malignant). The module also contains a custom
dataset class for the melanoma images. The dataset is a list of dictionaries, where each dictionary
contains the path to an image and the label of the image (0 for benign, 1 for malignant).
"""

import os
from torch.utils.data import Dataset
import torch
from PIL import Image
import numpy as np

# The images are stored in the following directories: Assumes data is located in ..data
# Each image is 224x224 pixels and has 3 channels (RGB)
DATA_DIR = "./dataset"
DATA_TRAIN_DIR = os.path.join(DATA_DIR, "train")
DATA_TEST_DIR = os.path.join(DATA_DIR, "test")
TRAIN_BENIGN_DIR = os.path.join(DATA_TRAIN_DIR, "Benign")
TRAIN_MALIGNANT_DIR = os.path.join(DATA_TRAIN_DIR, "Malignant")
TEST_BENIGN_DIR = os.path.join(DATA_TEST_DIR, "Benign")
TEST_MALIGNANT_DIR = os.path.join(DATA_TEST_DIR, "Malignant")


class MelanomaDataset(Dataset):
    """A custom dataset class for the melanoma images. The dataset is a list of dictionaries,
    where each dictionary contains the path to an image and the label of the image (0 for benign,
    1 for malignant).

    Args:
        dataset: a list of dictionaries, where each dictionary contains the path to an image
        and the label of the image (0 for benign, 1 for malignant).
        transform: a transform to apply to the images.

    Returns:
        sample: a dictionary containing the image and label of the image.
    """

    def __init__(self, dataset, transform=None, pipeline=None):
        self.dataset = dataset
        self.transform = transform
        self.pipeline = pipeline

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        label = self.dataset[idx][1]
        image = Image.open(self.dataset[idx][0])
        image = self.pipeline.process(image) if self.pipeline else image
        image = self.transform(image) if self.transform else image
        label = torch.tensor(int(label), dtype=torch.float)
        return image, label


def create_train_dataset():
    """Create a dataset from the images in the training directory. The dataset
    is a list of dictionaries, where each dictionary contains the path to an image
    and the label of the image (0 for benign, 1 for malignant).

    Returns:
        dataset: a list of dictionaries, where each dictionary contains the path to an image
        and the label of the image (0 for benign, 1 for malignant).
    """
    dataset = []
    for img in os.listdir(TRAIN_BENIGN_DIR):
        dataset.append((os.path.join(TRAIN_BENIGN_DIR, img), 0))

    for img in os.listdir(TRAIN_MALIGNANT_DIR):
        dataset.append((os.path.join(TRAIN_MALIGNANT_DIR, img), 1))

    dataset = np.array(dataset)
    # np.random.shuffle(dataset)
    return dataset


def create_test_dataset():
    """Create a dataset from the images in the testing directory. The dataset
    is a list of dictionaries, where each dictionary contains the path to an image
    and the label of the image (0 for benign, 1 for malignant).

    Returns:
        dataset: a list of dictionaries, where each dictionary contains the path to an image
        and the label of the image (0 for benign, 1 for malignant).
    """
    dataset = []
    for img in os.listdir(TEST_BENIGN_DIR):
        dataset.append((os.path.join(TEST_BENIGN_DIR, img), 0))

    for img in os.listdir(TEST_MALIGNANT_DIR):
        dataset.append((os.path.join(TEST_MALIGNANT_DIR, img), 1))

    dataset = np.array(dataset)
    # np.random.shuffle(dataset)
    return dataset
