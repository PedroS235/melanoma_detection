import os
import numpy as np
from PIL import Image
from glob import glob


# Function to collect all image paths from a directory
def collect_image_paths(directory):
    image_paths = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(("jpg", "jpeg", "png", "bmp", "tiff")):
                image_paths.append(os.path.join(root, file))
    return image_paths


# Collect all image paths from train and test directories
train_image_paths = collect_image_paths("dataset/train")
test_image_paths = collect_image_paths("dataset/test")
all_image_paths = train_image_paths + test_image_paths

# Initialize sums and squared sums
mean_r = 0
mean_g = 0
mean_b = 0
squared_mean_r = 0
squared_mean_g = 0
squared_mean_b = 0

n = 0  # total number of pixels

for path in all_image_paths:
    image = Image.open(path).convert("RGB")
    np_image = np.array(image) / 255.0  # Normalize to range [0, 1]

    mean_r += np.sum(np_image[:, :, 0])
    mean_g += np.sum(np_image[:, :, 1])
    mean_b += np.sum(np_image[:, :, 2])

    squared_mean_r += np.sum(np_image[:, :, 0] ** 2)
    squared_mean_g += np.sum(np_image[:, :, 1] ** 2)
    squared_mean_b += np.sum(np_image[:, :, 2] ** 2)

    n += np_image.shape[0] * np_image.shape[1]

# Calculate means
mean_r /= n
mean_g /= n
mean_b /= n

# Calculate standard deviations
std_r = np.sqrt(squared_mean_r / n - mean_r**2)
std_g = np.sqrt(squared_mean_g / n - mean_g**2)
std_b = np.sqrt(squared_mean_b / n - mean_b**2)

# Print the calculated values
print(f"Mean: [{mean_r:.3f}, {mean_g:.3f}, {mean_b:.3f}]")
print(f"Std: [{std_r:.3f}, {std_g:.3f}, {std_b:.3f}]")
