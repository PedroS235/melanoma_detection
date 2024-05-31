import cv2
import numpy as np
import melanoma_detection.params as params
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from PIL import Image


class SharpnessEnhancement:
    def __init__(self, sharpness_factor):
        self.sharpness_factor = sharpness_factor

    def __call__(self, img):
        return TF.adjust_sharpness(img, self.sharpness_factor)


class MelanomaMaskTransform:
    def __call__(self, img):
        cv2_img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

        # Apply Blur
        blur = cv2.GaussianBlur(cv2_img, (3, 3), 0)

        # Convert to grayscale
        gray_img = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)

        # Apply Otsu's thresholding
        _, mask = cv2.threshold(
            gray_img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )

        # Apply morphology for smoothness
        kernel = np.ones((2, 2), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=10)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=10)
        mask = mask.astype(bool)

        # Apply mask to original image, highliting melanoma area
        output_img = np.zeros_like(cv2_img)
        output_img[mask] = cv2_img[mask]

        # Convert the processed OpenCV image back to PIL format if needed
        final_img = Image.fromarray(cv2.cvtColor(output_img, cv2.COLOR_BGR2RGB))

        return final_img


TRANSFORM_TRAIN = transforms.Compose(
    [
        transforms.Resize((224, 224), antialias=True),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        SharpnessEnhancement(3),
        transforms.ToTensor(),
        transforms.Normalize(params.MEAN, params.STD),
    ]
)

TRANSFORM_VALIDATION = transforms.Compose(
    [
        transforms.Resize((224, 224)),  # Resize the image to 224x224 pixels
        SharpnessEnhancement(3),
        transforms.ToTensor(),
        transforms.Normalize(params.MEAN, params.STD),
    ]
)
