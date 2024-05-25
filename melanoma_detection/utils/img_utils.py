from PIL import Image, ImageEnhance
import matplotlib.pyplot as plt
import cv2
import numpy as np


class ImagePreprocessingPipeline:
    def __init__(
        self,
        contrast_factor: float = 2.0,
        sharpness_factor: float = 10.0,
        apply_mask=False,
    ):
        """Constructor of ImagePreprocessingPipeline

        Args:
            contrast_factor(float): the contrast factor to be applied
            sharpness_factor(float): the sharpness factor to be applied
        """
        self.contrast_factor = contrast_factor
        self.sharpness_factor = sharpness_factor
        self.apply_mask = apply_mask

    def process(self, img: Image.Image) -> Image.Image:
        """Add contrast and sharpness and masks the skin around the melanoma

        Args:
            img(PIL.Image): image to be processed
            mask_res(int): Resolution that the masking show apply.

        Returns (PIL.Image):
            processed image
        """

        enhanced_img = img.copy()

        # Apply Contrast
        constrast_enhancer = ImageEnhance.Contrast(enhanced_img)
        enhanced_img = constrast_enhancer.enhance(self.contrast_factor)

        # Apply Sharpness
        sharpness_enhancer = ImageEnhance.Sharpness(enhanced_img)
        enhanced_img = sharpness_enhancer.enhance(self.sharpness_factor)

        if self.apply_mask:
            enhanced_img = self.__mask_img(enhanced_img, 4)

        return enhanced_img

    def __mask_img(self, img: Image.Image, res: int) -> Image.Image:
        # Convert to OpenCV format
        cv2_img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

        # Apply Blur
        blur = cv2.GaussianBlur(cv2_img, (3, 3), 0)

        # Convert to grayscale
        gray_img = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)

        # Apply Otsu's thresholding
        _, mask = cv2.threshold(
            gray_img, 100, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )

        # Apply morphology for smoothness
        kernel = np.ones((2, 2), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=res)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=res)
        mask = mask.astype(bool)

        # Apply mask to original image, highliting melanoma area
        output_img = np.zeros_like(cv2_img)
        output_img[mask] = cv2_img[mask]

        # Convert the processed OpenCV image back to PIL format if needed
        final_img = Image.fromarray(cv2.cvtColor(output_img, cv2.COLOR_BGR2RGB))

        return final_img

def plot_image(img):
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


def main():

    benign_img = Image.open("./data/test/Benign/6318.jpg")
    malignant_img = Image.open("./data/test/Malignant/5605.jpg")

    pipeline = ImagePreprocessingPipeline(2, 10)

    benign_preprocessed = pipeline.process(benign_img)
    malignant_preprocessed = pipeline.process(malignant_img)

    # Display Images on a plot
    _, ax = plt.subplots(2, 2, figsize=(10, 5))

    ax[0][0].imshow(benign_img)
    ax[0][0].set_title("Benign Image")
    ax[0][0].axis("off")  # Turn off axis numbering and ticks

    ax[0][1].imshow(benign_preprocessed)
    ax[0][1].set_title("Processed Benign Image")
    ax[0][1].axis("off")  # Turn off axis numbering and ticks

    ax[1][0].imshow(malignant_img)
    ax[1][0].set_title("Malignant Image")
    ax[1][0].axis("off")  # Turn off axis numbering and ticks

    ax[1][1].imshow(malignant_preprocessed)
    ax[1][1].set_title("Processed Malignant Image")
    ax[1][1].axis("off")  # Turn off axis numbering and ticks

    plt.show()


if __name__ == "__main__":
    main()
