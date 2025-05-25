import numpy as np
import cv2


def create_thumbnail(image_array: np.ndarray, size: tuple = (256, 256)) -> np.ndarray:
    """
    Resamples and crops a NumPy image array to a target square size
    using bicubic interpolation.
    """
    if image_array.ndim != 3 or image_array.shape[2] != 3:
        raise ValueError("Input image array must be a 3D array with 3 channels (RGB).")

    # Get original dimensions
    orig_h, orig_w = image_array.shape[:2]
    target_h, target_w = size

    # Determine the dimension of the central square to crop
    min_dim = min(orig_h, orig_w)

    # Calculate cropping coordinates for the central square
    # Integer division ensures we get whole pixels
    start_row = (orig_h - min_dim) // 2
    end_row = start_row + min_dim
    start_col = (orig_w - min_dim) // 2
    end_col = start_col + min_dim

    # Crop the central square
    cropped_image = image_array[start_row:end_row, start_col:end_col]

    # Resize the cropped square image to the target size using bicubic interpolation.
    resized_image = cv2.resize(cropped_image, (target_w, target_h), interpolation=cv2.INTER_CUBIC)

    return resized_image


def blend_images(img1_arr, img2_arr, percentage):
    """Blends two images based on a percentage [0...100]."""
    if img1_arr.shape != img2_arr.shape:
        # If there is an error return a blank image.
        return np.zeros_like(img1_arr)

    # Convert percentage to a 0.0-1.0 alpha value for img2
    # Alpha 0.0 means all img1, alpha 1.0 means all img2
    alpha = percentage / 100.0

    img1_float = img1_arr.astype(np.float32)
    img2_float = img2_arr.astype(np.float32)

    # Perform the blending calculation
    blended_img = (1.0 - alpha) * img1_float + alpha * img2_float
    blended_img = np.clip(blended_img, 0.0, 1.0)

    return blended_img
