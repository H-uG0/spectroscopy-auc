import numpy as np
from PIL import Image


def extract_signal(filename, x_scaling=1.0, y_scaling=1.0, orientation=None):
    """
    Extracts the 1D signal from a simulated 2D TIFF image.

    Args:
        filename (str): Path to the TIFF file.
        x_scaling (float): Physical scale per pixel.
        y_scaling (float): Physical scale for intensity (quantization reverse).
        orientation (str, optional): 'horizontal' or 'vertical'. If None, auto-detected.
    Returns:
        x (array): The physical coordinate array.
        y_extracted (array): The reconstructed physical signal.
    """
    # Open the image
    with Image.open(filename) as img:
        img_arr = np.array(img)

    # Auto-detection of orientation based on aspect ratio
    if orientation is None:
        if img_arr.shape[0] > img_arr.shape[1]:
            orientation = "vertical"
        else:
            orientation = "horizontal"

    # Extract 1D signal by averaging across the perpendicular axis
    if orientation == "vertical":
        y_quantized = np.mean(img_arr, axis=1)
    else:
        y_quantized = np.mean(img_arr, axis=0)

    # Reconstruct the signal by reversing quantization
    if img_arr.dtype == np.uint16:
        max_digital_value = 65535
    elif img_arr.dtype == np.uint8:
        max_digital_value = 255
    else:
        raise ValueError(f"Unsupported image bit depth: {img_arr.dtype}")

    y_normalized = y_quantized / max_digital_value
    y_reconstructed = y_normalized * y_scaling

    x = np.arange(len(y_reconstructed)) * x_scaling

    return x, y_reconstructed
