import numpy as np
from PIL import Image


def simulate_sensor(
    y_high_res, width=100, height=20, bit_depth=16, orientation="horizontal"
):
    """
    Simulates the full capture process of a digital sensor.

    Args:
        y_high_res: The high-resolution 'analog' input signal.
        width: The number of horizontal pixels (Spatial Resolution).
        height: The number of vertical bands.
        bit_depth: The radiometric resolution (e.g., 16-bit = 65536 levels).
        orientation (str): 'horizontal' or 'vertical' orientation of the signal.
    Returns:
        image_data (array): The simulated quantized 2D image.
        scaling_factor (float): The factor used for normalization.
    """

    # --- Step 1: Spatial Discretization (The "Pixelation" Step) ---
    x_original = np.linspace(0, 1, len(y_high_res))
    x_sensor = np.linspace(0, 1, width)

    # Resample the signal onto the sensor's pixel grid (Linear Interpolation)
    y_spatial_discrete = np.interp(x_sensor, x_original, y_high_res)

    # --- Step 2: Amplitude Quantization (The "Bit-Depth" Step) ---
    max_physical_signal = np.max(y_high_res)
    if max_physical_signal == 0:
        max_physical_signal = 1.0

    # Calculate the max integer value (e.g., 65535)
    max_digital_value = (2**bit_depth) - 1

    # Normalize -> Scale -> Round -> Cast
    y_normalized = y_spatial_discrete / max_physical_signal
    y_quantized = np.round(y_normalized * max_digital_value).astype(
        np.uint16 if bit_depth > 8 else np.uint8
    )

    # --- Step 3: Sensor Geometry (Banding) ---
    # Replicate the 1D array vertically to create the 2D "spectroscopy band" look
    image_data = np.tile(y_quantized, (height, 1))

    if orientation == "vertical":
        image_data = image_data.T

    return image_data, max_physical_signal


def save_signal_as_tif(image_data, filename):
    """
    Saves image data as a TIFF file.
    """
    img = Image.fromarray(image_data)
    img.save(filename)
