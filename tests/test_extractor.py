import numpy as np
import pytest
from PIL import Image

from src.processing.extractor import extract_signal


def test_extract_signal_horizontal(tmp_path):
    """Test extracting signal from a fake horizontal image."""
    img_path = tmp_path / "test_horizontal.tif"

    # Create a 100x10 dummy image with gradient (horizontal means 10 rows, 100 cols)
    # The image is wider than it is tall, so auto-orientation = 'horizontal'
    # 'horizontal' extracts by averaging along axis=0 (across rows)
    data = np.zeros((10, 100), dtype=np.uint8)
    for i in range(100):
        data[:, i] = min(i * 2, 255)

    img = Image.fromarray(data)
    img.save(img_path)

    x, y = extract_signal(str(img_path))

    assert len(x) == 100
    assert len(y) == 100
    # At index 50, value should be 100 / 255
    assert np.isclose(y[50], 100 / 255)


def test_extract_signal_vertical(tmp_path):
    """Test extracting signal from a fake vertical image."""
    img_path = tmp_path / "test_vertical.tif"

    # Create a 100x10 dummy image (100 rows, 10 cols)
    # The image is taller than it is wide, so auto-orientation = 'vertical'
    # 'vertical' extracts by averaging along axis=1 (across columns)
    data = np.zeros((100, 10), dtype=np.uint8)
    for i in range(100):
        data[i, :] = min(i * 2, 255)

    img = Image.fromarray(data)
    img.save(img_path)

    x, y = extract_signal(str(img_path))

    assert len(x) == 100
    assert len(y) == 100
    # At index 25, value should be 50 / 255
    assert np.isclose(y[25], 50 / 255)
