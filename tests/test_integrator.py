import numpy as np

from src.processing.integrator import calculate_auc_simpson, calculate_auc_spline


def test_calculate_auc_simpson_basic():
    """Test Simpson's rule integration on a known function (y = 2x)."""
    x = np.array([0, 1, 2, 3, 4])
    y = 2 * x
    # Area of triangle: 0.5 * base * height = 0.5 * 4 * 8 = 16
    auc = calculate_auc_simpson(x, y)
    assert np.isclose(auc, 16.0)


def test_calculate_auc_spline_basic():
    """Test Spline integration on a known function."""
    x = np.array([0, 1, 2, 3, 4])
    y = np.array([0, 1, 4, 9, 16])  # y = x^2
    # Analytical integral of x^2 from 0 to 4 is (x^3)/3 evaluated at 4 = 64/3 = 21.333...
    auc = calculate_auc_spline(x, y)
    assert np.isclose(auc, 64 / 3, rtol=1e-2)


def test_calculate_auc_zero():
    """Test integration when y is all zeros."""
    x = np.array([0, 1, 2, 3, 4])
    y = np.zeros_like(x)
    assert calculate_auc_simpson(x, y) == 0.0
    assert calculate_auc_spline(x, y) == 0.0
