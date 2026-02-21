import numpy as np
from scipy.integrate import simpson
from scipy.interpolate import UnivariateSpline


def calculate_auc_trapezoidal(x, y):
    """
    Calculates Area Under Curve using Trapezoidal rule.
    """
    return np.trapezoid(y, x)


def calculate_auc_simpson(x, y):
    """
    Calculates Area Under Curve using Simpson's rule.
    """
    return simpson(y, x=x)


def calculate_auc_spline(x, y):
    """
    Calculates Area Under Curve using Spline integration.
    """
    spline = UnivariateSpline(x, y, s=0)
    return spline.integral(np.min(x), np.max(x))
