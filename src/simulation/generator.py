import numpy as np


def gaussian(x, amp, mu, sigma):
    """
    Returns a single Gaussian spike.
    Formula: A * exp( - (x - mu)^2 / (2 * sigma^2) )
    """
    return amp * np.exp(-1.0 * (x - mu) ** 2 / (2 * sigma**2))


def gaussian_integral(amp, sigma):
    """
    Analytical integral of a Gaussian over -inf to +inf.
    Formula: A * sigma * sqrt(2 * pi)
    """
    return amp * sigma * np.sqrt(2 * np.pi)


def lorentzian(x, amp, x0, gamma):
    """
    Returns a single Lorentzian spike.
    Formula: A * (gamma^2 / ((x - x0)^2 + gamma^2))
    """
    return amp * (gamma**2 / ((x - x0) ** 2 + gamma**2))


def lorentzian_integral(amp, gamma):
    """
    Analytical integral of a Lorentzian over -inf to +inf.
    Formula: A * gamma * pi
    """
    return amp * gamma * np.pi


def generate_ground_truth(n_points=1000, curve_type="gaussian", num_peaks=4, seed=None):
    """
    Generates a complex function composed of multiple random spikes.
    Args:
        n_points (int): Number of points in the generated signal.
        curve_type (str): 'gaussian', 'lorentzian', or 'mixed'.
        num_peaks (int): Number of spikes to combine.
        seed (int): Random seed for reproducibility.
    Returns:
        x (array): x-axis values
        y (array): y-axis continuous values
        true_auc (float): The mathematically exact Area Under Curve
    """
    rng = np.random.RandomState(seed)
    x = np.linspace(0, 100, n_points)
    y = np.zeros_like(x)
    true_auc = 0.0

    for _ in range(num_peaks):
        amp = rng.uniform(0.2, 1.0)
        mu = rng.uniform(10.0, 90.0)
        width = rng.uniform(1.0, 5.0)

        ctype = curve_type
        if curve_type == "mixed":
            ctype = rng.choice(["gaussian", "lorentzian"])

        if ctype == "gaussian":
            y += gaussian(x, amp, mu, width)
            true_auc += gaussian_integral(amp, width)
        elif ctype == "lorentzian":
            y += lorentzian(x, amp, mu, width)
            true_auc += lorentzian_integral(amp, width)

    return x, y, true_auc
