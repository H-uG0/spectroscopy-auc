import os
import sys

# Add src to python path
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.processing.extractor import extract_signal
from src.processing.integrator import (
    calculate_auc_simpson,
    calculate_auc_spline,
    calculate_auc_trapezoidal,
)
from src.simulation.generator import generate_ground_truth
from src.simulation.sensor import save_signal_as_tif, simulate_sensor
from src.utils.visualization import plot_signals, print_results


def run_pipeline():
    # 1. Generate Ground Truth
    print("Generating ground truth signal...")
    x_truth, y_truth, true_auc = generate_ground_truth(n_points=1000)

    # 2. Simulate Sensor & Save Image
    print("Simulating sensor and saving TIFF image...")
    image_file = "data/simulated_spectrum.tif"
    image_data, scaling_factor = simulate_sensor(
        y_truth, width=100, height=20, bit_depth=16
    )
    save_signal_as_tif(image_data, image_file)

    # 3. Extract Signal from Image
    print("Extracting signal from image...")
    # x_scaling=1.0 matches the default behavior of mapping pixels to unit steps
    x_extracted, y_extracted = extract_signal(
        image_file, x_scaling=1.0, y_scaling=scaling_factor
    )

    # 4. Calculate AUC
    print("Calculating AUC using various methods...")
    auc_trap = calculate_auc_trapezoidal(x_extracted, y_extracted)
    auc_simp = calculate_auc_simpson(x_extracted, y_extracted)
    auc_spline = calculate_auc_spline(x_extracted, y_extracted)

    # 5. Output Results
    print_results(true_auc, auc_trap, auc_simp, auc_spline)

    # 6. Optional: Visualization (Uncomment if needed)
    # plot_signals(x_truth, y_truth, x_extracted, y_extracted)


if __name__ == "__main__":
    if not os.path.exists("data"):
        os.makedirs("data")
    run_pipeline()
