import json
import os
import sys

import numpy as np

# Add src to python path
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.simulation.generator import generate_ground_truth
from src.simulation.sensor import save_signal_as_tif, simulate_sensor


def generate_dataset(output_dir="tests/dataset"):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    curve_types = ["gaussian", "lorentzian", "mixed"]
    n_variations = 3
    widths = [50, 100, 200, 400, 600, 800, 1000, 1500, 2000]
    heights = [20]
    bit_depths = [8, 16]
    orientations = ["horizontal"]

    test_cases = []
    print(f"Generating dataset in {output_dir}...")

    count = 0
    for ctype in curve_types:
        for var in range(n_variations):
            seed = sum(map(ord, ctype)) + var * 100
            x_truth, y_truth, true_auc = generate_ground_truth(
                n_points=2000, curve_type=ctype, seed=seed
            )

            for w in widths:
                for h in heights:
                    for bd in bit_depths:
                        for orient in orientations:
                            filename = (
                                f"spec_{ctype}_v{var}_w{w}_h{h}_bd{bd}_{orient}.tif"
                            )
                            filepath = os.path.join(output_dir, filename)

                            # Simulate
                            image_data, scaling_factor = simulate_sensor(
                                y_truth,
                                width=w,
                                height=h,
                                bit_depth=bd,
                                orientation=orient,
                            )

                            # Save
                            save_signal_as_tif(image_data, filepath)

                            test_cases.append(
                                {
                                    "filename": filename,
                                    "curve_type": ctype,
                                    "variation": var,
                                    "width": w,
                                    "height": h,
                                    "bit_depth": bd,
                                    "orientation": orient,
                                    "scaling_factor": scaling_factor,
                                    "true_auc": true_auc,
                                }
                            )
                            count += 1

    # Save metadata for test runner
    with open(os.path.join(output_dir, "metadata.json"), "w") as f:
        json.dump(test_cases, f, indent=4)

    print(f"Generated {count} test cases.")


if __name__ == "__main__":
    generate_dataset()
