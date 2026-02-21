import json
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

# Add src to python path
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.processing.extractor import extract_signal
from src.processing.integrator import calculate_auc_simpson, calculate_auc_spline


def run_performance_test(dataset_dir="tests/dataset"):
    metadata_path = os.path.join(dataset_dir, "metadata.json")
    if not os.path.exists(metadata_path):
        print("Metadata not found. Please run generate_test_dataset.py first.")
        return

    with open(metadata_path, "r") as f:
        test_cases = json.load(f)

    results = []
    print("Running performance tests...")

    for case in test_cases:
        filepath = os.path.join(dataset_dir, case["filename"])

        # Extract signal (map pixels back to the original 0-100 x-axis range)
        x_scaling = 100.0 / (case["width"] - 1)
        x, y_ext = extract_signal(
            filepath,
            x_scaling=x_scaling,
            y_scaling=case["scaling_factor"],
            orientation=case["orientation"],
        )

        # Integrate (Simpson)
        auc_simpson = calculate_auc_simpson(x, y_ext)

        # Integrate (Spline s=0)
        auc_spline = calculate_auc_spline(x, y_ext)

        true_auc = case["true_auc"]
        error_rel_simpson = abs(auc_simpson - true_auc) / true_auc * 100
        error_rel_spline = abs(auc_spline - true_auc) / true_auc * 100

        case["auc_simpson"] = auc_simpson
        case["auc_spline"] = auc_spline
        case["error_rel_simpson"] = error_rel_simpson
        case["error_rel_spline"] = error_rel_spline
        results.append(case)

    # Analysis & Visualization
    visualize_errors(results)

    return results


def visualize_errors(results):
    widths = sorted(list(set(r["width"] for r in results)))
    bit_depths = sorted(list(set(r["bit_depth"] for r in results)))
    curve_types = sorted(list(set(r["curve_type"] for r in results)))

    # Calculate means for heatmaps
    heatmap_simpson = np.zeros((len(bit_depths), len(widths)))
    heatmap_spline = np.zeros((len(bit_depths), len(widths)))
    for i, bd in enumerate(bit_depths):
        for j, w in enumerate(widths):
            errs_simpson = [
                r["error_rel_simpson"]
                for r in results
                if r["bit_depth"] == bd and r["width"] == w
            ]
            errs_spline = [
                r["error_rel_spline"]
                for r in results
                if r["bit_depth"] == bd and r["width"] == w
            ]
            heatmap_simpson[i, j] = np.mean(errs_simpson) if errs_simpson else 0.0
            heatmap_spline[i, j] = np.mean(errs_spline) if errs_spline else 0.0

    # General Width errors
    error_by_width_simpson = {w: [] for w in widths}
    error_by_width_spline = {w: [] for w in widths}
    for r in results:
        error_by_width_simpson[r["width"]].append(r["error_rel_simpson"])
        error_by_width_spline[r["width"]].append(r["error_rel_spline"])
    avg_error_width_simpson = [np.mean(error_by_width_simpson[w]) for w in widths]
    avg_error_width_spline = [np.mean(error_by_width_spline[w]) for w in widths]

    # Curve Type errors
    err_by_ctype_simpson = {ct: [] for ct in curve_types}
    err_by_ctype_spline = {ct: [] for ct in curve_types}
    for r in results:
        err_by_ctype_simpson[r["curve_type"]].append(r["error_rel_simpson"])
        err_by_ctype_spline[r["curve_type"]].append(r["error_rel_spline"])

    # Create figure
    fig = plt.figure(figsize=(16, 10))

    # 1. Heatmap (Simpson)
    ax1 = plt.subplot(2, 2, 1)
    cax1 = ax1.imshow(
        heatmap_simpson, cmap="viridis", aspect="auto", interpolation="nearest"
    )
    ax1.set_xticks(np.arange(len(widths)))
    ax1.set_yticks(np.arange(len(bit_depths)))
    ax1.set_xticklabels(widths)
    ax1.set_yticklabels(bit_depths)
    ax1.set_xlabel("Width (pixels)")
    ax1.set_ylabel("Bit Depth")
    ax1.set_title("Simpson Mean Relative Error (%)")
    fig.colorbar(cax1, ax=ax1, label="Relative Error (%)")

    # 2. Heatmap (Spline)
    ax2 = plt.subplot(2, 2, 2)
    cax2 = ax2.imshow(
        heatmap_spline, cmap="viridis", aspect="auto", interpolation="nearest"
    )
    ax2.set_xticks(np.arange(len(widths)))
    ax2.set_yticks(np.arange(len(bit_depths)))
    ax2.set_xticklabels(widths)
    ax2.set_yticklabels(bit_depths)
    ax2.set_xlabel("Width (pixels)")
    ax2.set_ylabel("Bit Depth")
    ax2.set_title("Spline (s=0) Mean Relative Error (%)")
    fig.colorbar(cax2, ax=ax2, label="Relative Error (%)")

    # 3. Avg Error vs Width (Log-Log)
    ax3 = plt.subplot(2, 2, 3)
    ax3.plot(
        widths,
        avg_error_width_simpson,
        marker="o",
        linestyle="-",
        color="dodgerblue",
        label="Simpson",
    )
    ax3.plot(
        widths,
        avg_error_width_spline,
        marker="s",
        linestyle="--",
        color="salmon",
        label="Spline (s=0)",
    )
    ax3.set_title("Global Avg Error vs Width")
    ax3.set_xlabel("Width (pixels)")
    ax3.set_ylabel("Relative Error (%)")
    ax3.set_xscale("log")
    ax3.set_yscale("log")
    ax3.grid(True, which="both", ls="-", alpha=0.5)
    ax3.legend()

    # 4. Avg Error by Curve Type Grouped Bar
    ax4 = plt.subplot(2, 2, 4)
    x = np.arange(len(curve_types))
    width = 0.35
    means_simpson = [np.mean(err_by_ctype_simpson[ct]) for ct in curve_types]
    means_spline = [np.mean(err_by_ctype_spline[ct]) for ct in curve_types]
    rects1 = ax4.bar(
        x - width / 2, means_simpson, width, label="Simpson", color="dodgerblue"
    )
    rects2 = ax4.bar(
        x + width / 2, means_spline, width, label="Spline (s=0)", color="salmon"
    )
    ax4.set_ylabel("Mean Relative Error (%)")
    ax4.set_title("Average Error by Curve Type")
    ax4.set_xticks(x)
    ax4.set_xticklabels([ct.capitalize() for ct in curve_types])
    ax4.legend()

    plt.tight_layout()
    plt.savefig("tests/performance_results.png")
    print("Optimization results saved to tests/performance_results.png")


if __name__ == "__main__":
    run_performance_test()
