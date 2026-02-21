import matplotlib.pyplot as plt


def plot_signals(
    x_truth, y_truth, x_ext, y_ext, title="Signal Reconstruction Validation"
):
    """
    Plots ground truth vs extracted signal.
    """
    plt.figure(figsize=(12, 6))
    plt.plot(x_truth, y_truth, "k-", alpha=0.3, label="Ground Truth (Analog)")
    plt.plot(x_ext, y_ext, "r--", label="Extracted (Reconstructed)")
    plt.title(title)
    plt.xlabel("X-axis")
    plt.ylabel("Amplitude")
    plt.grid(True)
    plt.legend()
    plt.show()


def print_results(true_auc, auc_trapezoidal, auc_simpson, auc_spline):
    """
    Prints comparison of AUC results.
    """
    print("\n--- AUC CALCULATION RESULTS ---")
    print(f"True AUC (Ground Truth): {true_auc:.5f}")
    print(
        f"AUC (Trapezoidal)      : {auc_trapezoidal:.5f} (Err: {abs(auc_trapezoidal-true_auc)/true_auc*100:.3f}%)"
    )
    print(
        f"AUC (Simpson)          : {auc_simpson:.5f} (Err: {abs(auc_simpson-true_auc)/true_auc*100:.3f}%)"
    )
    print(
        f"AUC (Spline)           : {auc_spline:.5f} (Err: {abs(auc_spline-true_auc)/true_auc*100:.3f}%)"
    )
