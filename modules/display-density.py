import argparse
import pickle
import numpy as np
import matplotlib.pyplot as plt

def haversine_distance(lat1, lon1, lat2, lon2):
    """
    Compute great-circle distance between two points on a sphere (Earth).
    All angles must be in radians.
    Returns distance in kilometers.
    """
    return 6371 * np.arccos(
        np.sin(lat1) * np.sin(lat2) +
        np.cos(lat1) * np.cos(lat2) * np.cos(lon1 - lon2)
    )

def compute_error(i, results, labels):
    """
    Compute the error metric for index i:
    - i=0: geodesic distance error (lat/lon)
    - i=1 or i=2: depth or magnitude error
    """
    if i == 0:
        lat_pred = np.deg2rad(results[:, 0])
        lon_pred = np.deg2rad(results[:, 1])
        lat_true = np.deg2rad(labels[:, 0])
        lon_true = np.deg2rad(labels[:, 1])
        return haversine_distance(lat_pred, lon_pred, lat_true, lon_true)
    else:
        return results[:, i + 1] - labels[:, i + 1]

def plot_density_and_boxplot(axs, i, errors, label, color_idx):
    """
    Plot both the histogram (density) and boxplot for a given metric.
    """
    # Lower panel: histogram
    ax_density = axs[1, i]
    hist, bins = np.histogram(errors, bins=100, range=ERROR_RANGES[i], density=True)
    bin_centers = 0.5 * (bins[1:] + bins[:-1])
    ax_density.plot(bin_centers, hist, c=f"C{color_idx}", lw=1.0, alpha=1.0, label=label)
    ax_density.fill_between(bin_centers, 0, hist, fc=f"C{color_idx}", alpha=0.1)

    # Upper panel: boxplot
    ax_box = axs[0, i]
    ax_box.boxplot(errors, sym="", vert=False, whis=[5, 95], positions=[color_idx], widths=0.5, labels=[label])
    if i != 0:
        ax_box.set_yticks([])

def add_vertical_guidelines(ax, positions):
    """
    Add light vertical reference lines at specified positions.
    """
    for pos in positions:
        ax.axvline(pos, color="grey", lw=0.5)

def main():
    parser = argparse.ArgumentParser(description="Compare prediction error densities between models.")
    parser.add_argument('--model', type=str, default='.', help='Primary model name prefix')
    parser.add_argument('--ref', type=str, default='.', help='Reference model name prefix')
    args = parser.parse_args()

    model_name = args.model
    ref_name = args.ref

    # Load results
    results, labels, _ = pickle.load(open(f"{model_name}-all-test-results.pkl", "rb"))
    if ref_name != ".":
        results_ref, labels_ref, _ = pickle.load(open(f"{ref_name}-all-test-results.pkl", "rb"))

    # Plot setup
    fig, axs = plt.subplots(2, 3, figsize=(16, 6), gridspec_kw={'height_ratios': [1, 3]})
    titles = ("Distance", "Depth", "Magnitude")
    for ax, title in zip(axs[0], titles):
        ax.set_title(title)
        ax.invert_yaxis()

    for i in range(3):
        # Axis labels and limits
        axs[1, i].set_xlabel("error [km]" if i < 2 else "error")
        axs[1, 0].set_ylabel("density" if i == 0 else "")
        axs[1, i].set_ylim(*Y_LIMITS[i])
        axs[0, i].set_xlim(*X_LIMITS[i])
        axs[1, i].set_xlim(*X_LIMITS[i])

        # Guidelines
        add_vertical_guidelines(axs[1, i], VLINES[i])

        # Reference model plot
        if ref_name != ".":
            errors_ref = compute_error(i, results_ref, labels_ref)
            plot_density_and_boxplot(axs, i, errors_ref, ref_name, 0)

        # Primary model plot
        errors = compute_error(i, results, labels)
        plot_density_and_boxplot(axs, i, errors, model_name, 1)

    plt.tight_layout()
    plt.savefig(f"{model_name}-density.jpeg")
    plt.close()

# Constant settings
ERROR_RANGES = ((0, 200), (-30, 30), (-2.0, 2.0))
X_LIMITS = ((0.0, 80.0), (-20.0, 20.0), (-0.5, 0.5))
Y_LIMITS = ((0.0, 0.1), (0.0, 0.2), (0.0, 5.0))
VLINES = [
    [0, 20, 40, 60, 80],
    [-20, -10, 0, 10, 20],
    [-0.4, -0.2, 0.0, 0.2, 0.4]
]

if __name__ == "__main__":
    main()

