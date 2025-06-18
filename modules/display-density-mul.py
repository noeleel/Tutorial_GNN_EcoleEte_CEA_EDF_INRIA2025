import argparse
import pickle
import numpy as np
import matplotlib.pyplot as plt

def load_results(file_prefix):
    """Load model results, labels, and index from a pickle file."""
    return pickle.load(open(f"{file_prefix}-all-test-results-mul.pkl", "rb"))

def compute_error(i, results, labels, use_logmag=False):
    """Compute error for the i-th feature."""
    if i == 0:  # Distance error using haversine formula
        lat_pred = np.radians(results[:, 0])
        lat_true = np.radians(labels[:, 0])
        lon_pred = np.radians(results[:, 1])
        lon_true = np.radians(labels[:, 1])
        return 6371 * np.arccos(
            np.clip(
                np.sin(lat_pred)*np.sin(lat_true) +
                np.cos(lat_pred)*np.cos(lat_true)*np.cos(lon_pred - lon_true),
                -1.0, 1.0
            )
        )
    elif i == 2 and use_logmag:  # Log-relative magnitude error
        return (np.log(results[:, 3]) - np.log(labels[:, 3])) / np.log(labels[:, 3])
    else:
        return results[:, i + 1] - labels[:, i + 1]

def plot_density_and_boxplot(i, results, labels, label_name, position, axs, error_ranges, use_logmag):
    """Plot both the density and the boxplot for a given error dimension."""
    error = compute_error(i, results, labels, use_logmag)

    # Density plot (bottom row)
    ax_density = axs[1, i]
    hist, bins = np.histogram(error, bins=100, range=error_ranges[i], density=True)
    bin_centers = 0.5 * (bins[1:] + bins[:-1])
    ax_density.plot(bin_centers, hist, lw=1., alpha=1, label=label_name, color=f"C{position}")
    ax_density.fill_between(bin_centers, 0, hist, alpha=0.1, color=f"C{position}")

    # Boxplot (top row)
    ax_box = axs[0, i]
    ax_box.boxplot(error, sym="", vert=False, whis=[5, 95], positions=[position], widths=0.5, labels=[label_name])
    if i != 0:
        ax_box.set_yticks([])

def add_grid_lines(ax, positions):
    """Add vertical reference lines for easier reading."""
    for pos in positions:
        ax.axvline(pos, color="grey", lw=0.5)

def main():
    parser = argparse.ArgumentParser(description="Plot distance, depth, and magnitude error distributions.")
    parser.add_argument('--model', type=str, default='.', help='Main model prefix')
    parser.add_argument('--ref', type=str, default='.', help='Reference model prefix')
    parser.add_argument('--logmag', action="store_true", help='Use logarithmic magnitude error')
    args = parser.parse_args()

    model_name = args.model
    ref_name = args.ref
    use_logmag = args.logmag

    # Load model results
    results, labels, _ = load_results(model_name)
    if ref_name != ".":
        results_ref, labels_ref, _ = load_results(ref_name)

    # Plot setup
    fig, axs = plt.subplots(2, 3, figsize=(16, 6), gridspec_kw={'height_ratios': [1, 3]})
    feature_titles = ("Distance", "Depth", "Magnitude")
    error_ranges = ((0, 200), (-30, 30), (-2.0, 2.0))
    xtick_values = [[0, 20, 40, 60, 80], [-20, -10, 0, 10, 20], [-0.4, -0.2, 0.0, 0.2, 0.4]]

    # Set axis labels and limits
    for i in range(3):
        axs[1, i].set_xlabel("error [km]" if i < 2 else "error")
        axs[1, i].set_xlim(error_ranges[i])
        axs[0, i].invert_yaxis()
        axs[0, i].set_title(feature_titles[i])
        axs[1, i].set_ylim((0.0, 0.1) if i == 0 else (0.0, 0.2) if i == 1 else (0.0, 5.0))
        add_grid_lines(axs[1, i], xtick_values[i])

    axs[1, 0].set_ylabel("density")

    # Plot for reference and target model
    for i in range(3):
        if ref_name != ".":
            plot_density_and_boxplot(i, results_ref, labels_ref, ref_name, 0, axs, error_ranges, use_logmag)
        plot_density_and_boxplot(i, results, labels, model_name, 1, axs, error_ranges, use_logmag)

    # Save figure
    out_file = f"{model_name}-density-mul-logmag.jpeg" if use_logmag else f"{model_name}-density-mul.jpeg"
    plt.tight_layout()
    plt.savefig(out_file)
    plt.close()

if __name__ == "__main__":
    main()

