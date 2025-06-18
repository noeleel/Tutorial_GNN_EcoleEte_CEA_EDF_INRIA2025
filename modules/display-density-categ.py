import argparse
import pickle
import numpy as np
import matplotlib.pyplot as plt

def load_results(filepath):
    """Load prediction results, labels, and index from a pickle file."""
    return pickle.load(open(filepath, "rb"))

def compute_error_distribution(results, labels, magnitude_threshold, error_range):
    """
    Split magnitude prediction errors into two categories based on the magnitude threshold.
    Returns histograms and bins for each category.
    """
    errors = results[:, 3] - labels[:, 3]
    mask_low = labels[:, 3] < magnitude_threshold
    mask_high = labels[:, 3] >= magnitude_threshold

    hist_low, bins_low = np.histogram(errors[mask_low], bins=100, range=error_range, density=True)
    hist_high, bins_high = np.histogram(errors[mask_high], bins=50, range=error_range, density=True)

    centers_low = 0.5 * (bins_low[1:] + bins_low[:-1])
    centers_high = 0.5 * (bins_high[1:] + bins_high[:-1])

    return (centers_low, hist_low, errors[mask_low]), (centers_high, hist_high, errors[mask_high])

def plot_density_and_boxplot(axs, centers_low, hist_low, errors_low, centers_high, hist_high, errors_high, threshold):
    """
    Plot the density and boxplot for magnitude prediction errors split by threshold.
    """
    # Density plot (top)
    ax_density = axs[0]
    ax_density.plot(centers_low, hist_low, c="C2", lw=1, alpha=1, label=f"M < {threshold}")
    ax_density.fill_between(centers_low, 0, hist_low, fc="C2", alpha=0.1)

    ax_density.plot(centers_high, hist_high, c="C3", lw=1, alpha=1, label=f"M ≥ {threshold}")
    ax_density.fill_between(centers_high, 0, hist_high, fc="C3", alpha=0.1)

    # Boxplot (bottom)
    ax_box = axs[1]
    box_low = ax_box.boxplot(errors_low, sym="o", vert=False, whis=[5, 95], positions=[0],
                             widths=0.5, flierprops=dict(markersize=5), patch_artist=True)
    for patch in box_low["boxes"]:
        patch.set_facecolor("C2")

    box_high = ax_box.boxplot(errors_high, sym="o", vert=False, whis=[5, 95], positions=[1],
                              widths=0.5, flierprops=dict(markersize=5), patch_artist=True)
    for patch in box_high["boxes"]:
        patch.set_facecolor("C3")

    ax_box.set_yticks([])

def add_vertical_guidelines(ax, positions):
    """Add vertical guide lines to a plot for visual reference."""
    for pos in positions:
        ax.axvline(pos, color="grey", lw=0.5)

def main():
    parser = argparse.ArgumentParser(description="Plot magnitude error densities split by magnitude threshold.")
    parser.add_argument('--models', nargs='+', type=str, required=True, help='Model name prefixes')
    parser.add_argument('--legend', nargs='+', type=str, required=True, help='Legend labels for models')
    parser.add_argument('--name', type=str, required=True, help='Output figure filename prefix')
    parser.add_argument('--limit', type=float, default=3.5, help='Magnitude threshold for splitting categories')
    args = parser.parse_args()

    plt.rcParams.update({"font.size": 16})
    fig, axs = plt.subplots(2, 1, figsize=(5, 6), gridspec_kw={'height_ratios': [3, 1]})

    error_range = (-0.75, 0.5)
    vlines = [-0.6, -0.4, -0.2, 0.0, 0.2, 0.4, 0.6]
    
    # Set axes limits and labels
    for ax in axs:
        ax.set_xlim(error_range)
        ax.set_xticks(vlines)
    axs[0].set_ylim(0.0, 5.0)
    axs[1].invert_yaxis()
    axs[1].set_title("Magnitude")

    # Add reference lines
    add_vertical_guidelines(axs[0], vlines)

    # Plot each model
    for model, label in zip(args.models, args.legend):
        results, labels, _ = load_results(f"{model}-all-test-results-mul.pkl")
        (centers_low, hist_low, errors_low), (centers_high, hist_high, errors_high) = \
            compute_error_distribution(results, labels, args.limit, error_range)

        plot_density_and_boxplot(axs, centers_low, hist_low, errors_low,
                                 centers_high, hist_high, errors_high,
                                 args.limit)

    axs[0].legend()
    fig.tight_layout()
    plt.savefig(f"{args.name}-density-categ.jpeg")
    plt.close()

if __name__ == "__main__":
    main()

