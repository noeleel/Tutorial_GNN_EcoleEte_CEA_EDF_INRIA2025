import argparse
import pickle
import numpy as np
import matplotlib.pyplot as plt

def haversine_distance(lat1, lon1, lat2, lon2):
    """
    Compute great-circle distance between two geographic coordinates.
    Input angles are in radians.
    Returns distance in kilometers.
    """
    return 6371 * np.arccos(
        np.sin(lat1) * np.sin(lat2) +
        np.cos(lat1) * np.cos(lat2) * np.cos(lon1 - lon2)
    )

def compute_rmse(errors, folds):
    """
    Compute RMSE per fold.
    """
    rmse_per_fold = np.zeros(10)
    for f in range(10):
        mask = folds == f
        rmse_per_fold[f] = np.sqrt(np.mean(errors[mask]**2))
    return rmse_per_fold

def plot_model_errors(axs, models, titles):
    """
    Load result files, compute RMSEs, and plot boxplots.
    """
    for i in range(3):  # Loop over Distance, Depth, Magnitude
        axs[i].set_xlim(-0.5, len(models) - 0.5)
        for j, model in enumerate(models):
            # Load test results: predictions, labels, and fold indices
            results, labels, index = pickle.load(open(f"{model}-all-test-results.pkl", "rb"))

            if i == 0:  # Distance error
                lat_pred = np.deg2rad(results[:, 0])
                lon_pred = np.deg2rad(results[:, 1])
                lat_true = np.deg2rad(labels[:, 0])
                lon_true = np.deg2rad(labels[:, 1])
                errors = haversine_distance(lat_pred, lon_pred, lat_true, lon_true)
            else:  # Depth or Magnitude error
                errors = results[:, i + 1] - labels[:, i + 1]

            rmse = compute_rmse(errors, index[:, 1])
            axs[i].boxplot(
                rmse,
                sym="",
                whis=[5, 95],
                positions=[j],
                widths=0.5,
                labels=[model]
            )

        axs[i].set_title(titles[i])

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Plot RMSE boxplots for model evaluation.")
    parser.add_argument('--name', type=str, default='', help='Prefix for the output image filename')
    parser.add_argument('--models', nargs='+', type=str, required=True, help='List of model name prefixes')
    args = parser.parse_args()

    # Initialize subplots
    fig, axs = plt.subplots(1, 3, figsize=(16, 8))
    titles = ("Distance", "Depth", "Magnitude")

    # Set common axis settings
    for ax, title in zip(axs, titles):
        ax.set_ylabel("error [km]" if title != "Magnitude" else "error")
        ax.set_title(title)

    axs[0].set_ylim(10.0, 30.0)
    axs[1].set_ylim(3.8, 5.0)
    axs[2].set_ylim(0.15, 0.3)

    # Plot boxplots for each metric
    plot_model_errors(axs, args.models, titles)

    # Save the figure
    plt.tight_layout()
    plt.savefig(f"{args.name}-boxplot.jpeg")
    plt.close()

if __name__ == "__main__":
    main()

