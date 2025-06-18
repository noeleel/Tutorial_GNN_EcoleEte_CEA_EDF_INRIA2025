import argparse
import pickle
import numpy as np
import matplotlib.pyplot as plt

def haversine_distance(lat1, lon1, lat2, lon2):
    """
    Calculate the great-circle distance between two points on Earth.
    All inputs are in radians.
    Returns distance in kilometers.
    """
    return 6371 * np.arccos(
        np.sin(lat1) * np.sin(lat2) +
        np.cos(lat1) * np.cos(lat2) * np.cos(lon1 - lon2)
    )

def compute_foldwise_rmse(errors, fold_indices):
    """
    Compute RMSE for each fold using the index array.
    `fold_indices` should be a 1D array of integers representing fold assignments.
    """
    rmse = np.zeros(10)
    for f in range(10):
        fold_mask = (fold_indices == f)
        squared_errors = (np.abs(errors)**2) * fold_mask
        rmse[f] = np.sqrt(np.sum(squared_errors) / np.sum(fold_mask))
    return rmse

def plot_rmse_boxplots(axs, models, metric_titles):
    """
    Load model outputs, compute RMSEs by fold, and create boxplots.
    """
    for i in range(3):  # Loop over Distance, Depth, Magnitude
        axs[i].set_xlim(-0.5, len(models) - 0.5)

        for j, model in enumerate(models):
            # Load predictions, ground truth labels, and fold indices
            results, labels, index = pickle.load(open(f"{model}-all-test-results.pkl", "rb"))

            if i == 0:
                # Compute geodesic distance error for location (lat, lon)
                lat_pred = np.deg2rad(results[:, 0])
                lon_pred = np.deg2rad(results[:, 1])
                lat_true = np.deg2rad(labels[:, 0])
                lon_true = np.deg2rad(labels[:, 1])
                errors = haversine_distance(lat_pred, lon_pred, lat_true, lon_true)
            else:
                # Compute simple difference for depth or magnitude
                errors = results[:, i + 1] - labels[:, i + 1]

            # Compute RMSE across 10 folds
            rmse = compute_foldwise_rmse(errors, index[:, 1])

            # Plot boxplot at the j-th position
            axs[i].boxplot(
                rmse,
                sym="",             # No outliers
                whis=[5, 95],       # 5th to 95th percentile whiskers
                positions=[j],
                widths=0.5,
                labels=[model]
            )

        axs[i].set_title(metric_titles[i])

def main():
    # Command-line argument parsing
    parser = argparse.ArgumentParser(description="Generate boxplots of RMSE for multiple models.")
    parser.add_argument('--name', type=str, default='', help='Prefix for output filename')
    parser.add_argument('--models', nargs='+', type=str, required=True, help='List of model identifiers')
    args = parser.parse_args()

    # Plot setup
    fig, axs = plt.subplots(1, 3, figsize=(16, 8))
    titles = ("Distance", "Depth", "Magnitude")

    # Axis labels and y-axis ranges
    y_labels = ["error [km]", "error [km]", "error"]
    y_limits = [(10.0, 30.0), (3.8, 5.0), (0.15, 0.3)]

    for ax, label, y_lim in zip(axs, y_labels, y_limits):
        ax.set_ylabel(label)
        ax.set_ylim(*y_lim)

    # Create boxplots for each metric
    plot_rmse_boxplots(axs, args.models, titles)

    # Save figure
    plt.tight_layout()
    plt.savefig(f"{args.name}-boxplot.jpeg")
    plt.close()

if __name__ == "__main__":
    main()

