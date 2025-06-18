import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

# === Load geodesic distance sensitivity data ===
sensi_path = "LDG_sensi_geodesic_dist.npy"
sensi = np.load(sensi_path, allow_pickle=True).item()

# === Define layers to process ===
layers = [1, 2, 4, 6, 8, 12, 16, 20]
network = "LDG"
base_path = "/home/noelee/ENE/GNN/EcoleEte_EDF_CEA_INRIA/oversmoothing"

# === Constants ===
epsilon = 0.1  # Offset for log-scale plotting

# === Begin processing for each layer ===
for layer in layers:
    model_key = f"{base_path}/{network}/GNN3-L-{layer}-DATA-select-2_5-0/res0/model.pt"
    dirichlet_path = f"{base_path}/{network}_GNN3_L{layer}_dirichlet.pkl"

    arr_s = sensi[model_key]
    distances = np.mean(arr_s[2, :, :-1, :], axis=-1)  # Shape: (samples, layers)

    # Load Dirichlet energy
    dirichlet_D, dirichlet_F = np.load(dirichlet_path, allow_pickle=True)

    # === Prepare DataFrame for Seaborn ===
    df = pd.DataFrame(distances, columns=[f"Layer {i+1}" for i in range(distances.shape[1])])
    df_melted = df.melt(var_name="Layers", value_name="Distance")
    df_melted["Distance_shifted"] = df_melted["Distance"] + epsilon
    layer_means = df.mean().values

    # === Plotting ===
    fig, ax1 = plt.subplots(figsize=(10, 6))

    sns.violinplot(data=df_melted, x="Layers", y="Distance_shifted",
                   palette="pastel", ax=ax1)
    sns.pointplot(data=df_melted, x="Layers", y="Distance", estimator=np.mean,
                  color="black", markers="D", scale=0.75, linestyles="", errwidth=0, ax=ax1)

    for i, mean_val in enumerate(layer_means):
        ax1.text(i, mean_val + 0.05, f"{mean_val:.2f}",
                 ha="center", va="bottom", fontsize=9, fontweight="bold", color="black")

    ax1.set_ylabel(r"Distance in km + $\epsilon = 0.1$")
    ax1.set_xlabel("Layer")
    ax1.tick_params(axis='x', rotation=45)
    ax1.grid(True, linestyle="--", alpha=0.3)

    # === Secondary Y-axis for Dirichlet energy ===
    ax2 = ax1.twinx()
    nlayers = dirichlet_D.shape[1]
    ax2.semilogy(range(nlayers), np.mean(dirichlet_D[:-1], axis=0), "r-o",
                 linewidth=3, label="Static Laplacian")
    ax2.semilogy(range(nlayers), np.mean(dirichlet_F[:-1], axis=0), "r:X",
                 linewidth=3, label="Dynamic Laplacian")
    ax2.set_ylabel("Dirichlet Energy", color="red")
    ax2.tick_params(axis='y', labelcolor='red')
    ax2.legend(loc="upper right")

    plt.title(f"{network}: Distance Distribution & Dirichlet Energy (L={layer})")
    fig.tight_layout()

    output_path = f"../boxplot_oversmoothing_L{layer}.png"
    plt.savefig(output_path)
    plt.close(fig)

# === Final call to ensure display in notebooks (optional in scripts) ===
plt.show()
