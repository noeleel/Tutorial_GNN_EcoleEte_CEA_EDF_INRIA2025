import networkx as nx
import matplotlib.pyplot as plt

def plot_spatial_graph(p, show_labels=True):
    """
    Plot the spatial graph of seismic stations using their lat/lon positions.

    Args:
        p: numpy array of shape (N, 2), where each row is [lat, lon] for a station.
        show_labels: bool, whether to show node labels.
    """
    G = nx.Graph()

    # Add nodes with positions
    for i, (lat, lon) in enumerate(p):
        G.add_node(i, pos=(lon, lat))  # (lon, lat) for proper map alignment

    # Optionally: connect nearby stations (e.g., within X km or nearest neighbors)
    # Here we connect k nearest neighbors for visualization
    from sklearn.neighbors import NearestNeighbors
    k = 3  # change as needed
    nbrs = NearestNeighbors(n_neighbors=k+1, algorithm='ball_tree').fit(p)
    distances, indices = nbrs.kneighbors(p)
    for i, neighbors in enumerate(indices):
        for j in neighbors[1:]:  # skip self
            G.add_edge(i, j)

    pos = nx.get_node_attributes(G, 'pos')
    nx.draw(G, pos, with_labels=show_labels, node_color='skyblue', node_size=200, edge_color='gray')
    plt.title("Spatial Graph of Seismic Stations")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.grid(True)
    plt.show()
import numpy as np
from scipy.stats import pearsonr

def compute_correlation_matrix(x):
    """
    Compute correlation matrix between stations based on waveform data.
    
    Args:
        x: ndarray of shape (E, T, C, N) — waveform data.
    
    Returns:
        corr_matrix: (N, N) matrix of similarity scores between stations.
    """
    E, T, C, N = x.shape
    # Flatten across events, time, and channels
    flattened = x.reshape(E * T * C, N)
    
    corr_matrix = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            corr_matrix[i, j], _ = pearsonr(flattened[:, i], flattened[:, j])
    return corr_matrix

def plot_feature_graph(corr_matrix, p, threshold=0.7):
    """
    Create and plot a graph using correlation matrix as edges.
    
    Args:
        corr_matrix: (N, N) matrix of similarities
        p: (N, 2) station positions
        threshold: min correlation to include an edge
    """
    G = nx.Graph()

    # Add nodes with positions
    for i, (lat, lon) in enumerate(p):
        G.add_node(i, pos=(lon, lat))

    # Add edges above threshold
    N = corr_matrix.shape[0]
    for i in range(N):
        for j in range(i + 1, N):
            if corr_matrix[i, j] >= threshold:
                G.add_edge(i, j, weight=corr_matrix[i, j])

    pos = nx.get_node_attributes(G, 'pos')
    edges = G.edges()
    weights = [G[u][v]['weight'] for u, v in edges]

    nx.draw(G, pos, with_labels=True, node_color='lightgreen', edge_color=weights,
            edge_cmap=plt.cm.viridis, width=2, node_size=300)
    plt.title(f"Feature Graph (Correlation ≥ {threshold})")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.grid(True)
    plt.colorbar(plt.cm.ScalarMappable(cmap='viridis'), label='Correlation')
    plt.show()
