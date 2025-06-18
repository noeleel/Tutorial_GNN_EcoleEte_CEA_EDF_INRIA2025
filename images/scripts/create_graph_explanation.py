import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import networkx as nx
import numpy as np

fig, axs = plt.subplots(5, 1, figsize=(10, 25))

# 1. Simple graph
G = nx.DiGraph()
edges = [(0,1), (1,2), (2,3), (3,0), (0,2)]
G.add_edges_from(edges)
pos = nx.spring_layout(G, seed=42)

nx.draw(G, pos, ax=axs[0], with_labels=True, node_color='lightblue', node_size=800)
axs[0].set_title("1. Graph: Nodes and Edges")
axs[0].axis('off')

# 2. Adjacency matrix
adj = nx.adjacency_matrix(G).todense()
im = axs[1].imshow(adj, cmap='Blues')
for (i, j), val in np.ndenumerate(adj):
    axs[1].text(j, i, int(val), ha='center', va='center', color='black', fontsize=14)
axs[1].set_title("2. Adjacency Matrix Representation")
axs[1].set_xlabel("Nodes")
axs[1].set_ylabel("Nodes")
axs[1].xaxis.set_major_locator(MaxNLocator(integer=True))
axs[1].yaxis.set_major_locator(MaxNLocator(integer=True))


# 3. GNN Concept - message passing (schematic)
axs[2].set_title("3. Graph Neural Network (GNN) Concept")
axs[2].axis('off')
axs[2].text(0.5, 0.6, "Nodes exchange\ninformation\nvia edges", fontsize=16)
# Simple schematic: draw two nodes and arrows
axs[2].arrow(0.4, 0.7, 0.2, 0, head_width=0.05, length_includes_head=True, color='navy')
axs[2].arrow(0.6, 0.5, -0.2, 0, head_width=0.05, length_includes_head=True, color='navy')
axs[2].plot([0.4, 0.6], [0.7, 0.7], 'o', markersize=30, color='lightblue')
axs[2].plot([0.6, 0.4], [0.5, 0.5], 'o', markersize=30, color='lightblue')
axs[2].text(0.4, 0.8, "Node A", fontsize=12, ha='center')
axs[2].text(0.6, 0.8, "Node B", fontsize=12, ha='center')

# 4. MPNN - message passing and update
axs[3].set_title("4. Message Passing Neural Network (MPNN)\nMessage + Aggregate + Update with update rule: $h_i^{(k)} = U^{(k)}( h_i^{(k-1)}, \sum_{j \in \mathcal{N}(i)} M^{(k)} (h_i^{(k-1)}, h_j^{(k-1)}, e_{ij}))$ ")
axs[3].axis('off')
# Boxes for steps
axs[3].text(0.1, 0.7, "1. Message from neighbors\n(m_ij)", bbox=dict(boxstyle="round,pad=0.5", fc="lightgreen"), fontsize=14)
axs[3].text(0.55, 0.7, "2. Aggregate messages\n(Σ m_ij)", bbox=dict(boxstyle="round,pad=0.5", fc="lightyellow"), fontsize=14)
axs[3].text(0.94, 0.7, "3. Update node state\n(h_i)", bbox=dict(boxstyle="round,pad=0.5", fc="lightcoral"), fontsize=14)


# 5. GCN - convolution on graph
axs[4].set_title("5. Graph Convolutional Network (GCN)\nLayer: $H^{(l+1)} = σ(D^{-1/2} A D^{-1/2} H^{(l)} W^{(l)})$")
axs[4].axis('off')
axs[4].text(0.05, 0.6, "GCN applies convolution\nvia normalized adjacency\nand learnable weights", fontsize=14)
axs[4].text(0.05, 0.3, "It's a special case of MPNN", fontsize=12, style='italic')

plt.tight_layout()
plt.savefig("../graph_explanation.png")
plt.show()
