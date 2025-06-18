import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

# Create a simple graph with 3 nodes A, B, C
G = nx.Graph()
G.add_edges_from([('A', 'B'), ('B', 'C'), ('C', 'A')])

# Node features (example vectors)
features = {
    'A': np.array([1, 0]),
    'B': np.array([0, 1]),
    'C': np.array([1, 1])
}

# Positions for consistent layout
pos = nx.circular_layout(G)

# Plot graph with nodes and edges
plt.figure(figsize=(12, 6))

# Left side: Message Passing Graph
plt.subplot(1, 2, 1)
nx.draw(G, pos, with_labels=True, node_size=1500, node_color='skyblue', font_size=16, font_weight='bold')

# Show node features inside nodes
for node, coord in pos.items():
    feat = features[node]
    plt.text(coord[0], coord[1]-0.1, f"{feat}", fontsize=12, ha='center', va='center', fontfamily='monospace')

plt.title("Message Passing Graph\n(Node Features and Edges)")
plt.savefig("../images/MPGNN.png")

# Right side: GCN Layer Visualization
plt.subplot(1, 2, 2)

# Draw nodes (empty circles)
for node, coord in pos.items():
    circle = plt.Circle(coord, 0.15, color='lightgreen', fill=True)
    plt.gca().add_patch(circle)
    # Original features (input)
    feat_in = features[node]
    plt.text(coord[0]-0.18, coord[1]+0.12, f"Input:\n{feat_in}", fontsize=10, ha='left', va='center', fontfamily='monospace')

# Show aggregation step (sum of neighbors)
agg_features = {}
for node in G.nodes():
    neighbors = list(G.neighbors(node))
    agg = sum(features[n] for n in neighbors)
    agg_features[node] = agg
    coord = pos[node]
    plt.text(coord[0]-0.18, coord[1]-0.05, f"Agg:\n{agg}", fontsize=10, ha='left', va='center', fontfamily='monospace')

# Weight matrix W (identity for simplicity)
W = np.eye(2)
def relu(x):
    return np.maximum(0, x)

updated_features = {}
for node, agg in agg_features.items():
    h = relu(W @ agg)
    updated_features[node] = h
    coord = pos[node]
    plt.text(coord[0]-0.18, coord[1]-0.22, f"Output:\n{h}", fontsize=10, ha='left', va='center', fontfamily='monospace')

# Draw edges with arrows to indicate message passing direction (undirected shown as two arrows)
for u, v in G.edges():
    x1, y1 = pos[u]
    x2, y2 = pos[v]
    plt.arrow(x1, y1, (x2-x1)*0.6, (y2-y1)*0.6, head_width=0.05, head_length=0.07, fc='gray', ec='gray', alpha=0.7)
    plt.arrow(x2, y2, (x1-x2)*0.6, (y1-y2)*0.6, head_width=0.05, head_length=0.07, fc='gray', ec='gray', alpha=0.7)

plt.title("Graph Convolutional Network Layer\n(Message Passing, Aggregation, Update)")
plt.axis('off')

plt.tight_layout()
plt.savefig("../GCN.png")
plt.show()