# === Import required libraries ===
import pandas as pd
import os, pickle, torch, timeit, sys
import numpy as np
from torchinfo import summary
from tqdm import tqdm
import torch.nn.functional as F
import torch.nn as nn
import gc
import glob


_modules = [os.path.join(os.getcwd(), "../../"),
            os.path.join(os.getcwd(), "../../modules"),
           os.path.join(os.getcwd(), "../../modules/model.py"),
           os.path.join(os.getcwd(), "../../modules/utils.py")]

for _module in _modules:
    if _module not in sys.path:
        sys.path.append(_module)
        print(f"{_module} added to sys.path")
        
from modules import utils as ut
from modules import model as md
import networkx as nx
import matplotlib.pyplot as plt


# === Clear CUDA memory and run garbage collection ===
torch.cuda.empty_cache()
gc.collect()

# === Set experiment parameters ===
seed = 0
device = "cuda:0"
np.random.seed(seed)
torch.manual_seed(seed)
torch.backends.cudnn.deterministic = True

# === Data and model paths ===


model_file = "/home/noelee/ENE/GNN/EcoleEte_EDF_CEA_INRIA/oversmoothing/SCDEC/GNN3-L-12-DATA-select-less-sta-0/res0/seed1/model.pt"
data_file = "/home/noelee/ENE/GNN/EcoleEte_EDF_CEA_INRIA/data/SCDEC/data-0.pkl"


minlatitude = 32
maxlatitude = 36
minlongitude = -120
maxlongitude = -116

def unscale_lat(x):
        x = x / 2 + 0.5
        return x * (maxlatitude - minlatitude) + minlatitude

def unscale_lon(x):
        x = x / 2 + 0.5
        return x * (maxlongitude - minlongitude) + minlongitude
    
torch.cuda.empty_cache()
gc.collect()

n_epochs = 800
batch_size = 32
lr = 0.0002
dropout_rate = 0.15
K = 5  # Number of nearest neighbors
nlayers = int(model_file.split("-L-")[1].split("-")[0])  # Number of GNN layers
concat = True
L2 = False
version= 3
max_eval = 64  # Batch size during inference

# === Load last dataset file ===
(x_test, y_test, pos, dist) = pickle.load(open(data_file, "rb"))

x_test = torch.permute(torch.Tensor(x_test).to(device),(0,3,1,2))
y_test = torch.Tensor(y_test[:,1:]).to(device)
pos = torch.Tensor(pos.T).to(device)
dist = torch.Tensor(dist).to(device)
# === Initialize and load pretrained model ===
model = md.GNN3(x_test.shape[1], K, dist, pos, dropout_rate, version, nlayers).to(device)
model_state = torch.load(open(model_file, "rb"))
model.load_state_dict(model_state)

K=5

# Spatial neighbourhood
knn = torch.topk(dist, K, largest=False)[1].cpu().detach().numpy()

# Feature neighbourhood
def knn_f(X, K=K):
    diff = X.unsqueeze(1) - X.unsqueeze(2)  # pairwise differences
    dist = torch.sum(diff**2, (0, 3))  # pairwise distances
    indices = torch.topk(dist, K, largest=False)[1]
    return indices

model.eval()

node_coords = {
    i: (unscale_lon(lon.item()), unscale_lat(lat.item())) for i, (lon, lat) in enumerate(pos.T.cpu())
}
def plot_neigbhds(node_coords, indices, name = ""):
    plt.figure()
    G = nx.DiGraph()
    for node_id, (lon, lat) in node_coords.items():
        G.add_node(node_id, pos=(lon, lat))
    for i, row in enumerate(indices):
        for j in row:
            G.add_edge(i, j)
    G.remove_edges_from(nx.selfloop_edges(G))
    plt.savefig(f"{name}.png")

def plot_graph(node_coords, knn, name=""):
    plt.figure()
    # Build graph from KNN
    G = nx.DiGraph()
    for node_id, (lon, lat) in node_coords.items():
        G.add_node(node_id, pos=(lon, lat))
    for i, row in enumerate(knn):
        for j in row:
            G.add_edge(i, j)
    G.remove_edges_from(nx.selfloop_edges(G))
    plt.savefig(f"{name}.png")

plot_graph(node_coords, knn, "Spatial_graph")
plot_neigbhds({k:node_coords[k] for k in knn[0] if k in node_coords}, "Spatial_neighbds_node_0")

X_init = x_test[:max_eval]
X = model.CNNlayer(X_init)
print(X.shape)
adj_features = knn_f(X)
plot_neigbhds({k:node_coords[k] for k in adj_features[0] if k in node_coords}, adj_features[0], "Feature_neighbds_node_0_L_0")
plot_graph(node_coords, adj_features, "Feature_graph_L0")
for i, L in enumerate(model.slayers):
    print(i)
    print(X.shape)
    X = L(X)
    adj_features = knn_f(X)
    plot_neigbhds({k:node_coords[k] for k in adj_features[0] if k in node_coords}, adj_features[0], f"Fearure_neighbds_node_0_L_{i+1}")
    plot_graph(node_coords, adj_features, f"Feature_graph_L{i+1}")
