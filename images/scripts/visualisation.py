import numpy as np
import torch
import os
import pickle
import matplotlib.pyplot as plt
from sklearn.neighbors import kneighbors_graph


folder = "/home/noelee//EcoleEte_EDF_CEA_INRIA/data/SCDEC"
data_file = os.path.join(folder, "data-0.pkl")
device = "cpu"

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

(x_train, y_train, pos, dist) = pickle.load(open(os.path.join(folder, data_file), "rb"))

K = 5
knn = torch.topk(torch.Tensor(dist), K, largest=False)[1].numpy()
adj = np.zeros((len(pos), len(pos)))
for i, row in enumerate(knn):
    for j in row:
        adj[i, j] = 1

event = 50
colors = ['r-', 'b-', 'g-', 'c-', 'm-', 'y-']
stations = ['ADO', 'ALP', 'ARV', 'AVM', 'BAK', 'BAR']

plt.clf()
fig, axs = plt.subplots(6, 3, figsize = (16, 12))

for i in range(6):
        for dim in range(3):

                ax = axs[i, dim]
                serie = x_train[event, i, :, dim]

                t = np.linspace(0, len(serie)-1, len(serie))
                ax.plot(t, serie, colors[i])

                ax.set_title(stations[i])

plt.savefig(os.path.join(folder, "../visu_series.jpeg"))

plt.clf()
fig, ax = plt.subplots()

plt.plot(unscale_lon(y_train[event,2]), unscale_lat(y_train[event, 1]), 'ko-')
for i in range(6):
        plt.plot(unscale_lon(pos[i, 1]), unscale_lat(pos[i, 0]), colors[i], marker = '^')
        plt.text(unscale_lon(pos[i, 1]), unscale_lat(pos[i, 0]), stations[i])

plt.savefig(os.path.join(folder, "../visu_map.jpeg"))

plt.clf()

for i in range(len(pos)):
        lat1, lon1 = unscale_lat(pos[i, 0]), unscale_lon(pos[i, 1])
        for j in range(len(pos)):
                if adj[i,j] > 0:
                        lat2, lon2 = unscale_lat(pos[j, 0]), unscale_lon(pos[j, 1])
                        plt.plot([lon1, lon2], [lat1, lat2], 'g-')
        plt.plot(lon1, lat1, 'b^')

plt.savefig(os.path.join(folder, "../visu_graphe.jpeg"))

print(knn)

"

folder = "/home/noelee//EcoleEte_EDF_CEA_INRIA/data/LDG"
data_file = os.path.join(folder, "data-0.pkl")

minlatitude = 40
maxlatitude = 53
minlongitude = -6
maxlongitude = 11


(x_train, y_train, pos, dist) = pickle.load(open(os.path.join(folder, data_file), "rb"))

K = 5
knn = torch.topk(torch.Tensor(dist), K, largest=False)[1].numpy()
adj = np.zeros((len(pos), len(pos)))
for i, row in enumerate(knn):
    for j in row:
        adj[i, j] = 1

events = [1, 2, 3]
colors = ['r-', 'b-', 'g-', 'c-', 'm-', 'y-']
stations = ['AVF', 'BAIF', 'BGF', 'CABF', 'CAF', 'CDF']

plt.clf()
fig, axs = plt.subplots(6, 3, figsize = (16, 12))

for i in range(6):
        for j, event in enumerate(events):

                ax = axs[i, j]
                serie = x_train[event, i, :, 0]

                t = np.linspace(0, len(serie)-1, len(serie))
                ax.plot(t, serie, colors[i])

                ax.set_title(stations[i])

plt.savefig(os.path.join(folder, "visu_series.jpeg"))

plt.clf()
fig, ax = plt.subplots()

plt.plot(unscale_lon(y_train[event,3]), unscale_lat(y_train[event, 2]), 'ko-')
for i in range(6):
        plt.plot(unscale_lon(pos[i, 1]), unscale_lat(pos[i, 0]), colors[i], marker = '^')
        plt.text(unscale_lon(pos[i, 1]), unscale_lat(pos[i, 0]), stations[i])

plt.savefig(os.path.join(folder, "visu_map.jpeg"))

plt.clf()

for i in range(len(pos)):
        lat1, lon1 = unscale_lat(pos[i, 0]), unscale_lon(pos[i, 1])
        for j in range(len(pos)):
                if adj[i,j] > 0:
                        lat2, lon2 = unscale_lat(pos[j, 0]), unscale_lon(pos[j, 1])
                        plt.plot([lon1, lon2], [lat1, lat2], 'g-')
        plt.plot(lon1, lat1, 'b^')

plt.savefig(os.path.join(folder, "visu_graphe.jpeg"))

print(knn)
