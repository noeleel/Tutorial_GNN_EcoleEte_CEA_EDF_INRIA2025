import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.geodesic import Geodesic
import numpy as np
import pickle, os, torch
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab

params = {'legend.fontsize': 'xx-large',       # Taille des légences
          'figure.figsize': (15, 5),           # Taille initiale des figures
         'axes.labelsize': 'xx-large',         # Taille des titres
         'axes.titlesize':'xx-large',          # Taille des titres de figures
         'xtick.labelsize':'xx-large',         # Taille des titres des abscisses
         'ytick.labelsize':'xx-large',         # Taille des titres des ordonnées
         'font.size' : 10.0,                   # Taille du texte
         'font.family': 'DejaVu Sans',         # Paramètre 1 de police du texte
         'font.serif': 'Times New Roman',      # Paramètre 2 de police du texte
         'font.style': 'normal',               # Paramètre 3 de police du texte
         'xtick.direction' : 'inout',          # Sens des abscisses
         'ytick.direction' : 'inout',          # Sens des ordonnées
         'xtick.major.size' : 5,               # Espacement entre les abscisses
         'ytick.major.size' : 5}               # Espacement entre les ordonnées

pylab.rcParams.update(params)                  # Mise à jour globale


for network in ["SCDEC", "LDG"]:
    folder = f"/home/noelee/ENE/GNN/EcoleEte_EDF_CEA_INRIA/data/{network}"
    data_file = "data-0.pkl"
    (x_train, y_train, pos, dist) = pickle.load(open(os.path.join(folder, data_file), "rb"))

    event = 10
    if network == "LDG":
        minlatitude = 40
        maxlatitude = 53
        minlongitude = -6
        maxlongitude = 11
    else:
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
    
        
    # Sample station data: list of tuples (name, lat, lon)
    colors = ['r-', 'b-', 'g-', 'c-', 'm-', 'y-']
    
    if network == "LDG":   
        stations = ['AVF', 'BAIF', 'BGF', 'CABF', 'CAF', 'CDF']
    else:
        stations = ['ADO', 'ALP', 'ARV', 'AVM', 'BAK', 'BAR']


    #               x_train             y_train
    # Dimension 0 : évènements          évènements
    # Dimension 1 : directions (3)      labels (4)
    # Dimension 2 : stations (50)       -
    # Dimension 3 : temps (2048)        -
    plt.clf()

    dict_components = {0 : "North-South",
                      1 : "East-West",
                      2 : "Up-Down"}
    if network == "SCDEC":
        plt.clf()
        fig, axs = plt.subplots(2, 3, figsize = (18,10), sharex=True)
        for i in range(2):
            for j in range(3):
                    ax = axs[i, j]
                    print(x_train.shape)
                    serie = x_train[event, i+1, :, j]
    
                    t = np.linspace(0, len(serie)-1, len(serie))
                    ax.plot(t, serie, colors[i+1])
    
                    ax.set_title(f"{stations[i+1]} - {dict_components[j]}")
                    ax.set_ylabel("Normalized Response")
                    ax.set_xlabel("Timesteps")
                    ax.set_ylim([-0.03, 0.03])
    else:
        plt.clf()
        fig, axs = plt.subplots(1,2, figsize = (18,5))
        for i in range(2):
            ax = axs[i]
            print(x_train.shape)
            serie = x_train[event, i, :, 0]

            t = np.linspace(0, len(serie)-1, len(serie))
            ax.plot(t, serie, colors[i])

            ax.set_title(f"{stations[i]}")
            ax.set_ylabel("Normalized Response")
            ax.set_xlabel("Timesteps")
            ax.set_ylim([-0.03, 0.03])
        
    plt.savefig(f"../{network}_visu_series.jpeg")


    # Event location (lat, lon)
    if network == "LDG":
        event_lat, event_lon = unscale_lat(y_train[event, 3]), unscale_lon(y_train[event,2])
    else:
        event_lat, event_lon = unscale_lat(y_train[event, 1]), unscale_lon(y_train[event,2])
    
    # Extract lats and lons from stations
    lats = [unscale_lat(pos[i, 0]) for i in range(len(stations))]
    lons = [unscale_lon(pos[i, 1]) for i in range(len(stations))]
    
    # Determine extent (min_lon, max_lon, min_lat, max_lat) with some padding
    extent = [minlongitude, maxlongitude,
              minlatitude, maxlatitude]
    
    lon_min, lon_max, lat_min, lat_max = extent
    lon_span = lon_max - lon_min
    lat_span = lat_max - lat_min
    
    # Aspect ratio = width / height = lon_span / lat_span
    fig_width = 10  # fixed width
    fig_height = 3 * (lat_span / lon_span)
    
    # Create figure and axis with Cartopy projection
    fig = plt.figure(figsize=(8,6))
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.set_extent(extent)
    ax.set_aspect('auto')
    
    # Add map features
    ax.add_feature(cfeature.LAND.with_scale('50m'))
    ax.add_feature(cfeature.OCEAN.with_scale('50m'))
    ax.add_feature(cfeature.COASTLINE.with_scale('50m'))
    ax.add_feature(cfeature.BORDERS.with_scale('50m'), linestyle=':')
    
    # Plot seismic stations                                                                   
    for i in range(len(stations)):
            plt.plot(unscale_lon(pos[i, 1]), unscale_lat(pos[i, 0]), 'r^', markersize=12, transform=ccrs.PlateCarree())
            plt.text(unscale_lon(pos[i, 1]) + 0.1, unscale_lat(pos[i, 0]) +0.1, stations[i], transform=ccrs.PlateCarree())
    
    plt.plot(unscale_lon(pos[i, 1]), unscale_lat(pos[i, 0]), 'r^', markersize=12, transform=ccrs.PlateCarree(), label="Station")
                                                         
    # Plot seismic event
    ax.plot(event_lon, event_lat, 'b*', markersize=12, label='Event', transform=ccrs.PlateCarree())
    
    # Add gridlines with labels
    gl = ax.gridlines(draw_labels=True, linewidth=1, color='gray', alpha=0.5, linestyle='--')
    
    # Configure which labels to show
    gl.top_labels = False
    gl.right_labels = False
    gl.left_labels = True
    gl.bottom_labels = True
    
    # Optionally, customize label style
    gl.xlabel_style = {'size': 10, 'color': 'gray'}
    gl.ylabel_style = {'size': 10, 'color': 'gray'}
    
    
    plt.legend()
    plt.title("Seismic Stations and Event with Wave Propagation")
    plt.savefig(f"../{network}_Seismic_map.png")
    #plt.show()
    
    
    ####################################################################################
    
    fig = plt.figure(figsize=(8,6))
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.set_extent(extent)
    ax.set_aspect('auto')
    
    # Add map features
    ax.add_feature(cfeature.LAND.with_scale('50m'))
    ax.add_feature(cfeature.OCEAN.with_scale('50m'))
    ax.add_feature(cfeature.COASTLINE.with_scale('50m'))
    ax.add_feature(cfeature.BORDERS.with_scale('50m'), linestyle=':')
    gl = ax.gridlines(draw_labels=True, linewidth=1, color='gray', alpha=0.5, linestyle='--')
    
    
    K = 5
    knn = torch.topk(torch.Tensor(dist), K, largest=False)[1].numpy()
    
    adj = np.zeros((len(pos), len(pos)))
    for i, row in enumerate(knn):
        for j in row:
            adj[i, j] = 1
            
    print(event_lon, event_lat)
    ax.plot(event_lon, event_lat, 'b*', markersize=12, label='Event', transform=ccrs.PlateCarree())
    
    for i in range(len(pos)):
        lat1, lon1 = unscale_lat(pos[i, 0]), unscale_lon(pos[i, 1])
        for j in range(len(pos)):
                if adj[i,j] > 0:
                        lat2, lon2 = unscale_lat(pos[j, 0]), unscale_lon(pos[j, 1])
                        ax.arrow(lon1, lat1, lon2 - lon1, lat2 - lat1, color='g', shape='full', lw=0, length_includes_head=True, head_width=.05)
                        ax.plot([lon1, lon2], [lat1, lat2], 'g-', transform=ccrs.PlateCarree())
        ax.plot(lon1, lat1, 'r^', markersize=12, transform=ccrs.PlateCarree())
            
    
    plt.savefig(f"../{network}_visu_graphe.jpeg")
plt.show()