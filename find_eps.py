# find_eps.py
# uses the k-Nearest Neighbors method to find epsilon for DBSCAN and then clusters
# waggle detections based on calculated epsilon and minPts = 6

import cv2
import matplotlib
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from matplotlib import pyplot as plt
from sklearn.neighbors import NearestNeighbors
from kneed import KneeLocator
import argparse


# take input as a video
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", required=True,
                help="path to the waggle detections .pkl file")
ap.add_argument("-s", "--save", type=bool, default=True, required=False,
                help="save the output, default = True")
ap.add_argument("-g", "--graph", type=bool, default=False, required=False,
                help="graph the output, default = False")
args = vars(ap.parse_args())


SAVE = args['save']
GRAPH = args['graph']
FILENAME = args['input']
LABEL = FILENAME.split('/')[-1].split('-')[0]
MINPTS = 6  # chosen to be 2*dimensionality of data based on https://www.ccs.neu.edu/home/vip/teach/DMcourse/2_cluster_EM_mixt/notes_slides/revisitofrevisitDBSCAN.pdf

# read in waggle detections
waggle_df = pd.read_pickle(FILENAME)


# do 6-nearest neighbors analysis
X = (waggle_df)
neighbors = NearestNeighbors(n_neighbors=6)
neighbors_fit = neighbors.fit(X)

# get and sort distances
distances, indices = neighbors_fit.kneighbors(X)
distances = np.sort(distances, axis=0)
distances = distances[:, 1]
distances_df = pd.DataFrame(distances, columns=['distance'])
distances_df['A'] = list(range(len(distances_df.index)))

# locate the "knee"/best value for the eps value
# S might need to be changed based on data
kneedle = KneeLocator(distances_df['A'], distances_df['distance'], S=10.0,
                      curve="convex", direction="increasing", interp_method='interp1d')

# save a plot of the knee
if GRAPH:
    plt.rcParams['font.size'] = '18'
    kneedle.plot_knee()
    plot = distances_df['distance'].plot().set_title(
        "Determining Epsilon Using 6 Nearest Neighbors", fontsize=24)
    plt.savefig(LABEL+"knee")

# get epsilon
print("epsilon is calculated to", kneedle.knee_y)
eps = kneedle.knee_y

# perform clustering using calculated eps and minPts=6
X = (waggle_df)
clust1 = DBSCAN(eps=eps, min_samples=MINPTS).fit(X)

# add clusters to df
waggle_df.loc[:, 'Cluster'] = clust1.labels_

print(len(waggle_df['Cluster'].unique()), "clusters found")

if GRAPH:
    fig, ax = plt.subplots(figsize=(8.8, 6.9), ncols=2,
                           gridspec_kw={'width_ratios': [10, 1]})

    cmap = matplotlib.cm.viridis
    x = waggle_df['x']
    y = waggle_df['y']
    bounds = waggle_df['Cluster']
    norm = matplotlib.colors.Normalize(vmin=-1, vmax=26)
    fig.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap),
                 cax=ax[1], orientation='vertical', label='Cluster')

    ax[0].scatter(x, y, c=bounds, cmap='viridis')
    ax[0].invert_yaxis()
    ax[0].set_xlabel('x/pixels', fontsize=16)
    ax[0].set_ylabel('y/pixels', fontsize=16)
    for label in (ax[0].get_xticklabels() + ax[0].get_yticklabels()):
        label.set_fontsize(16)
    ax[0].set_title('Detected Waggles and Clusters', fontsize=24)
    plt.savefig("FULL_clusters_"+LABEL+".svg")
    plt.show()

if SAVE:
    distances_df.to_csv("NN_eps.csv")
    waggle_df.to_pickle(
        '{}-findepscluster_{}_{}.pkl'.format(LABEL, int(eps), 6))
