# find_eps.py
# uses the k-Nearest Neighbors method to find epsilon for DBSCAN and then clusters
# waggle detections based on calculated epsilon and minPts = 6

import cv2
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
    kneedle.plot_knee()
    plot = distances_df['distance'].plot().set_title(
        "6 Nearest Neighbors for Waggle Detections")
    plt.savefig(LABEL+"knee")

# get epsilon
print("epsilon is calculated to", kneedle.knee_y)
eps = kneedle.knee_y

# perform clustering using calculated eps and minPts=6
X = (waggle_df)
clust1 = DBSCAN(eps=eps, min_samples=6).fit(X)

# add clusters to df
waggle_df.loc[:, 'Cluster'] = clust1.labels_

print(len(waggle_df['Cluster'].unique()), "clusters found")

if GRAPH:
    ploty = waggle_df.plot.scatter(
        x="x", y="y", c="Cluster", cmap="viridis", title="Detected Waggles and Clusters").invert_yaxis()
    plt.savefig("FULL_clusters_"+LABEL+".png")

if SAVE:
    distances_df.to_csv("NN_eps.csv")
    waggle_df.to_pickle(
        '{}-findepscluster_{}_{}.pkl'.format(LABEL, int(eps), 6))
