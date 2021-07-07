import cv2
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from matplotlib import pyplot as plt
from sklearn.neighbors import NearestNeighbors
from kneed import KneeLocator
import argparse

TIMESCALE = 2
SAVE = True
GRAPH = True
FILENAME = '/Users/mayaabodominguez/Desktop/Bee Lab/WaggleDanceTracker/smalltes2t-WaggleDetections.pkl'


# take input as a video
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", required=True,
                help="path to the waggle detections .pkl file")
ap.add_argument("-s", "--save", type=bool, default=True, required=False,
                help="save the output, default = True")
ap.add_argument("-g", "--graph", type=bool, default=False, required=False,
                help="graph the output, default = False")
args = vars(ap.parse_args())


SAVE = args['save']  # not sure if this handles no input
GRAPH = args['graph']
FILENAME = args['input']
LABEL = FILENAME.split('/')[-1].split('-')[0]
print(LABEL)


waggle_df = pd.read_pickle(FILENAME)
print(waggle_df.shape)

# waggle_df['frame'] = waggle_df['frame']*TIMESCALE

# do nearest neighbors
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
# these values might need to be changed based on data ... ? maybe not though
kneedle = KneeLocator(distances_df['A'], distances_df['distance'], S=10.0,
                      curve="convex", direction="increasing", interp_method='interp1d')
if GRAPH:
    kneedle.plot_knee()
    plot = distances_df['distance'].plot().set_title(
        "6 Nearest Neighbors for Waggle Detections")
    plt.savefig(LABEL+"knee")

# save output from here
print(kneedle.knee_y)
eps = kneedle.knee_y

X = (waggle_df)
clust1 = DBSCAN(eps=eps, min_samples=6).fit(X)
waggle_df.loc[:, 'Cluster'] = clust1.labels_

print(waggle_df['Cluster'].unique())

if GRAPH:
    # waggle_df['Cluster'] = waggle_df['Cluster'].astype('category')
    ploty = waggle_df.plot.scatter(
        x="x", y="y", c="Cluster", cmap="viridis", title="Detected Waggles and Clusters").invert_yaxis()
    plt.savefig("FULL_clusters_"+LABEL+".png")

if SAVE:
    distances_df.to_csv("NN_eps.csv")
    waggle_df.to_pickle(
        '{}-findepscluster_{}_{}.pkl'.format(LABEL, int(eps), 6))

# kneedle.plot()
