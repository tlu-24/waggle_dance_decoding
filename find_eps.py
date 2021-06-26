import cv2
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from matplotlib import pyplot as plt
from sklearn.neighbors import NearestNeighbors
from kneed import KneeLocator


TIMESCALE = 2
SAVE = True
GRAPH = True
FILENAME = '/Users/mayaabodominguez/Desktop/Bee Lab/WaggleDanceTracker/smalltes2t-WaggleDetections.pkl'
LABEL = FILENAME.split('/')[-1].split('-')[0]
print(LABEL)

waggle_df = pd.read_pickle(FILENAME)

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
kneedle.plot_knee()

# save output from here
print(kneedle.knee_y)
eps = kneedle.knee_y

X = (waggle_df)
clust1 = DBSCAN(eps=eps, min_samples=6).fit(X)
waggle_df.loc[:, 'Cluster'] = clust1.labels_

if SAVE:
    distances_df.to_csv("NN_eps.csv")
    waggle_df.to_pickle(
        '{}-findepscluster_{}_{}.pkl'.format(LABEL, int(eps), 6))

# kneedle.plot()
# plot = distances_df.plot()
# figgy = plot.get_figure()
# figgy.savefig(LABEL+"knee")
