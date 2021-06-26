import cv2
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from matplotlib import pyplot as plt
from sklearn.neighbors import NearestNeighbors

LABEL = 'maya'
TIMESCALE = 2
GRAPH = True
SEPARATE = False
EPS = 116
MINPTS = 6

# read potential waggles
waggle_df = pd.read_pickle('potential_waggles.pkl')
waggle_df_oldtime = waggle_df

# Scale time
# waggle_df['frame'] = waggle_df['frame']*TIMESCALE

# cluster the potential waggles
X = (waggle_df)
clust1 = DBSCAN(eps=63, min_samples=6).fit(X)
waggle_df.loc[:, 'Cluster'] = clust1.labels_
waggle_df_oldtime.loc[:, 'Cluster'] = clust1.labels_

# sort by cluster
waggle_df = waggle_df.sort_values(
    by=['Cluster', 'frame']).reset_index().drop(['index'], axis=1)

# plot by cluster
plot = waggle_df.plot.scatter(
    x="x", y="y", c="Cluster", cmap="viridis")
fig = plot.get_figure()
fig.savefig(LABEL+".png")

# plot all the clusters as their own graph, color-coded by frame, in a giant graph
if GRAPH:
    # get all the clusters #s
    num_clust = len(waggle_df['Cluster'].unique())
    print(num_clust)

    # create a giant plot for the subplots to go on
    fig, axes = plt.subplots(nrows=num_clust//3+1, ncols=3, figsize=(
        15, 15), sharex=True, sharey=True)
    plt.subplots_adjust(wspace=0.5, hspace=0.5)

    # plot each cluster
    for c in waggle_df['Cluster'].unique():
        cluster_df = waggle_df[waggle_df['Cluster'] == c]
        # print(c, "\n", cluster_df)
        i = c % 3
        j = c//3
        if c == -1:
            i = -1
            j = -1
        print(c, i, j)
        # cluster_df.plot(ax=axes[j, i], x='x', y='y', cmap='viridis')
        cluster_df.plot.scatter(
            ax=axes[j, i], x='x', y='y', c='frame', cmap='viridis')

        if SEPARATE:
            plot = cluster_df.plot.scatter(
                x='x', y='y', c='frame', cmap='viridis')
            plot.set_title('Cluster' + str(c))
            plt.ylim(0, 850)
            # corresponding function for the x-axis
            plt.xlim(0, 1200)
            fig1 = plot.get_figure()
            fig1.savefig("./cluster_plots/cluster_"+LABEL + str(c)+".png")
        axes[j, i].set_title('Cluster' + str(c))
        axes[j, i].invert_yaxis()

    plot.set_title('Cluster' + str(c))
    # fig = axs.get_figure()
    # ploty = waggle_df.plot.scatter(
    #     x="x", y="y", c="Cluster", cmap="viridis").invert_yaxis()
    # figgy = ploty.get_figure()
    # figgy.savefig("./cluster_plots/FULL_clusters_"+LABEL+".png")
    # fig.savefig("./cluster_plots/clusters_"+LABEL+".png")
    fig.savefig("./cluster_plots/AHHH.png")

# save cluster data
waggle_df.to_pickle('Clustering_{}_{}.pkl'.format(56, 6))
waggle_df_oldtime.to_pickle("frame.pkl")
waggle_df.to_csv('wagggle_detections.csv')
