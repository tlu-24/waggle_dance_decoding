import pandas as pd
import numpy as np


def cluster_separate(cluster):
    cluster = cluster.reset_index()
    # get the direction of the first waggle
    print(cluster.iloc[0])
    prev_vec = np.array(
        cluster.iloc[1].x-cluster.iloc[0].x, cluster.iloc[1].y-cluster.iloc[0].y)

    new_cluster_indices = []
    current_cut = cluster.iloc[1].frame
    prev_i = 0
    for i in range(1, len(cluster)):

        # get direction of current waggle
        curr_vec = np.array(cluster.iloc[i].x-cluster.iloc[prev_i].x,
                            cluster.iloc[i].y-cluster.iloc[prev_i].y)

        # print(curr_vec.dot(prev_vec))

        # if it changed directions, make a new cluster
        if curr_vec.dot(prev_vec) < 0:  # change
            new_cluster_indices.append((current_cut, cluster.iloc[i].frame-1))
            # print((current_cut, cluster.iloc[i].frame-1))
            current_cut = cluster.iloc[i].frame
    return new_cluster_indices


waggle_df = pd.read_pickle(
    "/Users/mayaabodominguez/Desktop/Bee Lab/WaggleDanceTracker/waggle_dance_decoding/Col3_061021_1124_C0011_segment-WaggleDetections_Cleaned.pkl")

waggle_df = waggle_df.sort_values(
    by=['Cluster', 'frame']).reset_index().drop(['index'], axis=1)

cluster_counter = 0

for i in list(waggle_df['Cluster'].unique()):
    if i > 3:
        break
    if i == -1:
        continue
    print('Cluster', i, "=======================")
    clust = waggle_df[waggle_df['Cluster'] == i]
    print(clust)

    new_breaks = cluster_separate(clust)
    print(new_breaks)

    for start, end in new_breaks:
        index = clust[clust['frame'] <= start].index
        # print(clust.loc[index, 'Cluster'])
        clust.loc[index, 'Cluster'] = cluster_counter
        cluster_counter += 1

    print(clust)
