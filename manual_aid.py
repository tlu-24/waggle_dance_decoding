# manual_aid.py
# This file provides methods and runs a script that creates a manifest file for "split.py"
# based on waggle detections from "DanceDetector.py" to help make manual decoding easier, also creates
# an info file about start times and video names of new videos.

# imports
import cv2
import numpy as np
import pandas as pd
import csv
import subprocess
import re
import math
import json
import os
import os.path
import argparse


# take inputs from the terminal
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", required=True,
                help="path to the input pkl waggle detections with clusters")
ap.add_argument("-o", "--outdir", required=True,
                help="out directory for the json manifest file")
ap.add_argument("-f", "--fps", type=int, required=True,
                help="the fps of the video")
ap.add_argument("-b", "--buffer", type=int, default=150, required=False,
                help="how many frames long the buffer on either end of the cluster")
args = vars(ap.parse_args())

FILENAME = args['input']
OUT_DIR = args['outdir']
OUT_PREFIX = FILENAME.split('-')[0].split('/')[-1]
BUFFER = args['buffer']  # frames
FPS = args['fps']


def make_ranges(starts, ends):
    """ 
    Takes the beginnings and ends of the clusters and creates and merges ranges that overlap in time
    (e.g. beginning of one is before end of another one) with BUFFER frames on either end.
        Inputs:
            - starts, list of the starting frames of each cluster
            - ends, list of the ending frames of each cluster
        Outputs: 
            - labels: list of names for new video clips, based on the start time of the clip
            - final_starts: list of starting frames for new cluster ranges
            - final_ends: list of starting frames for new cluster ranges
            - length: list of how long the new ranges are
    """
    final_starts = []
    final_ends = []
    length = []  # in seconds
    # name of output videos
    labels = [OUT_PREFIX + '_'+str(int(starts[0]/FPS))]
    current_start = starts[0]  # in frames
    final_starts.append(current_start-BUFFER/FPS)
    current_end = ends[0]

    # loop through the ranges, if there is overlap, add it to previous range, else
    # start a new one
    for si, s in enumerate(starts):
        # if overlap, update current_end to accommodate for buffer
        if s <= current_end + BUFFER:
            current_end = ends[si] + BUFFER
        # else start a new range, update current_end to end of previous range,
        # current_start to start of next range
        else:
            length.append(((current_end - current_start)+BUFFER)/FPS)
            final_ends.append(current_end)
            current_start = s - BUFFER
            current_end = ends[si] + BUFFER
            labels.append(OUT_PREFIX + '_' + str(int(current_start/FPS)))
            final_starts.append(current_start/FPS)
    final_ends.append(current_end)
    length.append(((current_end - current_start) + BUFFER)/FPS)
    return (labels, final_starts, final_ends, length)


# take input from DanceDetector.py
waggle_df = pd.read_pickle(FILENAME)
# Sort by cluster and then frame so the dataset is ordered in blocks of clusters
waggle_df = waggle_df.sort_values(
    by=['Cluster', 'frame']).reset_index().drop(['index'], axis=1)


cluster_ranges = []

clusters = []
coord_x = []
coord_y = []
names = []

print('number of clusters:', len(waggle_df['Cluster'].unique()))

# iterate through the clusters
for c in waggle_df['Cluster'].unique():

    # get info about cluster from dataframe
    clust = waggle_df[waggle_df['Cluster'] == c].reset_index()

    # Extract values from df
    start = clust.iloc[0, :]['frame']
    end = clust.iloc[-1, :]['frame']
    cluster = clust.iloc[0, :]['Cluster']

    # keep track of the clusters
    if c != -1:
        clusters.append(c)

        # where in space the cluster starts
        coord_x.append(clust.iloc[0, :]['x'])
        coord_y.append(clust.iloc[0, :]['y'])

    # Get range of frames where waggle occurs
    rang = np.arange(start, end, 1)
    cluster_ranges.append((c, (int(start), int(end))))


# loop through cluster ranges to get places to cut the video
starts = []
ends = []
label = []
for i, (c, (start, end)) in enumerate(cluster_ranges):
    starts.append(int(start))
    ends.append(int(end))
    label.append(c)

# get new ranges based on overlaps in time of clusters
labels, final_starts, final_ends, lengths = make_ranges(starts, ends)

# get which video clusters are a part of
for ci, c in enumerate(clusters):
    if c == -1:
        continue
    else:
        for si, s in enumerate(final_starts):
            if starts[ci] >= s and starts[ci] <= final_ends[si]:
                names.append(labels[si])
                break

# set up for dataframes
# has a row for each cluster: which (smaller) video its a part of, cluster #, what frame in the video
# it starts at, and the coordinates of the beginning of the cluster
info_dict = {'video': names, 'cluster': clusters,  'frame_start': starts[1:],
             'coord_x': coord_x, 'coord_y': coord_y}

out_dict = {'start_time': final_starts, 'length': lengths, 'rename_to': labels}


out_df = pd.DataFrame(out_dict)
info_df = pd.DataFrame(info_dict)

# create manifest file for split.py
out_df.to_json(OUT_DIR+'/'+'manual_detect_cuts-' +
               OUT_PREFIX+".json", orient='records')

# create the info file
info_df.to_csv(OUT_DIR+'/'+'manual_detect_info-' +
               OUT_PREFIX+".csv")
