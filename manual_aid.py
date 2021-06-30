# This file provides methods and runs a script that segments a video based on waggle detections from "DanceDetector.py"
# to help make manual decoding easier.

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

FILENAME = 'WaggleDetections-.pkl'
DRAW_BOXES = False
OUT_DIR = './manual_detection_aid/'
OUT_PREFIX = 'segment'
BUFFER = 100  # frames
FPS = 24

# take inputs
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

# this function takes 2 lists of frames to start and end
# and does the appropriate changes with correct overlap to output
# info for the json file (start, length, new label)


def make_ranges(starts, ends):
    final_starts = []
    final_ends = []
    length = []  # in seconds
    labels = [OUT_PREFIX + '_'+str(int(starts[0]/FPS))]
    current_start = starts[0]
    final_starts.append(current_start-BUFFER/FPS)
    current_end = ends[0]

    # loop through the ranges, if there is overlap, add it to previous range, else
    # start a new one
    for si, s in enumerate(starts):
        if s <= current_end + BUFFER:
            current_end = ends[si] + BUFFER
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


# do some sort of clustering, probably just use the cluster number to get start and stop frames

# need to add time handling!!!
cluster_ranges = []

clusters = []
coord_x = []
coord_y = []
names = []

# throw an error here if there is only one cluster
print('number of clusters:', len(waggle_df['Cluster'].unique()))

# iterate through the clusters
for c in waggle_df['Cluster'].unique():

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
        coord_y.append(clust.iloc[0, :]['x'])

    # Get range of frames where waggle occurs
    rang = np.arange(start, end, 1)
    cluster_ranges.append((c, (int(start), int(end))))
    # out.close()

# loop through cluster ranges to get places to cut the video
starts = []
ends = []
label = []


for i, (c, (start, end)) in enumerate(cluster_ranges):
    starts.append(int(start))
    ends.append(int(end))
    label.append(c)


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


info_dict = {'video': names, 'cluster': clusters,  'frame_start': starts[1:],
             'coord_x': coord_x, 'coord_y': coord_y}

out_dict = {'start_time': final_starts, 'length': lengths, 'rename_to': labels}

print(len(names), len(clusters), len(starts), len(coord_x), len(coord_y))

out_df = pd.DataFrame(out_dict)
info_df = pd.DataFrame(info_dict)

# create manifest file for split.py
out_df.to_json(OUT_DIR+'/'+'manual_detect_cuts-' +
               OUT_PREFIX+".json", orient='records')

# create the info file
info_df.to_csv(OUT_DIR+'/'+'manual_detect_info-' +
               OUT_PREFIX+".csv")
