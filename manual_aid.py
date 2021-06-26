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
BUFFER = 44  # frames
FPS = 24
BOX_SIZE = 100  # pt
DRAW_BOX = True

# this function takes 2 lists of frames to start and end
# and does the appropriate changes with correct overlap to output
# info for the json file (start, length, new label)


def make_ranges(starts, ends):
    final_starts = []
    final_ends = []
    length = []
    labels = ['segment0']
    current_start = starts[0]
    final_starts.append(current_start)
    current_end = ends[0]

    for si, s in enumerate(starts):
        if s <= current_end + BUFFER:
            current_end = ends[si]
        else:
            length.append(((current_end - current_start)/FPS))
            final_ends.append(current_end)
            labels.append(OUT_PREFIX + str(si))
            current_end = ends[si]
            current_start = s
            final_starts.append((current_start/FPS))
    final_ends.append(current_end)
    length.append(((current_end - current_start)/FPS))
    return (labels, final_starts, length)


# take input from DanceDetector.py
waggle_df = pd.read_pickle(FILENAME)
# Sort by cluster and then frame so the dataset is ordered in blocks of clusters
waggle_df = waggle_df.sort_values(
    by=['Cluster', 'frame']).reset_index().drop(['index'], axis=1)

# open(OUT_DIR + 'cluster' + c, 'x', )

# do some sort of clustering, probably just use the cluster number to get start and stop frames


cluster_ranges = []

# iterate through the clusters
for c in waggle_df['Cluster'].unique():
    # write to file a new line indicating new cluster
    out = open(OUT_DIR + 'cluster' + str(c) + '.csv', 'w')
    out.write('Cluster ' + str(c) + '\n' +
              '==================================\n')

    clust = waggle_df[waggle_df['Cluster'] == c].reset_index()
    clust.to_csv(out.name)
    if c == -1:
        # write to file the "noisy frames" using clust
        out.writelines(['========================================\n',
                        'noisy frames\n', '=================================\n'])
        continue
    # Extract values from df
    start = clust.iloc[0, :]['frame']
    # start = df.iloc[0, :]
    end = clust.iloc[-1, :]['frame']
    cluster = clust.iloc[0, :]['Cluster']
    # Get range of frames where waggle occurs
    rang = np.arange(start, end, 1)
    cluster_ranges.append((c, (start, end)))
    out.close()

# loop through cluster ranges to get places to cut the video
starts = []
ends = []
label = []
for i, (c, (start, end)) in enumerate(cluster_ranges):
    starts.append(int(start))
    ends.append(int(end))
    label.append(c)

# print("start", starts, "end", ends)
labels, final_starts, lengths = make_ranges(starts, ends)

out_dict = {'start_time': final_starts, 'length': lengths, 'rename_to': labels}
# print(len(out_dict['start_time']), len(
#     out_dict['length']), len(out_dict['rename_to']))
out_df = pd.DataFrame(out_dict)

# out_df.to_csv(OUT_DIR+"video_cuts1.csv")

# create manifest file for split.py
out_df.to_json(OUT_DIR+OUT_PREFIX+"_video_cuts.json", orient='records')


# maybe do some sort of visual cue ? have it turn on and off-able
