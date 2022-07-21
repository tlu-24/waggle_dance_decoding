# dance_vis.py
# This file creates a visualization for clustered waggle detection data on the video it was taken from.
# it currently doesn't work with the cleaned data for some reason, not sure why.

import cv2
import numpy as np
import pandas as pd
import sys
import argparse

# take input
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", required=True,
                help="path to the clustered waggle detections .pkl file")
ap.add_argument("-i2", "--input2", required=True,
                help="path to the clustered waggle detections .pkl file with more information")
ap.add_argument("-v", "--video", required=True,
                help="path to the video")
ap.add_argument("-s", "--save", type=bool, default=True, required=False,
                help="save the output, default = True")
ap.add_argument("-z", "--Visualize", type=bool, default=False, required=False,
                help="show visualization")
args = vars(ap.parse_args())

waggle_df = pd.read_csv(args['input'])
waggle_df_cluster = pd.read_csv(args['input2'])
path = args['video']
PRINT = True #args['Visualise']
SAVE = args['save']
LABEL = path.split('/')[-1].split('.')[0].split('_')
VIS = True #args['Visualise']
waggle_df = waggle_df.sort_values('frame')
waggle_df['x'] = (waggle_df['x']*1920).astype(int)
waggle_df['y'] = (waggle_df['y']*1080).astype(int)
waggle_df = waggle_df.loc[waggle_df['Cluster'] != -1]
waggle_df_cluster = waggle_df_cluster.sort_values('f1')
waggle_df_cluster['x1'] = waggle_df_cluster['x1']*1920#3840
waggle_df_cluster['y1'] = waggle_df_cluster['y1']*1080#2160
waggle_df_cluster['x2'] = waggle_df_cluster['x2']*1920#3840
waggle_df_cluster['y2'] = waggle_df_cluster['y2']*1080#2160

# read video
cap = cv2.VideoCapture(path) 
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

if cap.isOpened():
    print("successfully read video")
else:
    print("video not found")

# create output video
if SAVE:
    out = cv2.VideoWriter(str(args['input'])+'.mp4', cv2.VideoWriter_fourcc(
        'm', 'p', '4', 'v'), 30, (width, height))
counter = 0
prev_frame = None

# get info from df about current frame
f_counter = 0
current_frame = waggle_df.iloc[f_counter]
frameFirst = 0

# colors to draw the dots in
rgb = [(252, 186, 3), (252, 57, 3), (111, 252, 3), (3, 252, 194),
       (245, 66, 227), (153, 112, 149), (204, 116, 116), (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (0, 0, 255)]

# print("Starting visualization")
# video processing loop
while False:
    counter += 1
    ret, frame = cap.read()
    # Break when video ends
    if ret is False:
        break

    # loop through the frames and color a dot on frames
    # where a waggle is potentially detected
    while counter == current_frame['frame']:
        if PRINT:
            print("x,y,cluster", (int(current_frame['x']),
                                  int(current_frame['y'])), current_frame['Cluster'])
        if current_frame['Cluster'] == -1.0:
            cv2.circle(frame, (int(current_frame['x']),
                               int(current_frame['y'])), 10, (0, 0, 0), -1)
            cv2.circle(frame, (int(current_frame['x']),
                               int(current_frame['y'])), 10, (0, 0, 0), -1)
        else:
            # cv2.circle(frame, (int(current_frame['x']),
            #                    int(current_frame['y'])), 10, rgb[(int(current_frame['Cluster'])-1) % len(rgb)], -1)
            # cv2.circle(frame, (int(current_frame['x']),
            #                    int(current_frame['y'])), 10, rgb[(int(current_frame['Cluster'])-1) % len(rgb)], -1)

            cv2.drawContours(frame, waggle_df.iloc[f_counter]['contour'], -1, (0,255,0), 6)
        f_counter += 1
        if f_counter >= len(waggle_df) or counter >= len(waggle_df):
            break
        current_frame = waggle_df.iloc[f_counter]
    # add frame to output video
    if SAVE:
        out.write(frame)

    if VIS:
        # show the video
        cv2.imshow('frame', frame)

        # quit if user presses 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    if counter % 100 == 0:
        print(counter)

cap.release()
if SAVE:
    out.release()

top = []
bottom = []
left = []
right = []
frame1 = []
frame2 = []
# Iterating through the columns of
# dataframe
for index, row in waggle_df_cluster.iterrows():
    # Storing the rows of a column
    # into a temporary list
    li = row.tolist()
    # appending the temporary list

    left.append(int(li[1]))
    right.append(int(li[2]))
    bottom.append(int(li[3]))
    top.append(int(li[4]))
    frame1.append(li[5])
    frame2.append(li[6])
parts = list(zip(left, right, bottom, top, frame1, frame2))

cap = cv2.VideoCapture(path)#str(args['input'])+'.mp4')
ret, frame = cap.read()
h, w, _ = frame.shape

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
writers = [cv2.VideoWriter(f"Negative-part{start}-{end}.mp4",  fourcc, 30.0, (100,100)) for left, right, bottom, top, start, end in parts]

for i, part in enumerate(parts):
    left, right, bottom, top, start, end = part
    cap = cv2.VideoCapture(path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start-1)

    f = start
    while True:
        res, frame = cap.read()
        f += 1
        if f == end:
            break
        frame = frame[(bottom+top)//2-50:(bottom+top)//2+50, (left+right)//2-50:(left+right)//2+50, :]
        writers[i].write(frame)





for writer in writers:
    writer.release()

cap.release()

cv2.destroyAllWindows()
print("completed.")