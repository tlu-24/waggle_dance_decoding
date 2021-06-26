import cv2
import numpy as np
import pandas as pd

waggle_df = pd.read_pickle(
    '/Users/mayaabodominguez/Desktop/Bee Lab/WaggleDanceTracker/smalltes2t-findepscluster_19_6.pkl')
path = '/Users/mayaabodominguez/Desktop/Bee Lab/WaggleDanceTracker/split_vid/smalltes2t.mp4'
PRINT = False
SAVE = True
OUT_PREFIX = 'vis'
LABEL = path.split('/')[-1].split('.')[0].split('_')
VIS = True

# read video
cap = cv2.VideoCapture(path)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# create output video
if SAVE:
    out = cv2.VideoWriter(OUT_PREFIX+'.mp4', cv2.VideoWriter_fourcc(
        'm', 'p', '4', 'v'), 10, (width, height))
counter = 0
prev_frame = None

# get info from df about current frame
f_counter = 0
current_frame = waggle_df.iloc[f_counter]

# colors to draw the dots in
rgb = [(252, 186, 3), (252, 57, 3), (111, 252, 3), (3, 252, 194),
       (245, 66, 227), (153, 112, 149), (204, 116, 116), (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (0, 0, 255)]

# video processing loop
while True:
    counter += 1
    ret, frame = cap.read()

    # Break when video ends
    if ret is False:
        break

    # loop through the frames and color a dot based on cluster on frames
    # where a waggle is potentially detected
    while counter == current_frame['frame']:
        if PRINT:
            print("x,y,cluster", (int(current_frame['x']),
                                  int(current_frame['y'])), current_frame['Cluster'])
        if current_frame['Cluster'] == -1.0:
            cv2.circle(frame, (int(current_frame['x']),
                               int(current_frame['y'])), 10, (0, 0, 0), -1)
        else:
            cv2.circle(frame, (int(current_frame['x']),
                               int(current_frame['y'])), 10, rgb[int(current_frame['Cluster'])-1], -1)
        f_counter += 1
        if f_counter >= len(waggle_df):
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
# save video here


cap.release()
if SAVE:
    out.release()
cv2.destroyAllWindows()
