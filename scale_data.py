# scale_data.py
# utility file to convert data (coordinates) to a different resolution

import pandas as pd


ORIGINAL_W = 3840
ORIGINAL_H = 2160

NEW_W = 1920
NEW_H = 1080

FILENAME = '/Users/mayaabodominguez/Desktop/BeeLab/WaggleDanceTracker/waggle_dance_decoding/Col4_061021_1215_C0005_segment-findepscluster_77_6.pkl'


df = pd.read_pickle(FILENAME)

df_x = pd.Series(df['x'], name='norm_x')
df_y = pd.Series(df['y'], name='norm_y')

norm_x = df_x/ORIGINAL_W
norm_y = df_y/ORIGINAL_H

df['norm_x'] = norm_x
df['norm_y'] = norm_y

df['1080_x'] = norm_x*NEW_W
df['1080_y'] = norm_y*NEW_H

df.to_csv("Col4_061021_1215_C0005_segment-findepscluster_77_6_1080pscaled.csv")
