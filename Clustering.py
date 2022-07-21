from typing import List
import numpy as np
import pandas as pd
import argparse
from sklearn.cluster import DBSCAN

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", required=True,
                help="path to the input file with the unclustered detections(from dance_detector.py")
args = vars(ap.parse_args())

FILENAME = args['input']
df = pd.read_csv(FILENAME)
df = df.copy()
df['drop'] = False


for index, row in df.iterrows():
    # drop rows that don't meet a certian condititon
    if row['size']/row['sizeRect'] < 0.15:
        df.at[index, 'drop'] = True
    elif row['bases']/row['height'] > 3.75:
        df.at[index, 'drop'] = True
df = df[df['drop'] != True]
df = df.drop(columns = 'drop')

# Creating an empty list
x = []
y = []
frame = []
slope = []
# Iterating through the columns of
# dataframe
for index, row in df.iterrows():
    # Storing the rows of a column
    # into a temporary list
    li = row.tolist()
    # appending the temporary list

    x.append(li[1])
    y.append(li[2])
    frame.append(li[3])

waggle_dance = list(zip(x, y, frame))
X = (waggle_dance)
# Clusters the detections together based on the variables in waggle_dance
clust1 = DBSCAN(eps=25, min_samples=12).fit(X)
df.loc[:, 'Cluster'] = clust1.labels_

df.to_csv('{}-WaggleDetectionsClustered.pkl'.format(FILENAME.split('.')[0]))
