import matplotlib
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from matplotlib import pyplot as plt

waggle_df = pd.read_csv(
    "/Users/mayaabodominguez/Desktop/BeeLab/WaggleDanceTracker/waggle_dance_decoding/Col4_061021_1215_testclusters_w_1contour.csv")


fig, ax = plt.subplots(figsize=(8.8, 6.9), ncols=2,
                       gridspec_kw={'width_ratios': [10, 1]})

cmap = matplotlib.cm.viridis

x = waggle_df[['x']]
y = waggle_df['y']
bounds = waggle_df['Cluster']
norm = matplotlib.colors.Normalize(vmin=-1, vmax=26)
fig.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap),
             cax=ax[1], orientation='vertical', label='Cluster')

ax[0].scatter(x, y, c=bounds, cmap='viridis')
ax[0].invert_yaxis()
ax[0].set_xlabel('x/pixels', fontsize=16)
ax[0].set_ylabel('y/pixels', fontsize=16)
for label in (ax[0].get_xticklabels() + ax[0].get_yticklabels()):
    label.set_fontsize(16)
ax[0].set_title('Detected Waggles and Clusters', fontsize=24)
plt.savefig("ploty.png")
plt.show()
