import matplotlib
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from matplotlib import pyplot as plt

waggle_df = pd.read_csv(
    "/Users/mayaabodominguez/Desktop/BeeLab/WaggleDanceTracker/waggle_dance_decoding/Col4_061021_1215_testclusters_w_1contour.csv")


# fig, ax = plt.subplots(figsize=(8.8, 6.9), ncols=2,
#                        gridspec_kw={'width_ratios': [10, 1]})
fig, ax = plt.subplots(figsize=(8.8, 6.9))

cmap = matplotlib.cm.viridis

x = waggle_df[['x']]
y = waggle_df['y']
bounds = waggle_df['Cluster']
colors = ['tab:gray', 'tab:blue', 'tab:orange', 'tab:green',
          'tab:purple',  'tab:pink', 'tab:cyan', 'tab:olive', 'tab:brown', ]
shapes = ['.', '^', '+', '*', 'x', 's', 'd', 'X', 'p', 'P']
# norm = matplotlib.colors.Normalize(vmin=-1, vmax=26)
# fig.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap),
#              cax=ax[1], orientation='vertical', label='Cluster')

for i in range(len(waggle_df['dance'].unique())):
    if i == 0:
        label = 'detection not \nin bounds'
    else:
        label = 'dance ' + str(i)
    dance_df = waggle_df[waggle_df['dance'] == i]
    x = dance_df['x']
    y = dance_df['y']
    ax.scatter(x, y, color=colors[i % len(colors)],
               marker=shapes[i % len(shapes)], label=label, alpha=0.70)
ax.invert_yaxis()
ax.set_xlabel('x/pixels', fontsize=16)
ax.set_ylabel('y/pixels', fontsize=16)
ax.legend()
plt.legend(loc=2, prop={'size': 15})
for label in (ax.get_xticklabels() + ax.get_yticklabels()):
    label.set_fontsize(16)
ax.set_title('Detected Waggles Compared to Manual Detections', fontsize=24)
plt.savefig("ploty.png")
plt.show()
