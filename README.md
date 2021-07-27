# automatic_dance_decoding

automatic_dance_decoding is a modified version of [@Jreese18](https://github.com/Jreece18)'s [WaggleDanceTracker](https://github.com/Jreece18/WaggleDanceTracker) to create a generalized approach to automatic dance decoding. It uses foreground detection techniques to detect the waggle dances, and then tracking techniques to get the orientation and duration of each waggle run. 

## Installation

Clone by running the following command in your desired directory

```bash
git clone https://github.com/beelabhmc/waggle_dance_decoding.git
```

## Requirements
This code uses Python 3.7+. For more information on checking your python version and upgrading or installing see [here](https://realpython.com/installing-python/). Other requirements are below. I recommend using pip or Conda or another package manager to install the requirements.
- [OpenCV](https://pypi.org/project/opencv-python/) for computer vision methods
- [kneed](https://pypi.org/project/kneed/) for the Nearest Neighbor elbow determination
- [MatPlotLib](https://matplotlib.org/stable/users/installing.html) for graphing capabilities
- [pandas](https://pandas.pydata.org/pandas-docs/stable/getting_started/install.html) for dealing with data
- [numpy](https://numpy.org/install/) to work with the images
- [scikit-learn](https://scikit-learn.org/stable/install.html) for clustering

## Usage

This code largely uses argparser and terminal commands to run the code. An example terminal command is below:
```bash
python3.7 dance_detector.py -i path/to/inputvideo.mp4 -c path/to/contour.pkl 
```
Information on each script's usage can be found with the ``` -h``` switch: 
```bash
python3.7 scriptname.py -h
```
The scripts should be run in the following order:

1. split.py 
2. scale_calibration.py
3. dance_detector.py
4. (optional) manual_aid.py
5. find_eps.py
6. clean_detections.py
7. waggle_dance_tracker.py
Other steps


## File Descriptions

- cluster_runs.py - Clusters each waggle based on proximity run into a waggle dance based on a forward search (within 5s of the previous run ending). (from the original Readme)
- cluster_tests.py - A convenience script to aid in testing different clustering parameters
- convertfiles.py - A convenience script to convert dataframes in pkls to csvs, etc. 
- dance_detector.py- Uses computer vision techniques and foreground detection for waggle detection in a video of an observation hive
- dance_vis.py - A convenience script to visualize clustered waggle detections directly on the videos they came from, drawing a colored circle where and when a waggle dance was detected, colored based on clusters.
- detection_cleaning.py - Removes duplicate points based on distance to previous detections of the same cluster (two coordinates at the same frame in the same cluster).
- find_eps.py - Determines DBSCAN parameter epsilon using the Nearest Neighbor method outlined in (link to paper)
- manual_aid.py - Creates a manifest json file to split a long video into clips with waggle dances in it, and an info file about the clips
- scale_calibrate.py - Detects the reference card and calculates the dimensions of the bee in pixels for the dance detector step
- scale_data.py - Convenience script to convert coords in 4k to other resolutions
- split.py - Script modified from AntTracker as a wrapper script for ffmpeg calls to split video
- save_dances.py - Saves each waggle dance as a separate cropped video. The bee is highlighted during the waggle run as a visual aid.
- waggle_run_tracker.py - Using the detection dataset as a skeleton, tracks the bee throughout the waggle run. Saves data on orientation, waggle frequency, spatial coordinates for each cluster.

## Changes made to [@Jreese18](https://github.com/Jreece18)'s [WaggleDanceTracker](https://github.com/Jreece18/WaggleDanceTracker)
| Modified                                                                                                                                                                | New                                                                                                                |
|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------|
| **dance_detector.py** - added histogram equalization and adaptive thresholding - separated out the DBSCAN clustering step (see find_eps.py) - added input for scale of bees | **scale_calibrate.py** - gets size of bees in pixels to pass to dance_detector.py                                      |
| **waggle_dance_tracker.py** - made adjustments for thresholding - work in progress                                                                                          | **find_eps.py** - uses Nearest Neighbors to find clustering parameters for DBSCAN - perform DBSCAN clustering on data  |
| **split.py** - made adjustments for our video                                                                                                                               | **dance_vis.py, scale_data.py, convertfiles.py, cluster_tests.py, makeplot.py** - convenience files for various things |
