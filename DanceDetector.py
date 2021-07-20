# DanceDetector.py
# need link to Jordan Reese's github
# Uses foreground detection for waggle detection in video

import cv2
import numpy as np
import pandas as pd
import argparse
from sklearn.cluster import DBSCAN
from matplotlib import pyplot as plt
from sklearn.neighbors import NearestNeighbors

# take inputs
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", required=True,
                help="path to the input video")
ap.add_argument("-c", "--contour", required=True,
                help="path to pkl with scale information aobut the bees and size of the bee waggle contour")
ap.add_argument("-b", "--blur", type=int, default=25, required=False,
                help="blur radius, usually works best when about half the width of the bee")
ap.add_argument("-v", "--visualize", default=False, required=False,
                help="show visualizations")
args = vars(ap.parse_args())

contour_df = pd.read_pickle(args['contour'])

BEE_CONTOUR = contour_df['bee_area'][0] * 1.5
BLUR = args['blur']
VISUALIZE = args['visualize']
FILENAME = args['input']
LABEL = FILENAME.split('/')[-1].split('.')[0]


def equalizeMe(img_in):
  # equalize your image. Follow this guide:
  # https://towardsdatascience.com/histogram-equalization-a-simple-way-to-improve-the-contrast-of-your-image-bcd66596d815
  # segregate color streams
    b, g, r = cv2.split(img_in)
    h_b, bin_b = np.histogram(b.flatten(), 256, [0, 256])
    h_g, bin_g = np.histogram(g.flatten(), 256, [0, 256])
    h_r, bin_r = np.histogram(r.flatten(), 256, [0, 256])
# calculate cdf
    cdf_b = np.cumsum(h_b)
    cdf_g = np.cumsum(h_g)
    cdf_r = np.cumsum(h_r)

# mask all pixels with value=0 and replace it with mean of the pixel values
    cdf_m_b = np.ma.masked_equal(cdf_b, 0)
    cdf_m_b = (cdf_m_b - cdf_m_b.min())*255/(cdf_m_b.max()-cdf_m_b.min())
    cdf_final_b = np.ma.filled(cdf_m_b, 0).astype('uint8')

    cdf_m_g = np.ma.masked_equal(cdf_g, 0)
    cdf_m_g = (cdf_m_g - cdf_m_g.min())*255/(cdf_m_g.max()-cdf_m_g.min())
    cdf_final_g = np.ma.filled(cdf_m_g, 0).astype('uint8')
    cdf_m_r = np.ma.masked_equal(cdf_r, 0)
    cdf_m_r = (cdf_m_r - cdf_m_r.min())*255/(cdf_m_r.max()-cdf_m_r.min())
    cdf_final_r = np.ma.filled(cdf_m_r, 0).astype('uint8')
# merge the images in the three channels
    img_b = cdf_final_b[b]
    img_g = cdf_final_g[g]
    img_r = cdf_final_r[r]

    img_out = cv2.merge((img_b, img_g, img_r))
# validation
    equ_b = cv2.equalizeHist(b)
    equ_g = cv2.equalizeHist(g)
    equ_r = cv2.equalizeHist(r)
    equ = cv2.merge((equ_b, equ_g, equ_r))
    # print(equ)
    #cv2.imwrite('output_name.png', equ)
    return img_out
# Find large child contours in the frame and return the x,y coordinates and the frame in which the contour was found


def findChildContours(frame, frame_count):
    contours, hierarchy = cv2.findContours(
        frame, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    child_contours = []
    # Code taken from: https://stackoverflow.com/questions/22240746/recognize-open-and-closed-shapes-opencv
    hierarchy = hierarchy[0]
    for i, c in enumerate(contours):
        # Return only innermost contours with no child contours
        if hierarchy[i][2] < 0 and hierarchy[i][3] < 0:
            child_contours.append(c)
    x_coords = []
    y_coords = []
    size = []
    for c in child_contours:
        # Only save large contours # Originally 200
        if cv2.contourArea(c) > BEE_CONTOUR:
            m = cv2.moments(c)
            # Find Contour centre
            x = m['m10'] / m['m00']
            y = m['m01'] / m['m00']
            x_coords.append(x)
            y_coords.append(y)
            size.append(cv2.contourArea(c))
    # Which frame these contours found in
    frame_counts = [frame_count] * len(x_coords)
    # Zip lists to list of tuples # Size removed
    return list(zip(x_coords, y_coords, frame_counts))


### Motion Detector ###
path = FILENAME

cap = cv2.VideoCapture(path)


width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Type font for adding frame counter to video
font = cv2.FONT_HERSHEY_SIMPLEX
bottomLeftCornerOfText = (10, 500)
fontScale = 1
fontColor = (255, 255, 255)
lineType = 2

delay_time = 1  # Delay next loop for easier viewing
prev_frame = None
counter = 0  # Frame counter
potential_waggles = []  # List to save potential waggles
norm_img = np.zeros((width, height))
while True:
    counter += 1
    ret, frame = cap.read()

    #            cv2.cvtColor(eq_frame, cv2.COLOR_RGB2BGR))

    # Break when video ends
    if ret is False:
        break

    # eq_frame = cv2.normalize(frame, norm_img, 0, 255, cv2.NORM_MINMAX)
    eq_frame = equalizeMe(frame)

    # Threshold Image
    gray = cv2.cvtColor(eq_frame, cv2.COLOR_BGR2GRAY)
    gray_blur = cv2.GaussianBlur(gray, (BLUR, BLUR), 0)
    thresh = cv2.adaptiveThreshold(gray_blur, 255,
                                   cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 111, 10)

    # if counter == 51:
    #     plt.imsave('frame' + str(counter) + '.jpeg',
    #                cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    #     plt.imsave('eqframe' + str(counter) + '.jpeg',
    #                cv2.cvtColor(eq_frame, cv2.COLOR_RGB2BGR))
    #     plt.imsave('grayframe' + str(counter) + '.jpeg',
    #                cv2.cvtColor(gray, cv2.COLOR_RGB2BGR))
    #     plt.imsave('grayblur' + str(counter) + '.jpeg',
    #                cv2.cvtColor(gray_blur, cv2.COLOR_RGB2BGR))
    #     plt.imsave('thresh' + str(counter) + '.jpeg',
    #                cv2.cvtColor(thresh, cv2.COLOR_RGB2BGR))

    # If first frame, set current frame as prev_frame
    if prev_frame is None:
        prev_frame = thresh
    current_frame = thresh

    # Background Subtraction and Find Contours within image
    frame_diff = cv2.absdiff(current_frame, prev_frame)
    _, hierarchy = cv2.findContours(
        frame_diff, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)

    # Catch blank images and skip frame
    if hierarchy is None:
        continue
    else:
        find_waggles = findChildContours(frame_diff, counter)
        potential_waggles += find_waggles

    # Frame Counter
    cv2.putText(thresh, str(counter), (40, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (150, 0, 0), 2)

    if VISUALIZE:
        # Display the resulting frame
        cv2.imshow('normalized', eq_frame)
        cv2.imshow('Thresholded', thresh)
        cv2.imshow('Frame Diff', frame_diff)
        cv2.imshow('Grayscale', gray)
        cv2.imshow('GrayBlur', gray_blur)
        cv2.waitKey(delay_time)

    # Make current frame the previous frame for the next loop
    prev_frame = current_frame

    if VISUALIZE:
        # q to quit
        if cv2.waitKey(delay_time) & 0xFF == ord('q'):
            break
    if counter % 100 == 0:
        print(counter)
        # plt.imsave('thisone' + str(counter) + '.jpeg',
        #            cv2.cvtColor(eq_frame, cv2.COLOR_RGB2BGR))


cap.release()
if VISUALIZE:
    cv2.destroyAllWindows()
    cv2.waitKey(1)


# Convert all waggle like activity to DF
waggle_df = pd.DataFrame(potential_waggles, columns=['x', 'y', 'frame'])


waggle_df.to_pickle('{}-WaggleDetections.pkl'.format(LABEL))
waggle_df.to_csv('wagggle_detections.csv')
