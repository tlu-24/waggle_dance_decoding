import cv2
import argparse
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN


ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", required=True,
                help="path to the input video")
ap.add_argument("-c", "--contour", required=True,
               help="path to pkl with scale information aobut the bees and size of the bee waggle contour")
ap.add_argument("-b", "--blur", type=int, default=25, required=False,
                help="blur radius, usually works best when about half the width of the bee")
ap.add_argument("-save", "--save", default=False, required=False,
                help="save visualizations")
args = vars(ap.parse_args())

contour_df = pd.read_pickle(args['contour'])

BEE_CONTOUR = contour_df['bee_area'][0]
BLUR = args['blur']
SAVE = args['save']
FILENAME = args['input']
SAVE = True

def equalize_me(img_in):
    """ 
    Takes an image and performs histogram equalization.
        Inputs:
            - img_in: image to be equalized
        Outputs: 
            - histogram equalized
    """
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
    return img_out


# Find large child contours in the frame and return the x,y coordinates and the frame in which the contour was found
def find_child_contours(frame, frame_count):
    contours, hierarchy = cv2.findContours(frame, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    child_contours = []
    # Code taken from: https://stackoverflow.com/questions/22240746/recognize-open-and-closed-shapes-opencv
    hierarchy = hierarchy[0] 
    for i, c in enumerate(contours):
        # Return only innermost contours with no child contours 
        if hierarchy[i][2] < 0 and hierarchy[i][3] < 0:
            child_contours.append(c)
    x_coords = []
    y_coords = []
    sizes = []
    bases = []
    heights = []
    rect_size = []
    box_points = []
    for c in child_contours:
        # Only save large contour sizes
        if cv2.contourArea(c) > BEE_CONTOUR:
            m = cv2.moments(c)
            # Find Contour centre 
            x = m['m10'] / m['m00']
            y = m['m01'] / m['m00']
            cv2.drawContours(background, [c], -1, (0,255,0), 3)
            rect = cv2.minAreaRect(c)
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            cv2.drawContours(background, [box], 0, (0, 0, 255), 1)
            base = ((box[0][0] - box[1][0])**2 + (box[0][1] - box[1][1])**2)**0.5
            height = ((box[1][0] - box[2][0])**2 + (box[1][1] - box[2][1])**2)**0.5
            if height > base:
                base, height = height, base
            x_coords.append(x)
            y_coords.append(y)
            sizes.append(cv2.contourArea(c))
            rect_size.append(base*height)
            bases.append(base)
            heights.append(height)
            box_points.append(box)
    frame_counts = [frame_count] * len(x_coords) # Which frame these contours found in
    return list(zip(x_coords, y_coords, frame_counts, sizes, rect_size, bases, heights, box_points)) # Zip lists to list of tuples


### Motion Detector ### 
cap = cv2.VideoCapture(FILENAME)

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc = cv2.VideoWriter_fourcc(*'mp4v')

# VideoWriters for seeing the frame difference and the frame difference with the contour outlined
out1 = cv2.VideoWriter('framediff.mp4', fourcc, 30.0, (width,height), False)
out2 = cv2.VideoWriter('contourOutline.mp4', fourcc, 30.0, (width,height), True)

# Type font for adding frame counter to video
font                   = cv2.FONT_HERSHEY_SIMPLEX
bottomLeftCornerOfText = (10,500)
fontScale              = 1
fontColor              = (255,255,255)
lineType               = 2

delay_time = 1 # Delay next loop for easier viewing
prev_frame = None 
counter = 0 # Frame counter
potential_waggles = []
picture = True

while True:
    counter += 1
    ret, frame = cap.read()
    # Break when video ends
    if ret is False:
        break
    
    eq_frame = equalize_me(frame)
    gray = cv2.cvtColor(eq_frame, cv2.COLOR_BGR2GRAY)
    gray_blur = cv2.GaussianBlur(gray, (BLUR, BLUR), 0)
    thresh = cv2.adaptiveThreshold(gray_blur, 255,
                                   cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 91, 10)
    
    # If first frame, set current frame as prev_frame
    if prev_frame is None:
        prev_frame = thresh
    current_frame = thresh

    # Background Subtraction and Find Contours within image
    frame_diff = cv2.absdiff(current_frame, prev_frame) 
    _, hierarchy = cv2.findContours(frame_diff, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    background = frame_diff.copy()
    background = cv2.cvtColor(background, cv2.COLOR_GRAY2RGB)
    # Catch blank images and skip frame
    if hierarchy is None:
        continue
    else:
        find_waggles = find_child_contours(frame_diff, counter)
        potential_waggles += find_waggles
    
    # Frame Counter
    cv2.putText(thresh, str(counter), (40,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (150,0,0), 2)
    
    # Display the resulting frame
    cv2.imshow('Thresholded', thresh)
    cv2.imshow('Frame Diff', frame_diff)
    cv2.imshow('Grayscale', gray)
    cv2.imshow('GrayBlur', gray_blur)
    cv2.imshow('background', background)

    if SAVE:
        out1.write(frame_diff)
        out2.write(background)
     
    cv2.waitKey(delay_time)
    
    # Make current frame the previous frame for the next loop
    prev_frame = current_frame
    
    # q to quit
    if cv2.waitKey(delay_time) & 0xFF == ord('q'):
        break

    
cap.release()
cv2.destroyAllWindows()
cv2.waitKey(1)

### ROI Clustering ###

# Convert all waggle like activity to DF
waggle_df = pd.DataFrame(potential_waggles, columns=['x', 'y', 'frame', 'size',  'rect_size', 'base', 'height', 'box_points'])


waggle_df.to_pickle('{}-WaggleDetectionsUnclustered.pkl'.format(FILENAME.split('.')[0]))
waggle_df.to_csv('{}-WaggleDetectionsUnclustered.csv'.format(FILENAME.split('.')[0]))