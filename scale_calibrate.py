# scale_calibrate.py
# this file takes a bee video, and extracts the size of the checkerboard pattern on the
# reference card, then calculates the dimensions and area of bee in pixels

import cv2
import numpy as np
import argparse
import imutils
import statistics
import pandas as pd
from matplotlib import pyplot as plt


# take inputs
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
                help="path to the input video")
ap.add_argument("-o", "--outdir", required=True,
                help="path to the out directory")
ap.add_argument("-l", "--leftside", type=bool, required=True,
                help="is the reference card on the left side?")
ap.add_argument("-s", "--size", type=int, default=1.5, required=False,
                help="length of bee in cm")
ap.add_argument("-v", "--visualize", default=False, required=False,
                help="show visualizations")
args = vars(ap.parse_args())

LEFTSIDE = args['leftside']  # is the reference card on the left side
BEE_SIZE = args['size']  # length of bee in cm
VISUALIZE = args['visualize']  # show visualizations
OUT_DIR = args['outdir']  # directory for output
label = args['image'].split('.')[0].split('/')[-1]  # output filename

# Performs histogram equalization on a color image


def equalizeMe(img_in):
  # From:
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


def detect(c):
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.04*peri, True)
    h = 0
    w = 0
    shape = "nope"

    # if it has 4 sides, must be a rectangle or square
    if len(approx) == 4:
        (x, y, w, h) = cv2.boundingRect(approx)
        h = h
        w = w
        ar = w/float(h)
        shape = "square" if ar >= 0.95 and ar <= 1.05 else "rectangle"

    return (h, w, shape)


# open video
cap = cv2.VideoCapture(args['image'])


width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# loop through video and save the 80th frame to do the calibration from
# this is a hacky solution but oh well
prev_frame = None
counter = 0  # Frame counter
while True:
    counter += 1
    ret, frame = cap.read()

    # Break when video ends
    if ret is False:
        break

    # save image
    if counter == 80:
        plt.imsave('image' + '.jpeg',
                   cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        break

# load image
image = equalizeMe(cv2.imread('image.jpeg'))
height = image.shape[0]
width = image.shape[1]

# crop to just the scale card.
# Assuming that it is in the bottom quarter of the image on either the left or right side
if LEFTSIDE:
    crop = image[int(height*3/4):height, 0:width//2]
else:
    crop = image[int(height*3/4):height, width//2:width]

if VISUALIZE:
    cv2.imshow("crop", crop)

# resize it (but not really)
resized = imutils.resize(crop, width=width)
ratio = crop.shape[0] / float(resized.shape[0])

# convert the resized image to grayscale, blur it slightly,
# and threshold it
gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (1, 1), 0)
thresh = cv2.threshold(blurred, 128, 255, cv2.THRESH_BINARY_INV)[1]

# get contours from the thresholded images
cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
                        cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)

if VISUALIZE:
    cv2.imshow("threst", thresh)

# loop through all the contours, get the sqaures
squares = []
maxsquares = []
for c in cnts:
    M = cv2.moments(c)
    if M['m00'] == 0:
        continue
    cX = int((M['m10']/M['m00'])*ratio)
    cY = int((M['m01']/M['m00'])*ratio)
    (h_1, w_1, shape) = detect(c)
    h = h_1 * ratio
    w = w_1*ratio
    if shape != 'square' or h < 10:
        continue

    squares.append(((cX, cY), h, w))
    maxsquares.append(max(h, w))
    # multiply the contour (x, y)-coordinates by the resize ratio,
    # then draw the contours and the name of the shape on the image
    c = c.astype('float')
    c *= ratio
    c = c.astype('int')

    if VISUALIZE:
        cv2.drawContours(crop, [c], -1, (0, 255, 0), 2)
        cv2.putText(crop, shape + str(h) + " "+str(int(w)), (cX, cY), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 255, 255), 2)

if VISUALIZE:
    cv2.imshow("image", crop)
    cv2.waitKey(0)

# the median largest square is probably a reference square (which is 1cm in width)
pix_cm = statistics.median(maxsquares)

# calulate the size of a bee in pixels
bee_l = BEE_SIZE*pix_cm*0.47
bee_w = bee_l*0.4

bee_area = int(bee_l*bee_w)  # in pixels

out_dict = {'pixelspercm': [pix_cm], 'bee_len': [bee_l],
            'bee_w': [bee_w], 'bee_area': [bee_area]}
out_df = pd.DataFrame(
    out_dict)
out_df.to_pickle(OUT_DIR+label + '_scale.pkl')

# change later
print(squares, '\npixels per cm =', pix_cm, '\n bee length =',
      bee_l, '\n bee width =', bee_w, '\n bee area =', bee_area)
