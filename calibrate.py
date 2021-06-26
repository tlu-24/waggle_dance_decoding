import cv2
import imutils
import numpy as np
from matplotlib import pyplot as plt
from numpy.lib.function_base import average
import skimage as sk
import skimage.transform as sktr
from skimage import io, img_as_float
import glob

# load image
path = "../beans/IMG_2754_81.jpeg"
pg = io.imread(path)
height = len(pg[0])
width = len(pg)

print(width, height)
out = np.zeros((height+1, width+1, 3))
eq_frame = cv2.normalize(pg, out, 0, 255, cv2.NORM_MINMAX)
# eq_frame = equalizeMe(eq_frame)
# Threshold Image
gray = cv2.cvtColor(eq_frame, cv2.COLOR_BGR2GRAY)
# print(pg)
# dst = cv2.xphoto_WhiteBalancer.balanceWhite(gray, out)


def white_balance(img):
    # from https://stackoverflow.com/questions/46390779/automatic-white-balancing-with-grayworld-assumption
    result = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    avg_a = np.average(result[:, :, 1])
    avg_b = np.average(result[:, :, 2])
    result[:, :, 1] = result[:, :, 1] - \
        ((avg_a - 128) * (result[:, :, 0] / 255.0) * 1.1)
    result[:, :, 2] = result[:, :, 2] - \
        ((avg_b - 128) * (result[:, :, 0] / 255.0) * 1.1)
    result = cv2.cvtColor(result, cv2.COLOR_LAB2BGR)
    return result


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


def calculate_bee_size(img, h, w):
    # recognize the guide

    # do the math

    return True


thresholds = []

path = '../test.mp4'

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

    # Break when video ends
    if ret is False:
        break

    # eq_frame = cv2.normalize(frame, norm_img, 0, 255, cv2.NORM_MINMAX)
    eq_frame = equalizeMe(frame)
    # Threshold Image
    gray = cv2.cvtColor(eq_frame, cv2.COLOR_BGR2GRAY)
    gray_blur = cv2.GaussianBlur(gray, (11, 11), 0)
    # apply Otsu's automatic thresholding which automatically determines
    # the best threshold value
    (T, thresh) = cv2.threshold(gray_blur, 0, 255,
                                cv2.cv2.THRESH_OTSU)
    # cv2.imshow("Threshold", thresh)
    thresholds.append(T)
    # print("[INFO] otsu's thresholding value: {}".format(T))
    # visualize only the masked regions in the image
    # masked = cv2.bitwise_and(frame, image, mask=threshInv)
    # cv2.imshow("Output", masked)
    # cv2.waitKey(0)

    # If first frame, set current frame as prev_frame
    if prev_frame is None:
        prev_frame = thresh
    current_frame = thresh

    # Display the resulting frame

    cv2.imshow('Thresholded', thresh)
    # cv2.imshow('Frame Diff', frame_diff)
    cv2.imshow('Grayscale', gray)
    cv2.imshow('GrayBlur', gray_blur)
    # cv2.waitKey(delay_time)

    # Make current frame the previous frame for the next loop
    prev_frame = current_frame

    # q to quit
    if cv2.waitKey(delay_time) & 0xFF == ord('q'):
        break

    if counter > 1000:
        break
    if counter % 100 == 0:
        print(counter)


cap.release()
cv2.destroyAllWindows()
cv2.waitKey(1)

print(average(thresholds))
# final = white_balance(pg)
# final2 = equalizeMe(final)
# plt.imsave('out3.jpeg', final2)
# something with size!??

# threshold checking
