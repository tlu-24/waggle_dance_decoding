# WaggleRunTracker.py
# from jordan reese (need to get link)
# Tracks bees (ideally) for the range of the entire waggle run, gets orientation and angle info

import pandas as pd
import numpy as np
import argparse
import cv2
from scipy import interpolate

# take inputs
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", required=True,
                help="path to the input cleaned waggle detection pkl file")
ap.add_argument("-v", "--video", required=True,
                help="path to the video that has the waggles")
ap.add_argument("-V", "--visualize", type=bool, default=False, required=False,
                help="show visualizations")
ap.add_argument("-b", "--bbox_info", required=True,
                help="path to the file with scale information of the bees")
args = vars(ap.parse_args())

FILENAME = args['input']
LABEL = FILENAME.split('-')[0]
VISUALIZE = args['visualize']
VIDEO = args['video']
bbox_df = pd.read_pickle(args['bbox_info'])
BBOX_SIZE = int(bbox_df['bee_len'])*2
SAVE = True


# Load dataset
path = FILENAME
df = pd.read_pickle(path)

# Finds largest contour within bounding box


def findROIContour(thresh, bbox):
    bbox = map(int, bbox)
    x, y, w, h = bbox
    # ROI based off bounding box coordinates
    thresh_roi = thresh[y:y+h, x:x+w]
    # Mask of black pixels so only ROI is searched for contour
    mask = np.zeros((thresh.shape[0], thresh.shape[1]), np.uint8)
    mask[y:y+h, x:x+w] = thresh_roi

    # Taken from: https://stackoverflow.com/questions/54615166/find-the-biggest-contour-opencv-python-getting-errors
    contours = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    contour_areas = [(cv2.contourArea(contour), contour)
                     for contour in contours[0]]
    # If no contour, return None
    # print(contour_areas)
    if contour_areas is None or len(contour_areas) == 0:
        final_c = [None, None]
        print("ROIcontour none")
    else:
        # Find largest contour in box
        final_c = max(contour_areas, key=lambda x: x[0])
        # print("ROIcontour", final_c[1])
    return final_c[1]

# Find Centre coordinates of contour


def getContourMoment(contour):
    m = cv2.moments(contour)
    # Find Contour centre
    x = m['m10'] / m['m00']
    y = m['m01'] / m['m00']
    return int(x), int(y)

# Finds the full contour based on bounding box ROI


def findFullContour(thresh, centre):
    x, y = centre
    # Find all contours in image
    contours = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

    # Return contour that centre belongs to
    for c in contours[0]:
        dist = cv2.pointPolygonTest(c, (x, y), False)
        if dist == 1.0:
            final_contour = c
            # if cv2.contourArea(c) > size * 1.5:
            #    pass
            break
        else:

            final_contour = findROIContour(
                thresh, (x-BBOX_SIZE//2, y-BBOX_SIZE//2, BBOX_SIZE, BBOX_SIZE))

    return final_contour

# Fits a bounding box tightly around the contour


def getFittedBox(contour):
    rect = cv2.minAreaRect(contour)
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    return rect, box

# Converts a fitted bounding box to a straight one with no rotation


def rotatedBoxConverter(box):
    box_t = np.array(box).T
    x, y = min(box_t[0]), min(box_t[1])
    w, h = max(box_t[0]) - x, max(box_t[1]) - y
    return x, y, w, h

# Calculate avg area of box


def avgArea(box, total, count):
    x, y, w, h = box
    total += w * h
    avg = total / count
    return total, avg


def expandBox(img, bbox):
    contour = None
    x, y, w, h = bbox
    x -= 10
    w += 20
    y -= 10
    h += 20
    bbox = (x, y, w, h)
    print(bbox)
    contour = findROIContour(img, bbox)
    return bbox, contour

# Find which way object is facing by the direction in which bounding box moves, to be coupled with angle of bounding rect


def moveDirection(prev_bbox, bbox):
    x, y, w, h = bbox
    x0, y0, w0, h0 = prev_bbox

    xd = x - x0
    # if xd is negative, moved west, if xd is positive, moved east
    yd = y - y0
    # if yd is negative, moved north, if yd is positive, moved south
    movement = (xd, yd)
    return movement


def createMask(img):
    mask = np.zeros((img.shape[0], img.shape[1], 3), np.uint8)
    mask.fill(255)
    return mask


def anchorInterpolation(bbox, fx, fy, counter):
    # Interpolated bbox
    fx0, fy0 = int(fx(counter)), int(fy(counter))
    x0, y0, x1, y1 = fx0-BBOX_SIZE, fy0-BBOX_SIZE, fx0+BBOX_SIZE, fy0+BBOX_SIZE

    x, y, w, h = bbox

    if x in range(x0, x1) and y in range(y0, y1):
        success = True
    else:
        success = False
        bbox = fx0-BBOX_SIZE//2, fy0-BBOX_SIZE//2, BBOX_SIZE, BBOX_SIZE

    return success, bbox


df.drop('index', axis=1, inplace=True)


# Create output dataframe
final_df = pd.DataFrame(columns=[
                        'x', 'y', 'frame', 'contour', 'bbox', 'size', 'angle', 'euclid', 'cluster'])

for i in range(len(df['Cluster'].unique())-1):
    print("ahhhhhhhhhhhh", i)
    clust = df[df['Cluster'] == i].reset_index()
    print(clust['frame'])

    # Extract values from df
    start = clust.iloc[0, :]['frame']
    # start = df.iloc[0, :]
    end = clust.iloc[-1, :]['frame']
    cluster = clust.iloc[0, :]['Cluster']
    # Get range of frames where waggle occurs
    rang = np.arange(start, end, 1)
    # Frames where waggle missing from df
    missing = list(set(rang)-set(clust.frame.values))

    # Setup video
    counter = start
    cap = cv2.VideoCapture(VIDEO)
    print(cap.isOpened())
    cap.set(1, start)
    ret, frame = cap.read()
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if SAVE:
        out = cv2.VideoWriter(LABEL+'_tracked'+str(i)+'.mp4', cv2.VideoWriter_fourcc(
            'm', 'p', '4', 'v'), 10, (width, height))

    # Preprocessing
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (45, 45), 1)
    thresh_min, thresh_max = 120, 220
    thresh = cv2.adaptiveThreshold(gray, 255,
                                   cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 121, 1)

    # morphological operations
    kernel = np.ones((2, 2), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
    # opening = cv2.erode(opening, kernel, iterations=1)
    cv2.imwrite("opening.jpeg", opening)
    cv2.imwrite("opening_th.jpeg", thresh)
    # Interpolation functions
    fx = interpolate.interp1d(clust.frame, clust.x, kind='slinear')
    fy = interpolate.interp1d(clust.frame, clust.y, kind='slinear')

    # Find contour bounding box
    x, y = clust.iloc[0, :]['x'], clust.iloc[0]['y']
    print(x, y)
    bbox = int(x-BBOX_SIZE//2), int(y-BBOX_SIZE//2), BBOX_SIZE, BBOX_SIZE
    # Skip Cluster if contour cannot be found in first frame
    try:
        contour = findROIContour(opening, bbox)
        print("contour")
        if contour is None:
            print('Contour None 1')
            contour = findROIContour(thresh, bbox)
            opening = thresh  # For findFullContour
        centre = getContourMoment(contour)
        contour = findFullContour(opening, centre)
    except:
        print("********************************")
        continue
    # If full contour is too large, use only the contour within the bounding box
    # if cv2.contourArea(contour) > clust['size'].max():
    #     contour = findROIContour(opening, bbox)
    #     print(contour)
    rect, box = getFittedBox(contour)
    bbox = rotatedBoxConverter(box)
    print(rect, box, bbox)
    # Initialise variables
    prev_bbox = bbox
    prev_centre = centre
    found = True
    avg = bbox[2]*bbox[3]
    total = avg

    rois = []

    while counter < end:
        counter += 1
        print(counter)
        ret, frame = cap.read()
        if ret is False:
            print('ret is false')
            break

        # Preprocessing
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_blur = cv2.GaussianBlur(gray, (9, 9), 0)
        thresh = cv2.adaptiveThreshold(gray_blur, 255,
                                       cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 121, 1)
        kernel = np.ones((2, 2), np.uint8)

        opening = cv2.morphologyEx(
            thresh, cv2.MORPH_OPEN, kernel, iterations=2)
        opening = cv2.erode(opening, kernel, iterations=1)
        cv2.imshow("opening", opening)

        # If frame is in df, use df coords as a reference
        if counter not in missing:
            print('In DF')
            waggle = clust[clust['frame'] == counter].reset_index()
            print(counter, start, end)
            x, y = waggle.loc[0, 'x'], waggle.loc[0, 'y']
            bbox = int(x-BBOX_SIZE//2), int(y-BBOX_SIZE //
                                            2), BBOX_SIZE, BBOX_SIZE
            print(prev_bbox, bbox)

        # If frame not in df, use previous bounding box as a reference
        if counter in missing:
            print('Missing')
            bbox = prev_bbox[0] - int(BBOX_SIZE//2), prev_bbox[1] - \
                int(BBOX_SIZE//2), prev_bbox[2] + \
                BBOX_SIZE, prev_bbox[3] + BBOX_SIZE
            # print(prev_bbox, bbox)

        # If bbox goes out of frame, end tracking
        if bbox[0] < 0 or bbox[0]+bbox[2] > width or bbox[1] < 0 or bbox[1]+bbox[3] > height:
            print('Object out of bounds')
            final_df.loc[len(final_df)] = 0
            break

        # Erode outside bounding box for improved segmentation
        save = opening[int(bbox[1]):int(bbox[1]+bbox[3]),
                       int(bbox[0]):int(bbox[0]+bbox[2])]
        opening = cv2.erode(opening, kernel, iterations=3)
        opening[int(bbox[1]):int(bbox[1]+bbox[3]),
                int(bbox[0]):int(bbox[0]+bbox[2])] = save

        # Find contour from bbox. If None, lower threshold inside the bounding box
        contour = findROIContour(opening, bbox)
        if contour is None:
            print('Contour None 301')
            contour = findROIContour(thresh, bbox)
            opening = thresh  # For findFullContour
        # If contour still None, lower threshold value
        low = thresh_min
        # or too small
        while contour is None or cv2.contourArea(contour) <= 80:
            print('Contour still none')
            low -= 5
            thresh = cv2.threshold(gray, low, thresh_max, cv2.THRESH_BINARY)[1]

            opening[int(bbox[1]):int(bbox[1]+bbox[3]), int(bbox[0]):int(bbox[0]+bbox[2])
                    ] = thresh[int(bbox[1]):int(bbox[1]+bbox[3]), int(bbox[0]):int(bbox[0]+bbox[2])]
            contour = findROIContour(opening, bbox)
            if low < 60:
                break

        roi_contour = contour  # Save for use in findFullContour failure
        # Readjust centre and find contour in centred ROI
        centre = getContourMoment(contour)

        bbox = centre[0]-BBOX_SIZE//2, centre[1] - \
            BBOX_SIZE//2, BBOX_SIZE, BBOX_SIZE
        contour = findROIContour(opening, bbox)
        # If new bbox goes out of frame, end tracking
        if bbox[0] < 0 or bbox[0]+bbox[2] > width or bbox[1] < 0 or bbox[1]+bbox[3] > height:
            # print('Object out of bounds')
            final_df.loc[len(final_df)] = 0
            break
        low = thresh_min

        # or too small
        while contour is None or cv2.contourArea(contour) <= 80:
            print('Contour still none 336')
            low -= 5
            thresh = cv2.threshold(gray, low, thresh_max, cv2.THRESH_BINARY)[1]
            opening[bbox[1]:bbox[1]+bbox[3], bbox[0]:bbox[0]+bbox[2]
                    ] = thresh[bbox[1]:bbox[1]+bbox[3], bbox[0]:bbox[0]+bbox[2]]
            contour = findROIContour(opening, bbox)
            if low < 60:
                break
        rect, box = getFittedBox(contour)
        bbox = rotatedBoxConverter(box)

        # Compare bbox to interpolated bbox
        found, bbox = anchorInterpolation(bbox, fx, fy, counter)
        if bbox[0] < 0 or bbox[0]+bbox[2] > width or bbox[1] < 0 or bbox[1]+bbox[3] > height:
            # print('Object out of bounds')
            final_df.loc[len(final_df)] = 0
            break
        # If no overlap with interpolated bbox, use interpolated coordinates
        if found is False:
            contour = findROIContour(opening, bbox)
            # Remove once fixed
            low = thresh_min
            while contour is None:  # or too small
                # print('Contour still none')
                low -= 5
                thresh = cv2.threshold(
                    gray, low, thresh_max, cv2.THRESH_BINARY)[1]
                opening[bbox[1]:bbox[1]+bbox[3], bbox[0]:bbox[0]+bbox[2]
                        ] = thresh[bbox[1]:bbox[1]+bbox[3], bbox[0]:bbox[0]+bbox[2]]
                contour = findROIContour(opening, bbox)
                if low < 60:
                    break
            rect, box = getFittedBox(contour)

            rect, box = getFittedBox(contour)
            bbox = rotatedBoxConverter(box)

        # Get angle and contour size
        angle = rect[-1]
        size = cv2.contourArea(contour)

        if VISUALIZE:
            # VISUALS
            if ret:
                # Tracking success
                p1 = (int(bbox[0]), int(bbox[1]))
                p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
                cv2.rectangle(frame, p1, p2, (255, 0, 0), 2, 1)
                cv2.rectangle(opening, p1, p2, (255, 0, 0), 2, 1)
            else:
                cv2.putText(frame, "Tracking failure detected", (100, 80),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
            cv2.imshow("Tracking", frame)
            cv2.imshow("Threshold", opening)
            cv2.waitKey(1)
        if SAVE:
            out.write(frame)

        # Euclidean distance between previous and current centre of contour
        euclid = np.sqrt(
            np.square(centre[0] - prev_centre[0]) + np.square(centre[1] - prev_centre[1]))

        # Fill output df
        final_df.loc[len(final_df)] = [centre[0], centre[1],
                                       counter, contour, bbox, size, angle, euclid, cluster]
        # Track direction of box movement
        movement = moveDirection(prev_bbox, bbox)
        prev_centre = centre
        prev_bbox = bbox
        # Track avg size of bounding box
        total, avg = avgArea(bbox, total, (counter-start))

    cap.release()
    if SAVE:
        out.release()

    if VISUALIZE:
        cv2.destroyAllWindows()
        cv2.waitKey(1)


# Save output df to pickle file
final_df.to_pickle(LABEL+'-WaggleRuns.pkl')
