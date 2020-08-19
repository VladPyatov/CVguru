import numpy as np
import cv2
import imutils


class HSVDescriptor:
    def __init__(self, bins):
        # store the number of bins for the histogram
        self.bins = bins

    def describe(self, image):
        # convert image to hsv and initialize
        # the features to quantify the image
        image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        features = []

        # grab the dimensions and compute the center of the image
        (h, w) = image.shape[:2]
        cX, cY = int(0.5*w), int(0.5*h)

        # divide the image into four rectangles/segments
        # (top-left, top-right, bottom-right, bottom-left)
        segments = [(0, cX, 0, cY), (cX, w, 0, cY), (cX, w, cY, h), (0, cX, cY, h)]

        # construct the elliptical mask representing the center of the image
        axesX, axesY = int(0.75*w)//2, int(0.75*h)//2
        ellipseMask = np.zeros(image.shape[:2], dtype='uint8')
        cv2.ellipse(ellipseMask, (cX, cY), (axesX, axesY), 0, 0, 360, 255, -1)

        # loop over the segments
        for (startX, endX, startY, endY) in segments:
            # construct a mask for each corner of the image, substracting
            # the elliptical center from it
            cornerMask = np.zeros(image.shape[:2], dtype="uint8")
            cv2.rectangle(cornerMask, (startX, startY), (endX, endY), 255, -1)
            cornerMask = cv2.subtract(cornerMask, ellipseMask)

            # extract a color histogram from the image, then update the feature vector
            hist = self.histogram(image, cornerMask)
            features.extend(hist)

        # extract a color histogram from the elliptical region, then update the feature vector
        hist = self.histogram(image, ellipseMask)
        features.extend(hist)

        # return the feature vector
        return np.array(features)

    def histogram(self, image, mask=None):
        # extracting a 3D color histogram from the masked region of the
        # image, using the supplied number of bins per channel
        # the normalize the histogram
        hist = cv2.calcHist([image], [0,1,2], mask, self.bins, [0, 180, 0, 256, 0, 256])

        hist = cv2.normalize(hist, hist).flatten()

        return hist







