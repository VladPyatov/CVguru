from skimage import feature
import numpy as np


class LocalBinaryPatterns:
    def __init__(self, numPoints, radius):
        # store the number of points and radius
        self.numPoints = numPoints
        self.radius = radius

    def describe(self, image, eps=1e-7):
        # compute the Local Binary Patterns representation of the image,
        # and then use the LBP to build the histogram of patterns
        lbp = feature.local_binary_pattern(image, self.numPoints, self.radius, method="uniform")
        hist, _ = np.histogram(lbp.ravel(), bins=range(0, self.numPoints+3), range=range(0, self.numPoints+2))

        # normalize the histogram
        hist = hist.astype("float")
        hist /= (hist.sum()+eps)

        return hist


