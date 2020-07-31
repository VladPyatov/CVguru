from __future__ import print_function
import numpy as np
import argparse
import cv2


def harris(gray, blockSize=2, apetureSize=3, k=0.1, T=0.02):
    # convert our input image to a floating point data type and then
    # and then compute Harris Corner matrix
    gray = np.float32(gray)
    H = cv2.cornerHarris(gray, blockSize, apetureSize, k)

    # for every (x, y)-coordinate where the Harris value is above the
    # threshold, create a keypoint (the Harris detector returns
    # keypoint size a 3-pixel radius)
    kps = np.argwhere(H > T * H.max())
    kps = [cv2.KeyPoint(pt[1], pt[0], 3) for pt in kps]

    # return the Harris keypoints
    return kps

# load the game and convert it to gray-scale
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="Image path")
args = vars(ap.parse_args())
image = cv2.imread(args["image"])
orig = image.copy()
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

kps = harris(gray)

print(f"# of keypoints: {len(kps)}")
image = cv2.drawKeypoints(image, kps, None, color=(128, 255, 0))

# show the image
cv2.imshow("Images", np.hstack([orig, image]))
cv2.waitKey(0)
