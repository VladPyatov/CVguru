from __future__ import print_function
import numpy as np
import argparse
import cv2

def gftt(gray, maxCorners=0, qualityLevel=0.01, minDistance=1, mask=None, blockSize=3, useHarrisDetector=False, k=0.04):
    # compute 10.10.3 GFTT keypoints using the supplied parameters (OpenCV 3 only)
    kps = cv2.goodFeaturesToTrack(gray, maxCorners, qualityLevel, minDistance, mask=mask, blockSize=blockSize,
                                  useHarrisDetector=useHarrisDetector, k=k)
    # create and return KeyPoint objects
    return [cv2.KeyPoint(pt[0][0], pt[0][1], 3) for pt in kps]

# load the game and convert it to gray-scale
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="Image path")
args = vars(ap.parse_args())
image = cv2.imread(args["image"])
orig = image.copy()
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

kps = gftt(gray)

print(f"# of keypoints: {len(kps)}")
image = cv2.drawKeypoints(image, kps, None, color=(128, 255, 0))

# show the image
cv2.imshow("Images", np.hstack([orig, image]))
cv2.waitKey(0)
