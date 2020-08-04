from __future__ import print_function
import numpy as np
import argparse
import cv2

# load the game and convert it to gray-scale
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="Image path")
args = vars(ap.parse_args())
image = cv2.imread(args["image"])
orig = image.copy()
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

detector = cv2.ORB_create()
kps = detector.detect(gray, None)

print(f"# of keypoints: {len(kps)}")
image = cv2.drawKeypoints(image, kps, None, color=(128, 255, 0), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

# show the image
cv2.imshow("Images", np.hstack([orig, image]))
cv2.waitKey(0)