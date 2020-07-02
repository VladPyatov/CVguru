import numpy as np
import argparse
import cv2

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="Path to the image")
args = vars(ap.parse_args())

# load the image and show it
image = cv2.imread(args["image"])
cv2.imshow("Original", image)
cv2.waitKey(0)

# cv2 arithmetic
print(f"max of 255: {cv2.add(np.uint8([200]), np.uint8([100]))}")
print(f"min of 0: {cv2.subtract(np.uint8([50]), np.uint8([100]))}")

# NumPy arithmetic
print(f"wrap around: {np.uint8([200]) + np.uint8([100])}")
print(f"wrap around: {np.uint8([50]) + np.uint8([100])}")

# let's increase the intensity of all pixels in our image by 100
M = np.full(image.shape, 100, dtype="uint8")
added = cv2.add(image, M)
cv2.imshow("Added", added)

# similarly, we can subtract 50 from all pixels in our image
M = np.full(image.shape, 50, dtype="uint8")
subtracted = cv2.subtract(image, M)
cv2.imshow("Subtracted", subtracted)
cv2.waitKey(0)
