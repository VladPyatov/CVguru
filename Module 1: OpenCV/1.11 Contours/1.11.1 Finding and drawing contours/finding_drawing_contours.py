import numpy as np
import cv2
import argparse
import imutils

# construct the argument parser and parse the argument
ap = argparse.ArgumentParser()
ap.add_argument("-i","--image", required=True, help="Path to the image")
args = vars(ap.parse_args())

# load the image and convert it to grayscale
image = cv2.imread(args["image"])
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

cv2.imshow("Original", image)

# find all contours in the image and draw all contours on the image
cnts = cv2.findContours(gray, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
clone = image.copy()
cv2.drawContours(clone, cnts, -1, (0,255,0), 2)
print(f"Found {len(cnts)} contours")

# show the output image
cv2.imshow("All contours", clone)
cv2.waitKey(0)

# re-clone the image and close all open windows
clone = image.copy()
cv2.destroyAllWindows()

# loop over the contours individually and draw each of them
for (i, c) in enumerate(cnts):
    print(f"Drawing contour #{i+1}")
    cv2.drawContours(clone,[c],-1,(0,255,0), 2)
    cv2.imshow("Single contour", clone)
    cv2.waitKey(0)

# re-clone the image and close all open windows
clone = image.copy()
cv2.destroyAllWindows()

# find all contours in the image but this time keep only the EXTERNAL
# contours in the image
cnts = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
cv2.drawContours(clone, cnts, -1, (0,255,0), 2)
print(f"Found {len(cnts)} EXTERNAL contours")

# show the output image
cv2.imshow("All contours", clone)
cv2.waitKey(0)

# re-clone the image and close all open windows
clone = image.copy()
cv2.destroyAllWindows()

# loop over the contours individually
for c in cnts:
    # construct a mask by drawing only the current contour
    mask = np.zeros(gray.shape, dtype="uint8")
    cv2.drawContours(mask, [c], -1, 255, -1)

    # show the images
    cv2.imshow("Image", image)
    cv2.imshow("Mask", mask)
    cv2.imshow("Image + Mask", cv2.bitwise_and(image, image, mask=mask))
    cv2.waitKey(0)
