import argparse
import cv2

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="Path to the image")
args = vars(ap.parse_args())

image = cv2.imread(args["image"])
cv2.imshow("Original", image)
params = [(11, 21, 7), (11, 41, 21), (11, 61, 39)]

# loop over the diameter, sigma color, and sigma space
for (diameter, sigmaColor, sigmaSpace) in params:
    # apply bilateral filtering and display the image
    blurred = cv2.bilateralFilter(image, diameter, sigmaColor, sigmaSpace)
    cv2.imshow(f"Blurred d={diameter}, sc={sigmaColor}, ss={sigmaSpace}", blurred)
    cv2.waitKey(0)