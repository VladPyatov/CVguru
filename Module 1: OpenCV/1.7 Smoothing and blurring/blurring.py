import argparse
import cv2

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="Path to the image")
args = vars(ap.parse_args())

# load the image, display it, and initialize the list of kernel sizes
image = cv2.imread(args["image"])
cv2.imshow("Original", image)
kernelSizes = [(3, 3), (9, 9), (15, 15)]

# average blurring
for k_size in kernelSizes:
    blurred = cv2.blur(image, k_size)
    cv2.imshow(f"Average {k_size}", blurred)
    cv2.waitKey(0)

cv2.destroyAllWindows()
cv2.imshow("Original", image)

# Gaussian blurring
for k_size in kernelSizes:
    blurred = cv2.GaussianBlur(image, k_size, 0)

    cv2.waitKey(0)

# close all windows to cleanup the screen
cv2.destroyAllWindows()
cv2.imshow("Original", image)

# loop over the kernel sizes and apply a "Median" blur to the image
for k_size in (3, 9, 15):
    blurred = cv2.medianBlur(image, k_size)
    cv2.imshow(f"Median {k_size}", blurred)
    cv2.waitKey(0)
