import argparse
import cv2

# construct the argument parser and parse the arguments
arg_p = argparse.ArgumentParser()
arg_p.add_argument("-i", "--image", required=True, help="Path to the image")
args = vars(arg_p.parse_args())

# load the image, grab its dimensions, and show it
image = cv2.imread(args["image"])
(h, w) = image.shape[:2]
cv2.imshow("Original", image)

(b, g, r) = image[0, 0]
print(f"Pixel at (0,0) - Red: {r}, Green: {g}, Blue: {b}")

image[0,0] = (0,0,255)
(b, g, r) = image[0, 0]
print(f"Pixel at (0,0) - Red: {r}, Green: {g}, Blue: {b}")

# image center
cX, cY = w//2, h//2

# top-left image corner
topleft = image[0:cY, 0:cX]
cv2.imshow("Top-Left Corner", topleft)

# top-right, bottom-right and bottom-left corners
topright = image[0:cY, cX:w]
bottomright = image[cY:h, cX:w]
bottomleft = image[cY:h, 0:cX]

cv2.imshow("Top-Right Corner", topright)
cv2.imshow("Bottom-Right Corner", bottomright)
cv2.imshow("Bottom-Left Corner", bottomleft)

# green top-left corner
image[0:cY, 0:cX]= (0,255,0)
cv2.imshow("Updated", image)

cv2.waitKey(0)
