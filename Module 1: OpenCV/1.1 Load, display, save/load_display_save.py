import argparse
import cv2

# construct the argument parser and parse the arguments
arg_p = argparse.ArgumentParser()
arg_p.add_argument("-i", "--image", required=True, help="Path to the image")
arg_p.add_argument("-o", "--output", required=False, help="Output path of the image")
args = vars(arg_p.parse_args())

# load the image and show some basic information on it
image = cv2.imread(args["image"])
print(f"width: {image.shape[1]} pixels")
print(f"height: {image.shape[0]} pixels")
print(f"channels: {image.shape[2]}")

# show the image and wait for keypress
cv2.imshow("Image", image)
cv2.waitKey(0)

# save image
if args["output"]:
    cv2.imwrite(args["output"], image)




