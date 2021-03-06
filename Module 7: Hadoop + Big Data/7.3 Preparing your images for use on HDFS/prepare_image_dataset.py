from imutils import encodings
from imutils import paths
import progressbar
import argparse
import uuid
import cv2

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True, help="path to the directory that holds the images")
ap.add_argument("-o", "--output", required=True, help="path to output file for use on HDFS")
args = vars(ap.parse_args())

# grab the list of image paths in the dataset direstory and open the output file for writing
imagePaths = list(paths.list_images(args["dataset"]))
f = open(args["output"], "w")

# initialize the progress bar
widgets = ["Building Dataset: ", progressbar.Percentage(), " ", progressbar.Bar(), " ", progressbar.ETA()]
pbar = progressbar.ProgressBar(maxval=len(imagePaths), widgets=widgets).start()

# loop over the images in the dataset directory
for (i, path) in enumerate(imagePaths):
    # construct a unique ID for the image, encode the image as a string, and write
    # the data to a flattened txt file
    imageID = str(uuid.uuid4())
    image = encodings.base64_encode_image(cv2.imread(path))
    f.write(f"{imageID}\t{path}\t{image}\n")
    pbar.update(i)

# close the output file
pbar.finish()
f.close()


