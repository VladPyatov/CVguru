from __future__ import print_function
from sklearn.feature_extraction.image import extract_patches_2d
from ObjectDetector.object_detection import helpers
from ObjectDetector.descriptors import HOG
from ObjectDetector.utils import dataset, Conf
from imutils import paths
from scipy import io
import numpy as np
import progressbar
import argparse
import random
import cv2

# construct the argument parser and parse the command line arguments
ap = argparse.ArgumentParser()
ap.add_argument("-c", "--conf", required=True, help="path to the configuration file")
args = vars(ap.parse_args())

# load the configuration file
conf = Conf(args["conf"])

# initialize the HOG descriptor along with the list of data and labels
hog = HOG(orientations=conf["orientations"], pixelsPerCell=tuple(conf["pixels_per_cell"]),
          cellsPerBlock=tuple(conf["cells_per_block"]), normalize=conf["normalize"])
data = []
labels = []

trnPaths = list(paths.list_images(conf["image_dataset"]))
trnPaths = random.sample(trnPaths, int(len(trnPaths)*conf["percent_gt_images"]))
print("[INFO] describing training ROI's...")

# set up the progress bar
widgets = ["Extracting: ", progressbar.Percentage(), " ", progressbar.Bar(), " ", progressbar.ETA()]
pbar = progressbar.ProgressBar(maxval=len(trnPaths), widgets=widgets).start()

for (i, trnPath) in enumerate(trnPaths):
    # load the image, convert it to grayscale, and extract the image ID from the path
    image = cv2.imread(trnPath, cv2.IMREAD_GRAYSCALE)
    imageID = trnPath[trnPath.rfind("_")+1:].replace(".jpg", "")

    # load the annotation file associated with the image and extract the bounding box
    p = f"{conf['image_annotations']}/annotation_{imageID}.mat"
    bb = io.loadmat(p)["box_coord"][0]
    roi = helpers.crop_ct101_bb(image, bb, padding=conf["offset"], dstSize=tuple(conf["window_dim"]))

    # define the list of ROIs that will be described, based on whether or not the
    # horizontal flip of the image should be used
    rois = (roi, cv2.flip(roi, 1)) if conf["use_flip"] else (roi,)

    # loop over the ROIs
    for roi in rois:
        # extract features from the ROI and update the list of features and labels
        features = hog.describe(roi)
        data.append(features)
        labels.append(1)

    # update the progress bar
    pbar.update(i)

# grab the distraction image paths and reset the progress bar
pbar.finish()
dstPaths = list(paths.list_images(conf["image_distractions"]))
pbar = progressbar.ProgressBar(maxval=conf["num_distraction_images"], widgets=widgets).start()
print("[INFO] describing distraction ROI's...")

for i in np.arange(0, conf["num_distraction_images"]):
    # randomly select a distraction image, load it, convert it to grayscale, and
    # then extract random patches from the image
    image = cv2.imread(random.choice(dstPaths), cv2.IMREAD_GRAYSCALE)
    patches = extract_patches_2d(image, patch_size=tuple(conf["window_dim"]),
                                 max_patches=conf["num_distractions_per_image"])

    # loop over the patches
    for patch in patches:
        # extract features from the patch, then update the data and label list
        features = hog.describe(patch)
        data.append(features)
        labels.append(-1)

    # update the progress bar
    pbar.update(i)


# dump the dataset to file
pbar.finish()
print("[INFO] dumping features and labels to file...")
dataset.dump_dataset(data, labels, conf["features_path"], "features")
