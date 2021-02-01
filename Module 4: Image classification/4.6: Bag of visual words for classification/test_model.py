from __future__ import print_function
from project.descriptors import DetectAndDescribe
from project.ir import BagOfVisualWords
from imutils.feature import DescriptorExtractor_create, FeatureDetector_create
from imutils import paths
import argparse
import pickle
import imutils
import cv2

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--images", required=True, help="Path to the input images directory")
ap.add_argument("-c", "--codebook", required=True, help="Path to the codebook")
ap.add_argument("-m", "--model", required=True, help="Path to the classifier")
args = vars(ap.parse_args())

# initialize the keypoint detector, local invariant descriptor, and the descriptor pipeline
detector = FeatureDetector_create("GFTT")
descriptor = DescriptorExtractor_create("RootSIFT")
dad = DetectAndDescribe(detector, descriptor)

# load the codebook vocabulary and initialize the bovw transformer
vocab = pickle.loads(open(args["codebook"], "rb").read())
bovw = BagOfVisualWords(vocab)

# load the classifier
model = pickle.loads(open(args["model"], "rb").read())

# loop over the image paths
for imagePath in paths.list_images(args["images"]):
    # load and preprocess image
    image = cv2.imread(imagePath)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = imutils.resize(gray, width=min(320, image.shape[1]))

    # describe the image and classify it
    (kps, descs) = dad.describe(gray)
    hist = bovw.describe(descs)
    hist /= hist.sum()
    prediction = model.predict(hist)[0]

    # show the prediction
    filename = imagePath[imagePath.rfind("/")+1:]
    print(f"[PREDICTION] {filename}:{prediction}")
    cv2.putText(image, prediction, (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
    cv2.imshow("Image", image)
    cv2.waitKey(0)
