from __future__ import print_function
from ObjectDetector.utils import dataset
from ObjectDetector.utils import Conf
from sklearn.svm import SVC
import numpy as np
import argparse
import pickle

# construct the argument parser and parse the command line arguments
ap = argparse.ArgumentParser()
ap.add_argument('-c', "--conf", required=True, help="Path to the configuration file")
ap.add_argument('-n', "--hard-negatives", type=int, default=-1,
                help="Flag indicating whether or not hard negatives should be used")
args = vars(ap.parse_args())

# load the configuration file and the initial dataset
print("[INFO] loading dataset...")
conf = Conf(args["conf"])
data, labels = dataset.load_dataset(conf["features_path"], "features")

# check to see if the hard negatives flag was supplied
if args["hard-negatives"] > 0:
    print("[INFO] loading hard negatives...")
    hardData, hardLabels = dataset.load_dataset(conf["features_path"], "hard_negatives")
    data = np.vstack([data, hardData])
    labels = np.hstack([labels, hardLabels])

# train the classifier
print("[INFO] training classifier...")
model = SVC(kernel="linear", C=conf["C"], probability=True, random_state=42)
model.fit(data, labels)

# dump the classifier to file
print("[INFO] dumping classifier...")
f = open(conf["classifier_path"], "wb")
f.write(pickle.dumps(model))
f.close()

