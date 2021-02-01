from __future__ import print_function
from sklearn.metrics import classification_report
from sklearn.svm import LinearSVC
import sklearn
import numpy as np
import argparse
import pickle
import h5py
import cv2

# handle sklearn versions less than 0.18
if int((sklearn.__version__).split(".")[1]) < 18:
    from sklearn.grid_search import GridSearchCV
else:
    from sklearn.model_selection import GridSearchCV

# constructing the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True, help="Path to the directory that contains the original images")
ap.add_argument("-f", "--features-db", required=True, help="Path the features database")
ap.add_argument("-b", "--bovw-db", required=True, help="Path to where the bag-of-visual-words database")
ap.add_argument("-m", "--model", required=True, help="Path to the output classifier")

args = vars(ap.parse_args())

# open the features and the bovw databases
featuresDB = h5py.File(args["features_db"])
bovwDB = h5py.File(args["bovw_db"])

print("[INFO] loading data ...")
trainData, trainLabels = bovwDB["bovw"][:300], featuresDB["image_ids"][:300]
testData, testLabels = bovwDB["bovw"][300:], featuresDB["image_ids"][300:]

# prepare the labels by removing the filename from the image ID -> use only class name
trainLabels = [l.split(":")[0] for l in trainLabels]
testLabels = [l.split(":")[0] for l in testLabels]

# define the grid of parameters to explore, then start the grid search where
# we evaluate a Linear SVM for each value of C
print("[INFO] tuning hyperparameters ...")
params = {"C": [0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0]}
model = GridSearchCV(LinearSVC(random_state=42), params, cv=3)
model.fit(trainData, trainLabels)
print("[INFO] best hyperparameters: {}".format(model.best_params_))
# show a classification report
print("[INFO] evaluating...")
predictions = model.predict(testData)
print(classification_report(testLabels, predictions))

# loop over a small sample from the test set
for i in np.random.choice(np.arange(300, 500), size=(20,), replace=False):
    # randomly grab a test image, load it, and classify
    label, filename = featuresDB["image_ids"][i].split(":")
    image = cv2.imread(f"{args['dataset']}/{label}/{filename}")
    prediction = model.predict(bovwDB["bovw"][i].reshape(1, -1))[0]

    # show the prediction
    print(f"[PREDICTION] {filename}:{prediction}")
    cv2.putText(image, prediction, (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
    cv2.imshow("Image", image)
    cv2.waitKey(0)

# close the databases
featuresDB.close()
bovwDB.close()

# dump the classification file
print("[INFO] dumping classifier to file ...")
f = open(args["model"], "wb")
f.write(pickle.dumps(model))
f.close()


