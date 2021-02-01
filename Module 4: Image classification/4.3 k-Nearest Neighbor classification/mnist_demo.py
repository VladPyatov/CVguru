from __future__ import print_function
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from sklearn import datasets
from skimage import exposure
import numpy as np
import sklearn
import imutils
import cv2

# handle older versions of sklearn
if int(sklearn.__version__.split(".")[1]) < 18:
    from sklearn.cross_validation import train_test_split
else:
    from sklearn.model_selection import train_test_split

# load MNIST digits dataset
mnist = datasets.load_digits()

# split mnist data: 0.75 for train, 0.25 for testing
(trainData, testData, trainLabels, testLabels) = train_test_split(np.array(mnist.data), mnist.target,
                                                                  test_size=0.5, random_state=42)
# split train data
(trainData, valData, trainLabels, valLabels) = train_test_split(trainData, trainLabels, test_size=0.1, random_state=84)

# show the sizes of each data split
print(f"training data points: {len(trainLabels)}")
print(f"validation data points: {len(valLabels)}")
print(f"testing data points: {len(testLabels)}")

# initialize the list of accuracies
kVals = range(1,30,2)
accuracies = []

# loop over various values of 'k' for the k-Nearest Neighbor classifier
for k in kVals:
    # train the kNN classifier with the current value of 'k
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(trainData, trainLabels)

    # evaluate the model and update the accuracies list
    score = model.score(valData, valLabels)
    accuracies.append(score)
    print(f"k={k}, accuracy={score*100:.2f}")

# the value of k that gives the largest accuracy
i = int(np.argmax(accuracies))
print(f"k={kVals[i]} achieved highest accuracy of {accuracies[i]*100:.2f} on validation data")

# retrain classifier using the best k value and predict the labels of the test data
model = KNeighborsClassifier(n_neighbors=kVals[i])
model.fit(trainData, trainLabels)
predictions = model.predict(testData)

# show a final classification report demonstrating the accuracy of the classifier for each of the digits
print("EVALUATION ON TESTING DATA")
print(classification_report(testLabels, predictions))

# loop over a few random digits
for i in list(map(int, np.random.randint(0,len(testLabels), size=(5,)))):
    # grab the image and classify it:
    image = testData[i]
    prediction = model.predict(image.reshape(1, -1))[0]

    image = image.reshape((8, 8)).astype("uint8")
    image = exposure.rescale_intensity(image, out_range=(0, 255))
    image = imutils.resize(image, width=16, inter=cv2.INTER_CUBIC)

    # show the prediction
    print("I think that digit is: {}".format(prediction))
    cv2.imshow("Image", image)
    cv2.waitKey(0)

