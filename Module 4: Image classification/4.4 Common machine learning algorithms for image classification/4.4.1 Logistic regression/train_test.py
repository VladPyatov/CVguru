from __future__ import print_function
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from sklearn import datasets
import numpy as np
import imutils
import cv2
import sklearn

# handle older versions of sklearn
if int(sklearn.__version__.split(".")[1]) < 18:
    from sklearn.cross_validation import train_test_split
else:
    from sklearn.model_selection import train_test_split

print("[INFO] fetching data...")
dataset = datasets.fetch_lfw_people(min_faces_per_person=70, funneled=True, resize=0.5)
(trainData, testData, trainLabels, testLabels) = train_test_split(dataset.data, dataset.target,
                                                                  test_size=0.25, random_state=42)

# train the model and show the classification report
print("[INFO] training model...")
model = LogisticRegression()
model.fit(trainData, trainLabels)
print(classification_report(testLabels, model.predict(testData), target_names=dataset.target_names))

# loop over a 10 random images
for i in list(map(int, np.random.randint(0, testLabels.shape[0], size=(10,)))):
    # grab the image and the name, then resize the image so we can better see it
    image = testData[i].reshape((62, 47))
    name = dataset.target_names[testLabels[i]]
    image = imutils.resize(image.astype("uint8"), width=image.shape[1]*3, inter=cv2.INTER_CUBIC)

    # classify the face
    prediction = model.predict(testData[i].reshape(1, -1))[0]
    prediction = dataset.target_names[prediction]

    print(f"[PREDICTION] predicted: {prediction}, actual: {name}")
    cv2.imshow("Face", image)
    cv2.waitKey(0)
