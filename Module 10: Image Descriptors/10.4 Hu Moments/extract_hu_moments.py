import cv2
import imutils

image = cv2.imread("more_shapes_example.png")
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Compute the hu moments feature vector for the entire image and show it
moments = cv2.HuMoments(cv2.moments(image)).flatten()
print(f"Original moments: {moments}")
cv2.imshow("Image", image)
cv2.waitKey(0)

# find the contours of the three planes in the image
cnts = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)

# loop over each contour
for i, c in enumerate(cnts):
    # extract the ROI from the image and compute the Hu Moments feature
    # vector for the ROI
    x, y, w, h = cv2.boundingRect(c)
    roi = image[y:y+h, x:x+w]
    moments = cv2.HuMoments(cv2.moments(roi)).flatten()

    print(f"Moments for plain #{i}: {moments}")
    cv2.imshow(f"ROI #{i}", roi)
    cv2.waitKey(0)

