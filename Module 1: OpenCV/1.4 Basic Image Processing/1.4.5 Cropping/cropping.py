import cv2

image = cv2.imread("florida_trip.png")
cv2.imshow("Original", image)

# crop the face from the image
face = image[85:250, 85:220]
cv2.imshow("Face", face)
cv2.waitKey(0)

# crop the entire body
body = image[90:450, 0:290]
cv2.imshow("Body", body)
cv2.waitKey(0)