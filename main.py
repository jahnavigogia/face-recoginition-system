from typing import Any

import cv2
import glob

from cv2 import UMat, Mat
from numpy import ndarray, dtype, generic

count = 1

# Load the pre-trained Haarcascades classifier
face_cascade = cv2.CascadeClassifier('/Users/jahnavi/PycharmProjects/Face-Recoginition-System/deep-learning-model.xml')

# Initialize the webcam
cap = cv2.VideoCapture(0)

while True:
    # Read a frame from the webcam
    ret, frame = cap.read()

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    # Draw rectangles around the faces
    for (x, y, w, h) in faces:
        rect = cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

        if rect.any():
            status = 1
            cv2.imwrite(f"images/{count}.png", frame)
            count = count + 1

    # Display the frame
    cv2.imshow('Face Recognition', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


# Release the webcam and close the OpenCV window
cap.release()
cv2.destroyAllWindows()
