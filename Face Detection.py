import cv2
from random import randrange

# An advanced face detection app
__author__ = 'Michael Khoshahang'
# Video link: https://www.youtube.com/watch?v=XIrOM9oP3pA

# Load some pre-trained data on face frontals from opencv (haar cascade algorithm)
trained_face_data = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Capture video from webcam
camera = cv2.VideoCapture(0)

# Iterate forever over frames
while True:
    # Reading the current frame from the camera
    successful_frame_read, frame = camera.read()

    # Converting to grayscale
    grayscaled_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    face_coordinates = trained_face_data.detectMultiScale(grayscaled_frame)

    # Draw rectangles around the faces
    for (x, y, w, h) in face_coordinates:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (randrange(256), randrange(256), randrange(256)), randrange(11))

    # Display the image with the faces
    cv2.imshow('Face Detector', frame)

    if cv2.waitKey(1) == ord('q') or cv2.waitKey(1) == ord('x'):
        break

print('Operation completed successfully')
