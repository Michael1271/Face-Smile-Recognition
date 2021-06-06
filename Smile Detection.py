import cv2
from random import randrange

# An advanced smile detection app
__author__ = 'Michael Khoshahang'

# Load some pre-trained data on face frontals and smiles from opencv (haar cascade algorithm)
# Face & Smile classifiers
face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
smile_detector = cv2.CascadeClassifier('haarcascade_smile.xml')

# Capture video from webcam
camera = cv2.VideoCapture(0)

# Iterate forever over frames
while True:
    # Reading the current frame from the camera
    successful_frame_read, frame = camera.read()

    if not successful_frame_read:
        break

    # Converting to grayscale
    grayscaled_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces first
    face_coordinates = face_detector.detectMultiScale(grayscaled_frame)

    # Run face detection within each of the faces
    for (x, y, w, h) in face_coordinates:
        # Draw rectangles around the faces
        cv2.rectangle(frame, (x, y), (x + w, y + h), (randrange(256), randrange(256), randrange(256)), randrange(11))
        face = frame[y:y + h, x:x + w]  # get the sub frame (using numpy N-dimensional array slicing)

        # Converting to grayscale
        grayscaled_face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)

        # scaleFactor blurs the picture and makes it easier to detect faces
        smile_coordinates = smile_detector.detectMultiScale(grayscaled_face, scaleFactor=1.7, minNeighbors=20)

        # Run smile detection within each of the smiles
        for (x_, y_, w_, h_) in smile_coordinates:
            # Draw rectangles around the smiles
            cv2.rectangle(face, (x_, y_), (x_ + w_, y_ + h_), (randrange(256), randrange(256), randrange(256)), randrange(11))

        # Label the face as smiling
        if len(smile_coordinates) > 0:
            cv2.putText(frame, 'Smiling', (x, y + h + 40), fontScale=3, fontFace=cv2.FONT_HERSHEY_PLAIN, color=(randrange(256), randrange(256), randrange(256)))
    # Display the image with the faces
    cv2.imshow('Smile Detector', frame)

    if cv2.waitKey(1) == ord('q') or cv2.waitKey(1) == ord('x'):
        break

# Cleanup
camera.release()
cv2.destroyAllWindows()
print('Operation completed successfully')
