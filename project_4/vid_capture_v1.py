""" Project 4: Emotion Recognition

Captures live video from camera.
"""

import cv2
import numpy as np

face_cascade = cv2.CascadeClassifier('frontal_face.xml')

def video_feed():
    """Opens a window that displays live video feed

    Arguments:
        None

    Returns:
        Nothing
    """

    cap = cv2.VideoCapture(0)

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()

        # Analyze each frame to detect a face
        frame = detect_faces(frame)

        # Display the resulting frame
        cv2.imshow('frame', frame)
        if cv2.waitKey(10) == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

def detect_faces(frame):
    """Identifies a face within a frame using haar cascades

    Arguments:
        frame: a frame of the video

    Returns:
        Updated frame complete with bounding box around a face,
        should there be one
    """

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        roi_gray = gray[y:y + h, x:x + w]

    return frame

if __name__ == "__main__":
    video_feed()
