""" Project 4: Emotion Recognition

Captures live video from camera.
"""

import cv2
import numpy as np

face_cascade = cv2.CascadeClassifier('frontal_face.xml')

# ----------------
# Helper Functions
# ----------------

def get_faces(image, cascade, width=None, height=None):
    """Gets the faces of the image

    If there are no width and height parameters provided, we simply run the
    default CascadeClassifier

    We optimize CascadeClassifier by doing the following:
        Setting the minSize to half of the rolling average
        Setting the maxSize to 3 halves of the rolling average
        scaleFactor is a generic optimization

    Arguments:
        image: the image to recognize frames from
        cascade: the cascade to detect faces with
        width: the average width of the bounding boxes
        height: the average height of the bounding boxes

    Outputs:
        A 2D vector of the faces detected in the cascade
    """

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    if width is None or height is None:
        return cascade.detectMultiScale(gray)

    return cascade.detectMultiScale(gray,
                                    scaleFactor=1.3,
                                    minSize=(3*width/4, 3*height/4),
                                    maxSize=(3*width/2, 3*height/2))


# --------------
# Main Functions
# --------------

def video_feed():
    """Opens a window that displays live video feed

    Arguments:
        None

    Returns:
        Nothing
    """

    video = cv2.VideoCapture(0)

    # Take in the first frame for an estimate
    width, height = capture_first_frame(video)

    while True:
        # Capture frame-by-frame
        ret, frame = video.read()

        # Analyze each frame to detect a face
        frame = detect_faces(frame, width, height)

        # Display the resulting frame
        cv2.imshow('frame', frame)
        if cv2.waitKey(10) == 27:
            break

    video.release()
    cv2.destroyAllWindows()


def capture_first_frame(video):
    """Takes the first frame for an estimate

    Arguments:
        video: captured video from camera

    Returns:
        width: width of a detected face
        height: height of a detected face
    """

    ret, frame = video.read()

    faces = get_faces(frame, face_cascade)
    face = faces[0]
    width, height = face[2], face[3]

    return width, height


def detect_faces(frame, width, height):
    """Identifies a face within a frame using haar cascades

    Arguments:
        frame: a frame of the video
        width: width of a detected face
        height: height of a detected face

    Returns:
        Updated frame complete with bounding box around a face,
        should there be one
    """

    faces = get_faces(frame, face_cascade, width, height)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

    return frame

if __name__ == "__main__":
    video_feed()
