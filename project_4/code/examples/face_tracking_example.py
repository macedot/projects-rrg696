import cv2
import sys
import numpy as np

faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')
eyeCascade = cv2.CascadeClassifier('haarcascade_eye.xml')
mouthCascade = cv2.CascadeClassifier('haarcascade_mcs_mouth.xml')

video_capture = cv2.VideoCapture(0)

def findMouth(frame, eyes, faces):
    mouths = mouthCascade.detectMultiScale(
            frame,
            scaleFactor=1.2,
            minNeighbors=8,
            minSize=(30, 10),
            flags = cv2.CASCADE_SCALE_IMAGE)

    if len(mouths) <= 0:
        return 0, 0, 0, 0

    eyeBottom = 0

    for eye in eyes:
        eyeX, eyeY, eyeW, eyeH = eye
        if (eyeY + eyeH) > eyeBottom:
            eyeBottom = eyeY + eyeH

    faceX, faceY, faceW, faceH = faces[0]
    faceBottom = faceY + faceH

    for mouth in mouths:
        mouthX, mouthY, mouthW, mouthH = mouth
        if mouthY > (eyeBottom + 50) and (mouthY + mouthH) < faceBottom:
            return mouth

    return 0, 0, 0, 0

if __name__ == '__main__':
    while True:
        # Capture frame-by-frame
        ret, frame = video_capture.read()

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.equalizeHist(gray)

        faces = faceCascade.detectMultiScale(
            gray,
            scaleFactor=1.2,
            minNeighbors=6,
            minSize=(50, 50),
            flags = cv2.CASCADE_SCALE_IMAGE)

        eyes = eyeCascade.detectMultiScale(
            gray,
            scaleFactor=1.2,
            minNeighbors=6,
            minSize=(50, 50),
            flags = cv2.CASCADE_SCALE_IMAGE)


        # Draw a rectangle around the faces
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        # Draw a rectangle around the eyes
        if len(eyes) >= 2:
            eyes = eyes[:2]
        for (x, y, w, h) in eyes:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)

        # Draw a rectangle around the mouth
        if len(faces) > 0 and len(eyes) > 0:
            x, y, w, h = findMouth(gray2, eyes, faces)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

        # Display the resulting frame
        cv2.imshow('Video', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything is done, release the capture
    video_capture.release()
    cv2.destroyAllWindows()