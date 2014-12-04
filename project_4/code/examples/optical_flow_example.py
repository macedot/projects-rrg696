#!/usr/bin/env python

import numpy as np
import cv2

#For face
faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')
eyeCascade = cv2.CascadeClassifier('haarcascade_eye.xml')
mouthCascade = cv2.CascadeClassifier('haarcascade_mcs_mouth.xml')
mouthCascade = cv2.CascadeClassifier('mouth.xml')

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


def draw_flow(img, frame, flow, step=8):
    #Tracks face
    faces = faceCascade.detectMultiScale(
        img,
        scaleFactor=1.2,
        minNeighbors=6,
        minSize=(50, 50),
        flags = cv2.CASCADE_SCALE_IMAGE)

    #Tracks eyes
    eyes = eyeCascade.detectMultiScale(
        img,
        scaleFactor=1.2,
        minNeighbors=6,
        minSize=(50, 50),
        flags = cv2.CASCADE_SCALE_IMAGE)

    #Calculates the Optical Flow
    h, w = img.shape[:2]
    y, x = np.mgrid[step/2:h:step, step/2:w:step].reshape(2,-1)
    fx, fy = flow[y,x].T
    lines = np.vstack([x, y, x+fx, y+fy]).T.reshape(-1, 2, 2)
    lines = np.int32(lines + 0.5)
    vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    #Rectangle for each feature
    rectangle_face = [0, 0, 0, 0]
    rectangle_eye1 = [0, 0, 0, 0]
    rectangle_eye2 = [0, 0, 0, 0]
    rectangle_mouth = [0, 0, 0, 0]

    new_lines_face = []
    new_lines_eye1 = []
    new_lines_eye2 = []
    new_lines_mouth = []

    mouth = []

    #Gets the coordinates for each feature
    for (x, y, w, h) in faces:
        rectangle_face = [x, y, x+w, y+h]

    if (len(eyes) > 0):
        x, y, w, h = eyes[0]
        rectangle_eye1 = [x, y, x+w, y+h]
    if (len(eyes) > 1):
        x, y, w, h = eyes[1]
        rectangle_eye2 = [x, y, x+w, y+h]

    if len(faces) > 0 and len(eyes) > 0:
        mouth = findMouth(vis, eyes, faces)
        x, y, w, h = mouth
        rectangle_mouth = [x, y, x+w, y+h]

    #Number of lines moving/Optical Flow for Face
    for (x1, y1), (x2, y2) in lines:
        if rectangle_face[0] < x1 < rectangle_face[2] and rectangle_face[1] < y1 < rectangle_face[3] :
            new_lines_face.append([[x1 , y1],[x2, y2]])
        if rectangle_eye1[0] < x1 < rectangle_eye1[2] and rectangle_eye1[1] < y1 < rectangle_eye1[3] :
            new_lines_eye1.append([[x1 , y1],[x2, y2]])
        if rectangle_eye2[0] < x1 < rectangle_eye2[2] and rectangle_eye2[1] < y1 < rectangle_eye2[3] :
            new_lines_eye2.append([[x1 , y1],[x2, y2]])

        if rectangle_mouth[0] < x1 < rectangle_mouth[2] and rectangle_mouth[1] < y1 < rectangle_mouth[3] :
            new_lines_mouth.append([[x1 , y1],[x2, y2]])


    #Puts all the points into a nice numpy array
    new_lines_face = np.array(new_lines_face)
    new_lines_eye1 = np.array(new_lines_eye1)
    new_lines_eye2 = np.array(new_lines_eye2)
    new_lines_mouth = np.array(new_lines_mouth)

    #Draws optical flow lines
    #lines are consist of such information => [ original_x, original_y , later_x, later_y ]
    # cv2.polylines(vis, new_lines_face, 0, (0, 255, 0))
    cv2.polylines(frame, new_lines_eye1, 0, (255, 0, 0))
    cv2.polylines(frame, new_lines_eye2, 0, (255, 0, 0))
    cv2.polylines(frame, new_lines_mouth, 0, (0, 0, 255))

    #Draws the optical flow origin point for face and eyes
    for (x1, y1), (x2, y2) in new_lines_face:
        cv2.circle(frame, (x1, y1), 1, (0, 255, 0), -1)

    for (x1, y1), (x2, y2) in new_lines_eye1:
        cv2.circle(frame, (x1, y1), 1, (255, 0, 0), -1)

    for (x1, y1), (x2, y2) in new_lines_eye2:
        cv2.circle(frame, (x1, y1), 1, (255, 0, 0), -1)

    for (x1, y1), (x2, y2) in new_lines_mouth:
        cv2.circle(frame, (x1, y1), 1, (0, 0, 255), -1)

    # #Draws face rectangle
    # for (x, y, w, h) in faces:
    #     cv2.rectangle(vis, (x, y), (x+w, y+h), (0, 255, 0), 1)
    #
    # #Draws eye/s rectangle
    # if len(eyes) >= 2:
    #     eyes = eyes[:2]
    # for (x, y, w, h) in eyes:
    #     cv2.rectangle(vis, (x, y), (x+w, y+h), (255, 0, 0), 1)
    #
    # if len(mouth) > 0:
    #      x, y, w, h = mouth
    #      cv2.rectangle(vis, (x, y), (x+w, y+h), (0, 0, 255), 1)

    return frame

if __name__ == '__main__':
    cam = cv2.VideoCapture(0)  #video.create_capture(fn)
    ret, prev = cam.read()
    prevgray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)

    while True:
        ret, img = cam.read()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        flow = cv2.calcOpticalFlowFarneback(prevgray, gray, 0, 5, 3, 2, 3, 5, 1, 0)
        prevgray = gray

        # showFrame, moving, emotion = draw_flow(gray, flow)
        showFrame = draw_flow(gray, img, flow)

        cv2.imshow('flow', showFrame)
        ch = 0xFF & cv2.waitKey(1)
        if ch == 27:
            break

    cv2.destroyAllWindows()