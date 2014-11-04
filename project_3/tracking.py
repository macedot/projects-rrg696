"""Project 3: Tracking.

In this project, you'll track objects in videos.
"""

import os
import cv2
import cv2.cv as cv
import math
import numpy as np

# ----------------
# Helper Functions
# ----------------


def get_features(frame, feature_params):
    """Gets the feature points in the frame (image) given

    Arguments:
        frame: A frame (image) of the video
        feature_params: parameters to the Shi-Tomasi feature detector

    Outputs:
        a tuple of a gray scale image and the feature points found
    """

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    features = cv2.goodFeaturesToTrack(gray, mask=None, **feature_params)

    return (gray, features)


def get_frame_count(video):
    """Gets the frame count in the video

    First we try to get the number of frames by video.get(propid),
    however since this isn't supported for all videos, we iterate through the
    frames if this isn't supported

    Arguments:
        video: the video itself

    Outputs:
        The number of the frames in the video
    """
    num_frames = int(video.get(cv.CV_CAP_PROP_POS_AVI_RATIO))

    if num_frames > 0:
        return num_frames

    num_frames = 0

    # iterate through all the frames
    while True:
        _, frame = video.read()
        if frame is None:
            break

        num_frames += 1

    # reset the video to the start
    video.set(cv2.cv.CV_CAP_PROP_POS_FRAMES, 0)

    return num_frames


def get_background(video):
    """Function to compute the background of the video images

    This works by averaging all the frames in the video, and using video.get

    Arguments:
        video: the video itself

    Outputs:
        A background of the image from averaging
    """

    # get the initial frame as the background
    _, background = video.read()

    # get the number of frames inside of the video
    num_frames = get_frame_count(video)

    alpha = 0.5
    alpha_incr = 0.5 / num_frames

    while True:
        _, frame = video.read()
        if frame is None:
            break

        # update the alpha beta values
        alpha += alpha_incr
        beta = 1 - alpha

        # get the next background
        background = cv2.addWeighted(background, alpha, frame, beta, 0)

    # reset the video to the start
    video.set(cv.CV_CAP_PROP_POS_FRAMES, 0)

    return background


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


def find_coordinates(face):
    """Translates face's representation to a 4-tuple

    Arguments:
        face: A 1D array consisting of [x, y, w, h]

    Outputs:
        A 4-tuple of (x, y, x2, y2)
    """

    return (face[0],
            face[1],
            face[0] + face[2],
            face[1] + face[3])


def least_squares(face, width, height):
    """Calculates the least squares distance

    Arguments:
        face: the current face to calculate
        width: the current width to estimate the distance to
        height: the current height to estimate the distance to

    Outputs:
        The least-squares distance
    """

    return (width - face[2]) ** 2 + (height - face[3]) ** 2


def get_best_face(faces, width, height):
    """Gets the best face, according to a least squares fit

    Arguments:
        faces: the faces to evaluate
        width: the current width to estimate the distance to
        height: the current height to estimate the distance to

    Outputs:
        The best fit

    """

    # Only one face, so return
    if len(faces) == 1:
        return faces[0]

    # calculate least squares on the rest
    min_distance = least_squares(faces[0], width, height)
    min_index = 0

    for i in xrange(len(faces)):
        if least_squares(faces[i], width, height) < min_distance:
            min_index = i

    return faces[min_index]


def get_circle(image, radius=None):
    """Gets the circle in the image using HoughCircles

    The arguments are described as:
        gray: the grayscale input image
        cv.CV_HOUGH_GRADIENT: the function to use
        5: the dp (inverse scale to the accumulator)
        1000: the minimum distance between circles.
              Since we're only detecting one, we can make this
              an arbitrarily large number
        param1: The method specific parameter to CV_HOUGH_GRADIENT
        param2: The accumulator threshold, we set this to be a relatively big
        number (fine tuned) as there should be only one circle

        minRadius and maxRadius are fine-tuned parameters to speed up the
        process of choosing a circle
    """

    # no radius given, use defaults
    if radius is None:
        return cv2.HoughCircles(image,
                                cv.CV_HOUGH_GRADIENT,
                                dp=2,
                                minDist=1000,
                                param1=30,
                                param2=50)

    # else speed it up, with the minRadius and maxRadius
    return cv2.HoughCircles(image,
                            cv.CV_HOUGH_GRADIENT,
                            dp=5,
                            minDist=1000,
                            param1=30,
                            param2=50,
                            minRadius=int(radius - 3),
                            maxRadius=int(radius + 7))


def find_corners_of_circle(circle):
    """Finds the corners of the circle to append"""

    x, y, r = tuple(circle)

    return (int(x - r),
            int(y - r),
            int(x + r),
            int(y + r))

# --------------
# Main Functions
# --------------


def track_ball_1(video):
    """Track the ball's center in 'video'.

    Arguments:
      video: an open cv2.VideoCapture object containing a video of a ball
        to be tracked.

    Outputs:
      a list of (min_x, min_y, max_x, max_y) four-tuples containing the pixel
      coordinates of the rectangular bounding box of the ball in each frame.

    Notes:
    This function works as follows:
        First we compute the average background of the video by the function
        get_background (written above).

        Then using that background as the first parameter to the
        absdiff function, we subtract the background for every frame in
        the video.

        Then we find the contours in this processed frame, and draw a bounding
        box around it, and append to coordinates

        This function is used in track_ball_2 and track_ball_3 similarly
    """

    # The list to be returned
    coordinates = []

    # get the background of the video for a 100 frames
    background = get_background(video)

    # iterate through all the frames
    while True:
        _, frame = video.read()
        if frame is None:
            break

        # subtract the background, convert to grayscale and threshold
        diff = cv2.absdiff(frame, background)
        diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        _, threshold = cv2.threshold(diff, 100, 255, cv2.THRESH_BINARY)

        # find the contours
        contours, _ = cv2.findContours(threshold,
                                       cv2.RETR_TREE,
                                       cv2.CHAIN_APPROX_NONE)

        areas = [cv2.contourArea(c) for c in contours]
        if len(areas) > 0:
            # get the largest contour possible
            max_idx = np.argmax(areas)
            cnt = contours[max_idx]

            # append to the list
            x, y, w, h = cv2.boundingRect(cnt)
            coordinates.append((x,
                                y,
                                x + w - 1,
                                y + h - 1))

    return coordinates


def track_ball_2(video):
    """The same as track_ball_1"""
    return track_ball_1(video)


def track_ball_3(video):
    """The same as track_ball_2"""
    return track_ball_1(video)


def track_ball_4(video):
    """This function uses HoughCircles to detect the circles

    Since we can assume that we are tracking a circle, we can detect the circle
    using Hough Circles, and then use those to calculate the bounding box of
    the circles themselves

    We take a rolling average of the radius, so we can speed up our
    HoughCircles to provide minRadius and maxRadius as we go
    """

    # The list to be returned
    coordinates = []

    # initialize the radius
    _, frame = video.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    circle = get_circle(gray)
    circle = circle.flatten()
    coordinates.append(find_corners_of_circle(circle))

    radius = circle[2]

    # keep track of how many frames we went through so far
    count = 1

    # iterate through all the frames
    while True:
        _, frame = video.read()
        if frame is None:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        circle = get_circle(gray, radius)

        # no suitable matches so append the last one again
        if circle is None:
            coordinates.append(coordinates[-1])
        else:
            circle = circle.flatten()
            coordinates.append(find_corners_of_circle(circle))

            # update the rolling radius
            r = circle[2]
            radius += (r - radius) / count

        count += 1

    return coordinates


def track_face(video):
    """Identifies the face in the video using CascadeClassifiers

    This function works by using a default CascadeClassifier provided by OpenCV
    and then running that on the each of the video's frames, and calculating
    the bounding box for each of those

    We can speed up the process by providing the CascadeClassifier a specified
    bounding box so we can filter outliers

    We can get this by providing the CascadeClassifier a "good image" and then
    finding an estimate of the dimensions of the bounding box off of that
    Here we assume that the estimate is the first frame in the video

    We then add that face to the list of matches
    """

    coordinates = []
    face_cascade = cv2.CascadeClassifier('cascades/frontal_face.xml')

    # take in the first frame for an estimate
    _, frame = video.read()

    faces = get_faces(frame, face_cascade)
    face = faces[0]
    width, height = face[2], face[3]

    coordinates.append(find_coordinates(face))

    # iterate through the remaining frames
    while True:
        _, frame = video.read()
        if frame is None:
            break

        faces = get_faces(frame, face_cascade, width, height)

        # no suitable matches, so append the last element again
        if len(faces) == 0:
            coordinates.append(coordinates[-1])
        else:
            best = get_best_face(faces, width, height)
            coordinates.append(find_coordinates(best))

    return coordinates
