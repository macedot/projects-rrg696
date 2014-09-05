"""Project 0: Image Manipulation with OpenCV.

In this assignment, you will implement a few basic image
manipulation tasks using the OpenCV library.

Use the unit tests is image_manipulation_test.py to guide
your implementation, adding functions as needed until all
unit tests pass.
"""

import cv2
import numpy

def flip_image(original, horizontal, vertical):
    if horizontal and vertical:
        direction = -1
    elif horizontal:
        direction = 1
    elif vertical:
        direction = 0
    else:
        return original
    
    flipped = original

    cv2.flip(original, direction, flipped)
    
    return flipped

def negate_image(original):
    negated = 255-original

    return negated

def swap_blue_and_green(original):
    swapped = original

    b = swapped[:,:,1]
    g = swapped[:,:,0]
    r = swapped[:,:,2]

    swapped = cv2.merge((b,g,r))

    return swapped
