"""Project 2: Stereo vision.

In this project, you'll extract dense 3D information from stereo image pairs.
"""

import cv2
import math
import numpy as np
import StringIO


def rectify_pair(image_left, image_right, viz=False):
    """Computes the pair's fundamental matrix and rectifying homographies.

    Arguments:
      image_left, image_right: 3-channel images making up a stereo pair.

    Returns:
      F: the fundamental matrix relating epipolar geometry between the pair.
      H_left, H_right: homographies that warp the left and right image so
        their epipolar lines are corresponding rows.
    """

    # Initiate SIFT detector
    sift = cv2.SIFT()

    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(image_left, None)
    kp2, des2 = sift.detectAndCompute(image_right, None)

    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)

    flann = cv2.FlannBasedMatcher(index_params, search_params)

    matches = flann.knnMatch(des1, des2, k=2)

    # store all the good matches as per Lowe's ratio test
    good = []
    for m, n in matches:
        if m.distance < 0.68 * n.distance:
            good.append(m)

    src_pts = np.float32([kp1[m.queryIdx].pt for m in good])
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good])

    # find the fundamental matrix
    F, mask = cv2.findFundamentalMat(src_pts, dst_pts, cv2.RANSAC, 3, 0.99)
    src_pts = src_pts.flatten()
    dst_pts = dst_pts.flatten()

    # rectify the images, produce the homographies: H_left and H_right
    retval, H_left, H_right = cv2.stereoRectifyUncalibrated(
        src_pts, dst_pts, F, image_left.shape[:2])

    return F, H_left, H_right


def disparity_map(image_left, image_right):
    """Compute the disparity images for image_left and image_right.

    Arguments:
      image_left, image_right: rectified stereo image pair.

    Returns:
      an single-channel image containing disparities in pixels,
        with respect to image_left's input pixels.
    """

    window_size = 3
    min_disp = 16
    num_disp = 112 - min_disp
    stereo = cv2.StereoSGBM(minDisparity=min_disp,
                            numDisparities=num_disp,
                            SADWindowSize=window_size,
                            uniquenessRatio=10,
                            speckleWindowSize=100,
                            speckleRange=32,
                            disp12MaxDiff=1,
                            preFilterCap=20,
                            P1=8 * 3 * (window_size ** 2),
                            P2=32 * 3 * (window_size ** 2),
                            fullDP=False
                            )

    # compute the disparity and convert the 16bit disparity into 8 bit
    disp = stereo.compute(image_left, image_right) / 16
    disp = disp.astype('uint8')

    return disp

# PLY file header
ply_header = '''ply
format ascii 1.0
element vertex %(vert_num)d
property float x
property float y
property float z
property uchar red
property uchar green
property uchar blue
end_header
'''


def point_cloud(disparity_image, image_left, focal_length):
    """Create a point cloud from a disparity image and a focal length.

    Arguments:
      disparity_image: disparities in pixels.
      image_left: BGR-format left stereo image, to color the points.
      focal_length: the focal length of the stereo camera, in pixels.

    Returns:
      A string containing a PLY point cloud of the 3D locations of the
        pixels, with colors sampled from left_image. You may filter low-
        disparity pixels or noise pixels if you choose.
    """
    h, w = image_left.shape[:2]
    Q = np.float32([[1, 0,  0, w / 2],
                    [0, -1,  0,  h / 2],  # turn points 180 deg around x-axis,
                    [0, 0, focal_length,  0],  # so that y-axis looks up
                    [0, 0,  0,  1]])

    # reproject image points to 3D space, compute the colors
    points = cv2.reprojectImageTo3D(disparity_image, Q)
    colors = cv2.cvtColor(image_left, cv2.COLOR_BGR2RGB)
    mask = disparity_image > disparity_image.min()
    out_points = points[mask]
    out_colors = colors[mask]

    # write PLY string data to StringIO object and return the contents
    cloud = StringIO.StringIO()
    verts = np.hstack([out_points, out_colors])
    cloud.write(ply_header % dict(vert_num=len(verts)))
    np.savetxt(cloud, verts, '%f %f %f %d %d %d')

    return cloud.getvalue()
