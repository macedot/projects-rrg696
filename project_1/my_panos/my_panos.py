import cv2
import pano_stitcher
import numpy

def main():
    cards_1 = cv2.imread("cards_1.jpg")
    cards_2 = cv2.imread("cards_2.jpg")
    cards_3 = cv2.imread("cards_3.jpg")

    H_1 = pano_stitcher.homography(cards_2, cards_1)
    H_2 = pano_stitcher.homography(cards_2, cards_2)
    H_3 = pano_stitcher.homography(cards_2, cards_3)

    cards_1_warped, cards_1_origin = pano_stitcher.warp_image(cards_1, H_1)
    cards_2_warped, cards_2_origin = pano_stitcher.warp_image(cards_2, H_2)
    cards_3_warped, cards_3_origin = pano_stitcher.warp_image(cards_3, H_3)

    images = (cards_1_warped, cards_3_warped, cards_2_warped)
    origins = (cards_1_origin, cards_3_origin, cards_2_origin)

    pano = pano_stitcher.create_mosaic(images, origins)

    cv2.imwrite("pano.png", pano)

if __name__ == "__main__":
    main()
