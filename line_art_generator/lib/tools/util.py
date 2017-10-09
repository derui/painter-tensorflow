# coding: utf-8

import cv2 as cv


def resize_image(img, size):

    img_size = list(reversed(img.shape[0:2]))
    if size < img_size:
        img_resized = cv.resize(img, tuple(size), interpolation=cv.INTER_AREA)
    else:
        img_resized = cv.resize(img, tuple(size), interpolation=cv.INTER_CUBIC)

    return img_resized
