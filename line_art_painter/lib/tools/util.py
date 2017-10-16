# coding: utf-8

import math
import cv2 as cv


def resize_image(img, fixed_size, crop=False):

    correct_size = get_corrected_size(fixed_size, img.shape[1], img.shape[0])

    if correct_size < img.shape:
        img_resized = cv.resize(img, correct_size, interpolation=cv.INTER_AREA)
    else:
        img_resized = cv.resize(img, correct_size, interpolation=cv.INTER_CUBIC)

    if crop:
        img_cropped = img_resized[0:fixed_size, 0:fixed_size]
        return img_cropped
    return img_resized


def get_corrected_size(fixed_size, width, height):
    """Detect the edge of an image is less than other side"""

    ratio = 0
    if width <= height:
        ratio = fixed_size / width
    else:
        ratio = fixed_size / height

    return (math.ceil(ratio * width), math.ceil(ratio * height))
