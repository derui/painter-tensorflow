# coding: utf-8

import cv2 as cv


def resize_image(img, fixed_size):

    correct_size = get_corrected_size(fixed_size, img.shape[1], img.shape[0])

    if correct_size < img.shape:
        img_resized = cv.resize(img, correct_size, interpolation=cv.INTER_AREA)
    else:
        img_resized = cv.resize(img, correct_size, interpolation=cv.INTER_CUBIC)

    img_cropped = img_resized[0:fixed_size, 0:fixed_size]
    return img_cropped


def get_corrected_size(fixed_size, width, height):
    """Detect the edge of an image is less than other side"""

    def correct_size(w, h):
        ratio = fixed_size / w
        return (max(w * ratio, fixed_size), h * ratio)

    size = correct_size(width, height)

    if size[0] >= fixed_size and size[1] >= fixed_size:
        return int(size[0]), int(size[1])

    size = correct_size(size[1], size[0])
    return (int(size[1]), int(size[0]))
