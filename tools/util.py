# coding: utf-8

import pathlib
import io
import math
import cv2 as cv


def get_obj_iterator(s3client, bucket, key_prefix):
    """Get an iterator to iterate all objects filtered by key_prefix"""

    paginator = s3client.get_paginator('list_objects')
    operation_parameters = {'Bucket': bucket,
                            'Prefix': key_prefix}
    page_iterator = paginator.paginate(**operation_parameters)

    return page_iterator


def get_obj_exclusion_detector(s3client, bucket, key):
    """Return function to exclude object from image processing with filename set"""

    ret = []
    with io.BytesIO() as f:
        s3client.download_fileobj(bucket, key, f)
        f.seek(0)

        ret = set(map(lambda x: x.decode('utf-8').strip(), f.readlines()))

    def func(key):
        path = pathlib.PurePath(key).with_suffix("")

        return path.name not in ret

    return func


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
