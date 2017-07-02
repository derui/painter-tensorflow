# coding: utf-8

import cv2 as cv
import os
import queue
import time


def load_exclude_names(excludes_dir):
    """load file names from the directory.

    @return list that files should exclude in process. Not contained extension of file
            in list.
    """
    excludes = []
    for r, _, files in os.walk(excludes_dir):
        names = [v for v, _ in map(lambda x: os.path.splitext(x), files)]
        excludes.extend(names)

    return set(excludes)


def walk_images(image_dir, exclude_files, per_yield_files):

    return_files = []
    ignored_files = 0
    for root, _, files in os.walk(image_dir):
        for f in files:
            n, _ = os.path.splitext(os.path.basename(f))
            if n in exclude_files:
                ignored_files += 1
                continue

            return_files.append((root, f))

            if len(return_files) >= per_yield_files:
                yield (return_files, ignored_files)

                return_files = []

    if len(return_files) > 0:
        yield (return_files, ignored_files)


def make_generic_image_processor(read_func, write_func, process):
    """make generic-image-processor funciton.

    generic image processor has three processes: reader, writer, and image processing.
    it accept only two arguments, such as input path and output path.

    @return new generic image processor
    """
    def func(in_path, out_path):
        img = read_func(in_path)
        img = process(img)
        write_func(img, out_path)

    return func


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
