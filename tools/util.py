# coding: utf-8

import cv2 as cv
import os
import queue


# Return a new priority queue contains pathes prioritized with inode
def make_sequential_queue(directory, excludes=[]):
    q = queue.PriorityQueue()
    all_files = {}

    for root, _, files in os.walk(directory):
        all_files.update({
            os.stat(os.path.join(root, f)).st_ino: os.path.join(root, f)
            for f in files
        })

    keys = sorted(all_files.keys())
    for key in keys:
        f = all_files[key]
        n = os.path.splitext(os.path.basename(f))
        if n in excludes:
            continue
        q.put((key, f))

    return q


# Return a new function to read with given function and path read from queue
def sequential_read_dir(rq, wq, read_func):
    def process():
        while True:
            _, path = rq.get()

            try:
                image = read_func(path)
                wq.put((image, path))
            except Exception as e:
                print('Exception raised in reader thread: {}'.format(e))

            rq.task_done()

    return process


# Return a new function to write data that are from queue via given function
def queue_writer(directory, wq, write_func):
    def process():
        while True:
            item = wq.get()

            write_func(item, directory)
            wq.task_done()

    return process


def resize_image(img, fixed_size):

    correct_size = get_corrected_size(fixed_size, img.shape[1], img.shape[0])

    if correct_size < img.shape:
        img_resized = cv.resize(img, correct_size, interpolation=cv.INTER_AREA)
    else:
        img_resized = cv.resize(
            img, correct_size, interpolation=cv.INTER_CUBIC)

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
