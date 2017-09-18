import os
import argparse
import pathlib
from datetime import datetime
import numpy as np
import cv2 as cv
import concurrent.futures
import queue
from . import util
from tflib import util as tfutil

argparser = argparse.ArgumentParser(description='Extract edge layer of a color image')
argparser.add_argument('input_dir', type=str, help='the directory included images to extract edge layer')
argparser.add_argument('-d', dest='out_dir', type=str, required=True)
argparser.add_argument('-s', '--size', dest='size', type=int)
argparser.add_argument('-e', dest='excludes_dir', type=str)

args = argparser.parse_args()

neiborhood8 = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]], np.uint8)
FIXED_SIZE = args.size


class Ignore(Exception):
    pass


def extract_edge(img):

    img_dilate = cv.dilate(img, neiborhood8, iterations=1)
    img_diff = cv.absdiff(img, img_dilate)
    img_diff_not = cv.bitwise_not(img_diff)
    img_diff_not = cv.cvtColor(img_diff_not, cv.COLOR_RGB2GRAY)

    if FIXED_SIZE is not None:
        img_diff_not = cv.adaptiveThreshold(img_diff_not, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 7, 8)
        img_diff_not = util.resize_image(img_diff_not, FIXED_SIZE)
        img_dilate = cv.erode(img_diff_not, neiborhood8, iterations=1)
        img_dilate = cv.dilate(img_diff_not, neiborhood8, iterations=1)
        img_diff = cv.absdiff(img_diff_not, img_dilate)
        img_diff_not = cv.bitwise_not(img_diff)
        # img_diff_not = cv.adaptiveThreshold(img_diff_not, 255,
        #                                 cv.ADAPTIVE_THRESH_GAUSSIAN_C,
        #                                 cv.THRESH_BINARY, 7, 8)

    img_diff_not = cv.cvtColor(img_diff_not, cv.COLOR_GRAY2RGB)

    return img_diff_not


def read_image(path):
    img = cv.imread(path)
    if img is None:
        raise Exception("OpenCV can not load %s" % (path))
    return img


def write_image(img, out_path):

    cv.imwrite(out_path, img)


excludes = []
if args.excludes_dir is not None:
    for r, _, files in os.walk(args.excludes_dir):
        excludes.extend(files)

if not pathlib.Path(args.out_dir).exists():
    os.makedirs(str(pathlib.Path(args.out_dir)), exist_ok=True)


image_processor = tfutil.make_generic_processor(read_image, write_image,
                                                extract_edge)

num = 0
for files, ignored_files in tfutil.walk_files(args.input_dir, excludes, 100):
    with concurrent.futures.ThreadPoolExecutor(max_workers=16) as e:
        futures = []
        for root, f in files:
            futures.append(
                e.submit(image_processor,
                         str(pathlib.Path(root) / f),
                         str(pathlib.Path(args.out_dir) / f)))

        for future in concurrent.futures.as_completed(futures):
            try:
                future.result()
            except Exception as exc:
                print('exception: %s' % exc)

    num += 100
    print('{}: Completed {} items, {} ignored.'.format(datetime.now(), num, ignored_files))

