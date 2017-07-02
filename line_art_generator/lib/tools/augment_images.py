import random
import concurrent.futures
import hashlib
import os
import argparse
import cv2 as cv
import numpy as np
from . import util

argparser = argparse.ArgumentParser(
    description='Resize images that equals size of pair files')
argparser.add_argument(
    '--input_dir', type=str, required=True, help='the directory of original images')
argparser.add_argument(
    '--out_dir', type=str, required=True, help='the directory of line art images')
argparser.add_argument(
    '--size', type=int, default=128, help="size of image")
argparser.add_argument(
    '--count', type=int, default=200, help="count images augmented")

args = argparser.parse_args()


def make_out_path(directory, path):

    prefix = path[0:2]
    painted = os.path.join(directory, 'painted', prefix)
    line_art = os.path.join(directory, 'line_art', prefix)

    if not os.path.exists(painted):
        os.makedirs(painted, mode=0o755, exist_ok=True)

    if not os.path.exists(line_art):
        os.makedirs(line_art, mode=0o755, exist_ok=True)
 
    return os.path.join(painted, path), os.path.join(line_art, path)


def process(filename, count, directory, size):
    painted = cv.imread(filename['painted'])
    line_art = cv.imread(filename['line_art'])

    def make_hash(data):
        h = hashlib.sha1()
        h.update(data)
        return h.hexdigest()

    def inner_process(painted, line_art):
        height, width, _ = painted.shape
        y_pos = random.randrange(height - size)
        x_pos = random.randrange(width - size)

        painted = painted[y_pos:y_pos+size, x_pos:x_pos+size]
        line_art = line_art[y_pos:y_pos+size, x_pos:x_pos+size]

        name = make_hash(np.ndarray.tobytes(painted))
        name = name + '.png'

        painted_path, line_art_path = make_out_path(directory, name)

        cv.imwrite(painted_path, painted)
        cv.imwrite(line_art_path, line_art)

    for _ in range(count):
        inner_process(painted, line_art)


painted_list = []
line_art_list = []
for (root, _, files) in os.walk(os.path.join(args.input_dir, 'painted')):
    painted_list = [os.path.join(root, f) for f in files]
    painted_list = sorted(painted_list)

for (root, _, files) in os.walk(os.path.join(args.input_dir, 'line_art')):
    line_art_list = [os.path.join(root, f) for f in files]
    line_art_list = sorted(line_art_list)

file_list = [{'painted': p, 'line_art': l} for p, l in zip(painted_list, line_art_list)]

with concurrent.futures.ThreadPoolExecutor(6) as e:
    num = 0
    threads = [
        e.submit(process, path, args.count, args.out_dir, args.size)
        for path in file_list
    ]

    for future in concurrent.futures.as_completed(threads):
        try:
            future.result()
        except Exception as e:
            print("Error raised {}".format(e))
        else:
            num = num + 1
            if num % 100 == 0:
                print("processed {}".format(num))
