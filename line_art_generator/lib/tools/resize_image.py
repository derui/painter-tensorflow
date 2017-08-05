import hashlib
import os
import argparse
import cv2 as cv
import numpy as np
from . import util

argparser = argparse.ArgumentParser(
    description='Resize images that equals size of pair files')
argparser.add_argument(
    '--original_dir', type=str, help='the directory of original images')
argparser.add_argument(
    '--line_art_dir', type=str, help='the directory of line art images')
argparser.add_argument('-d', dest='out_dir', type=str, required=True)

args = argparser.parse_args()
MINIMUM_SIZE = 128

resize_factors = [
    6 / 7, 6 / 8, 6 / 9, 6 / 10, 6 / 11, 6 / 12, 6 / 13, 6 / 14, 6 / 15, 6 / 16,
    6 / 17, 6 / 18, 6 / 19, 6 / 20, 6 / 21, 6 / 22, 6 / 23, 6 / 24
]


def resize_as_original(origin, line_art):

    origin_shape = origin.shape
    line_art_shape = line_art.shape

    if origin_shape > line_art_shape:
        origin = util.resize_image(origin, list(reversed(line_art_shape[0:2])))
    elif origin_shape < line_art_shape:
        line_art = util.resize_image(line_art, list(reversed(origin_shape[0:2])))

    return origin, line_art


def resize_image(origin, line_art, factor):

    h, w, _ = origin.shape
    h = int(h * factor)
    w = int(w * factor)

    if h < MINIMUM_SIZE or w < MINIMUM_SIZE:
        return origin, line_art

    origin = cv.resize(origin, (w, h), interpolation=cv.INTER_AREA)
    line_art = cv.resize(line_art, (w, h), interpolation=cv.INTER_AREA)
    mx = np.max(line_art)
    mn = np.min(line_art)
    line_art = 255 * ((line_art - mn) / (mx - mn))

    return origin, line_art


def blur_image(origin, line_art, factor):

    origin = cv.GaussianBlur(origin, (5, 5), 0, 0)

    return origin, line_art


def alpha_to_white(img):
    if len(img.shape) > 2 and img.shape[2] > 3:
        alpha_channel = img[:, :, 3:4]
        h, w, _ = alpha_channel.shape
        alpha_channel = np.reshape(alpha_channel, [h, w])
        b, g, r, _ = cv.split(img)
        b[alpha_channel == 0] = 255
        g[alpha_channel == 0] = 255
        r[alpha_channel == 0] = 255

        img = cv.merge([b, g, r])

    if len(img.shape) != 2:
        img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    return img


def read_image(origin, line_art):
    origin_img = cv.imread(origin, -1)
    line_art_img = cv.imread(line_art, -1)

    if origin_img is None:
        raise Exception("can not load painted: %s" % (origin))

    if line_art_img is None:
        raise Exception("can not load line_art: %s" % (line_art))

    if origin_img.shape[2] > 3:
        origin_img = origin_img[:, :, :3]

    line_art_img = alpha_to_white(line_art_img)

    return origin_img, line_art_img


def write_image(origin, line_art, out_dir):
    def make_hash(data):
        h = hashlib.sha1()
        h.update(data)
        return h.hexdigest()

    h = make_hash(np.ndarray.tobytes(origin))
    origin_d = os.path.join(args.out_dir, 'painted', h[0:2])
    line_art_d = os.path.join(args.out_dir, 'line_art', h[0:2])

    if not os.path.exists(origin_d):
        os.makedirs(origin_d, 0o755, exist_ok=True)
    if not os.path.exists(line_art_d):
        os.makedirs(line_art_d, 0o755, exist_ok=True)

    writefname = os.path.join(origin_d, "{}{}".format(h, '.png'))
    cv.imwrite(writefname, origin)

    writefname = os.path.join(line_art_d, "{}{}".format(h, '.png'))
    cv.imwrite(writefname, line_art)


file_list = {}
for (root, _, files) in os.walk(args.original_dir):
    for f in files:
        fname, _ = os.path.splitext(f)
        file_list[fname] = [os.path.join(root, f)]

for (root, _, files) in os.walk(args.line_art_dir):
    for f in files:
        fname, _ = os.path.splitext(f)
        file_list[fname].append(os.path.join(root, f))

for k in file_list.keys():
    origin, line_art = read_image(file_list[k][0], file_list[k][1])
    origin, line_art = resize_as_original(origin, line_art)
    write_image(origin, line_art, args.out_dir)

    for factor in resize_factors:
        ro, rl = resize_image(origin, line_art, factor)
        write_image(ro, rl, args.out_dir)
