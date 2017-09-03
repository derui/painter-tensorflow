import os
import argparse
import cv2 as cv
import concurrent.futures
from datetime import datetime
from . import util
from tflib import util as tfutil

argparser = argparse.ArgumentParser(description='Resize image to fixed size')
argparser.add_argument(
    'input_dir',
    type=str,
    help='the directory of image to resize and crop to fixed size')
argparser.add_argument('-d', dest='out_dir', type=str, required=True)
argparser.add_argument('-e', dest='excludes_dir', type=str)
argparser.add_argument('-s', '--size', dest='size', type=int)

args = argparser.parse_args()

FIXED_SIZE = 512 if args.size is None else args.size


def read_image(path):
    img = cv.imread(path)
    if img is None:
        raise Exception("OpenCV can not load %s" % (path))
    return img


def write_image(img, out_path):

    dirname, fname = os.path.split(os.path.abspath(out_path))
    if not os.path.exists(dirname):
        os.makedirs(dirname, 0o755, exist_ok=True)

    if len(img.shape) == 2:
        img = cv.cvtColor(img, cv.COLOR_GRAY2RGB)

    cv.imwrite(out_path, img)


def process(img):

    return util.resize_image(img, FIXED_SIZE)


def to_out_path(out_dir, f):
    name, _ = os.path.splitext(f)
    return os.path.join(out_dir, f[:2], "{}.png".format(name))


image_processor = tfutil.make_generic_processor(read_image, write_image,
                                              process)

excludes = []
if args.excludes_dir is not None:
    excludes = tfutil.load_exclude_names(args.excludes_dir)

num = 0
for files, ignored_files in tfutil.walk_files(args.input_dir, excludes, 100):
    with concurrent.futures.ThreadPoolExecutor(max_workers=16) as e:
        futures = []
        for root, f in files:
            futures.append(
                e.submit(image_processor,
                         os.path.join(root, f), to_out_path(args.out_dir, f)))

        for future in concurrent.futures.as_completed(futures):
            try:
                future.result()
            except Exception as exc:
                print('exception: %s' % exc)

    num += 100
    print('{}: Completed {} items, {} ignored.'.format(datetime.now(), num, ignored_files))
