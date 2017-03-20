import os
import argparse
import cv2 as cv
import concurrent.futures
from . import util

argparser = argparse.ArgumentParser(description='Resize image to fixed size')
argparser.add_argument(
    'input_dir',
    type=str,
    help='the directory of image to resize and crop to fixed size')
argparser.add_argument('-d', dest='out_dir', type=str, required=True)
argparser.add_argument('-s', '--size', dest='size', type=int)

args = argparser.parse_args()


def process(path, out_dir):

    img = cv.imread(path, cv.IMREAD_COLOR)
    if img is None:
        raise Exception("OpenCV can not load %s" % (path))

    img = util.resize_image(img, FIXED_SIZE)

    dirname, fname = os.path.split(os.path.abspath(path))
    prefix = fname[0:2]
    fname, ext = os.path.splitext(fname)
    d = os.path.join(out_dir, prefix)
    if not os.path.exists(d):
        os.makedirs(d, 0o755, exist_ok=True)

    writefname = os.path.join(d, "{}{}".format(fname, ext))

    cv.imwrite(writefname, img)


FIXED_SIZE = 512 if args.size is None else args.size

with concurrent.futures.ThreadPoolExecutor(max_workers=6) as executor:
    futures = {}
    for (r, _, files) in os.walk(args.input_dir):
        for f in files:
            futures[executor.submit(process, os.path.join(r, f),
                                    args.out_dir)] = f

    print('Number of resizing images {}'.format(len(futures.items())))

    num = 0
    for future in concurrent.futures.as_completed(futures):
        path = futures[future]
        try:
            future.result()
        except Exception as exc:
            print('%s generated as exception: %s' % (path, exc))
        else:
            num += 1
            if num % 100 == 0:
                print('Completed {} files'.format(num))
