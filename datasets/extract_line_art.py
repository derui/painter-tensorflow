import os
import argparse
import numpy as np
import cv2 as cv
import concurrent.futures

argparser = argparse.ArgumentParser(
    description='Extract edge layer of a color image')
argparser.add_argument(
    'input_dir',
    type=str,
    help='the directory included images to extract edge layer')
argparser.add_argument('-d', dest='out_dir', type=str, required=True)

args = argparser.parse_args()

neiborhood8 = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]], np.uint8)


def extract_edge(path, out_dir):

    img = cv.imread(path, cv.IMREAD_GRAYSCALE)
    if img is None:
        raise Exception("OpenCV can not load %s" % (path))
    at = cv.adaptiveThreshold(img, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C,
                              cv.THRESH_BINARY, 7, 8)
    img_dilate = cv.dilate(at, neiborhood8, iterations=1)
    img_diff = cv.absdiff(at, img_dilate)
    img_diff_not = cv.bitwise_not(img_diff)

    dirname, fname = os.path.split(os.path.abspath(path))
    fname, ext = os.path.splitext(fname)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir, 0o755, exist_ok=True)

    writefname = "%s/%s%s" % (out_dir, fname, ext)

    cv.imwrite(writefname, img_diff_not)


with concurrent.futures.ThreadPoolExecutor(max_workers=6) as executor:
    futures = {}
    for (r, _, files) in os.walk(args.input_dir):
        for f in files:
            futures[executor.submit(extract_edge,
                                    os.path.join(r, f), args.out_dir)] = f

    print('Number of resizing images {}'.format(len(futures.items())))

    for future in concurrent.futures.as_completed(futures):
        path = futures[future]
        try:
            future.result()
        except Exception as exc:
            print('%s generated as exception: %s' % (path, exc))
        else:
            print('%s is completed extraction of edge' % path)
