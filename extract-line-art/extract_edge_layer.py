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

def extract_edge(path, out_dir):

    img = cv.imread(path, cv.IMREAD_COLOR)
    if img is None:
        raise Exception("OpenCV can not load %s" % (path))

    canny = cv.Canny(img, 50.0, 200.0, None, 3, True)

    bitwise = np.copy(canny)

    np.bitwise_not(canny, bitwise)

    dirname, fname = os.path.split(os.path.abspath(path))
    fname, ext = os.path.splitext(fname)
    if not os.path.exists(out_dir):
        os.mkdir(out_dir, 0o755)

    writefname = "%s/%s%s" % (out_dir, fname, ext)

    cv.imwrite(writefname, bitwise)

with concurrent.futures.ThreadPoolExecutor(max_workers=6) as executor:
    futures = {}
    for (r, _, files) in os.walk(args.input_dir):
        for f in files:
            futures[executor.submit(extract_edge, os.path.join(r, f), args.out_dir)] = f

    print('Number of resizing images {}'.format(len(futures.items())))

    for future in concurrent.futures.as_completed(futures):
        path = futures[future]
        try:
            future.result()
        except Exception as exc:
            print('%s generated as exception: %s' % (path, exc))
        else:
            print('%s is completed extraction of edge' % path)
