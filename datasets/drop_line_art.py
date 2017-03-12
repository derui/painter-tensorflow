import os
import argparse
import numpy as np
import cv2 as cv
import concurrent.futures

argparser = argparse.ArgumentParser(description='Drop images are line-art ')
argparser.add_argument(
    'input_dir', type=str, help='the directory images included')
argparser.add_argument('-d', dest='out_dir', type=str, required=True)

args = argparser.parse_args()


def is_line_art(img):
    """Detect color image"""
    grayed = cv.cvtColor(img, cv.COLOR_RGB2GRAY)

    r_diff = np.abs(img[::, ::, 2] - grayed)
    g_diff = np.abs(img[::, ::, 1] - grayed)
    b_diff = np.abs(img[::, ::, 0] - grayed)

    thresholds = 25
    likelihood = 0.8
    diffs = np.array(
        [r_diff < thresholds, g_diff < thresholds, b_diff < thresholds])

    return np.alltrue(diffs > likelihood)


def drop_line_art(path, out_dir):

    img = cv.imread(path, cv.IMREAD_COLOR)
    if img is None:
        raise Exception("OpenCV can not load %s" % (path))

    if is_line_art(img):
        print('Ignore {}'.format(path))
        return

    dirname, fname = os.path.split(os.path.abspath(path))
    fname, ext = os.path.splitext(fname)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir, 0o755, exist_ok=True)

    writefname = "%s/%s%s" % (out_dir, fname, ext)

    cv.imwrite(writefname, img)


with concurrent.futures.ThreadPoolExecutor(max_workers=6) as executor:
    futures = {}
    for (r, _, files) in os.walk(args.input_dir):
        for f in files:
            futures[executor.submit(drop_line_art,
                                    os.path.join(r, f), args.out_dir)] = f

    print('Number of images {}'.format(len(futures.items())))

    for future in concurrent.futures.as_completed(futures):
        path = futures[future]
        try:
            future.result()
        except Exception as exc:
            print('%s generated as exception: %s' % (path, exc))
        else:
            print('%s is not line-art' % path)
