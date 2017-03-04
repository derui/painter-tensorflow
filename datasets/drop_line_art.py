import os
import argparse
import numpy as np
import cv2 as cv
import concurrent.futures

argparser = argparse.ArgumentParser(description='Drop images are line-art ')
argparser.add_argument(
    'input_dir',
    type=str,
    help='the directory images included')
argparser.add_argument('-d', dest='out_dir', type=str, required=True)

args = argparser.parse_args()


def is_color_image(img):
    """Detect color image"""
    grayed = cv.cvtColor(img, cv.COLOR_RGB2GRAY)

    grayed_mean = np.mean(grayed)

    r_mean = np.mean(img[::-1, ::-1, 2])
    g_mean = np.mean(img[::-1, ::-1, 1])
    b_mean = np.mean(img[::-1, ::-1, 0])

    thresholds = 0.95
    means = np.array(
        [r_mean / grayed_mean, g_mean / grayed_mean, b_mean / grayed_mean])

    return not np.alltrue(means > thresholds)


def drop_line_art(path, out_dir):

    img = cv.imread(path, cv.IMREAD_COLOR)
    if img is None:
        raise Exception("OpenCV can not load %s" % (path))

    if not is_color_image(img):
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
