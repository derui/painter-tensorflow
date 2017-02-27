import os
import argparse
import numpy as np
import cv2 as cv
import concurrent.futures

argparser = argparse.ArgumentParser(description='Resize image to fixed size')
argparser.add_argument(
    'input_dir',
    type=str,
    help='the directory of image to resize and crop to fixed size')
argparser.add_argument('-d', dest='out_dir', type=str, required=True)
argparser.add_argument('-s', '--size', dest='size', type=int)

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


def resize_image(path, out_dir):

    img = cv.imread(path, cv.IMREAD_COLOR)
    if img is None:
        raise Exception("OpenCV can not load %s" % (path))

    if not is_color_image(img):
        print('Ignore {}'.format(path))
        return

    correct_size = get_corrected_size(FIXED_SIZE, img.shape[1], img.shape[0])
    img_resized = cv.resize(img, correct_size, interpolation=cv.INTER_CUBIC)

    img_cropped = img_resized[0:FIXED_SIZE, 0:FIXED_SIZE]

    dirname, fname = os.path.split(os.path.abspath(path))
    fname, ext = os.path.splitext(fname)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir, 0o755, exist_ok=True)

    writefname = "%s/%s%s" % (out_dir, fname, ext)

    cv.imwrite(writefname, img_cropped)


def get_corrected_size(fixed_size, width, height):
    """Detect the edge of an image is less than other side"""

    def correct_size(w, h):
        if w >= fixed_size:
            return (w, h)
        else:
            ratio = fixed_size / w
            return (max(w * ratio, fixed_size), h * ratio)

    size = correct_size(width, height)
    size = correct_size(size[1], size[0])
    return (int(size[1]), int(size[0]))


FIXED_SIZE = 512 if args.size is None else args.size

with concurrent.futures.ThreadPoolExecutor(max_workers=6) as executor:
    futures = {}
    for (r, _, files) in os.walk(args.input_dir):
        for f in files:
            futures[executor.submit(resize_image,
                                    os.path.join(r, f), args.out_dir)] = f

    print('Number of resizing images {}'.format(len(futures.items())))

    for future in concurrent.futures.as_completed(futures):
        path = futures[future]
        try:
            future.result()
        except Exception as exc:
            print('%s generated as exception: %s' % (path, exc))
        else:
            print('%s is resized' % path)
