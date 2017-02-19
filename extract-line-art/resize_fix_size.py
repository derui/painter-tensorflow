import os
import argparse
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

def resize_image(path, out_dir):

    img = cv.imread(path)
    if img is None:
        raise Exception("OpenCV can not load %s" % (path))

    correct_size = get_corrected_size(FIXED_SIZE, img.shape[1], img.shape[0])
    img_resized = cv.resize(img, correct_size, interpolation=cv.INTER_CUBIC)

    img_cropped = img_resized[0:FIXED_SIZE, 0:FIXED_SIZE]

    dirname, fname = os.path.split(os.path.abspath(path))
    fname, ext = os.path.splitext(fname)
    if not os.path.exists(out_dir):
        os.mkdir(out_dir, 0o755)

    writefname = "%s/%s%s" % (out_dir, fname, ext)

    cv.imwrite(writefname, img_cropped)


def get_corrected_size(fixed_size, width, height):
    """Detect the edge of an image is less than other side"""

    def correct_size(w, h):
        if w >= fixed_size and h >= fixed_size:
            return (w, h)
        else:
            ratio = fixed_size / w
            return (w * ratio, h * ratio)

    size = correct_size(width, height)
    size = correct_size(size[1], size[0])
    return (int(size[1]), int(size[0]))


FIXED_SIZE = 512 if args.size is None else args.size

with concurrent.futures.ThreadPoolExecutor(max_workers=6) as executor:
    futures = {}
    for (r, _, files) in os.walk(args.input_dir):
        for f in files:
            futures[executor.submit(resize_image, os.path.join(r, f), args.out_dir)] = f

    print('Number of resizing images {}'.format(len(futures.items())))

    for future in concurrent.futures.as_completed(futures):
        path = futures[future]
        try:
            future.result()
        except Exception as exc:
            print('%s generated as exception: %s' % (path, exc))
        else:
            print('%s is resized' % path)
