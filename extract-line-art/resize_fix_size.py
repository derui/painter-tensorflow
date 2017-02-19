import os
import argparse
import cv2 as cv

argparser = argparse.ArgumentParser(description='Resize image to fixed size')
argparser.add_argument(
    'image',
    type=str,
    help='the file name of image to extract edge layer')
argparser.add_argument('-d', dest='out_dir', type=str, required=True)
argparser.add_argument('-s', '--size', dest='size', type=int)

args = argparser.parse_args()

img = cv.imread(args.image)
if img is None:
    print("OpenCV can not load %s" % (args.image))
    exit(2)


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

correct_size = get_corrected_size(FIXED_SIZE, img.shape[1], img.shape[0])
img_resized = cv.resize(img, correct_size, interpolation=cv.INTER_CUBIC)

img_cropped = img_resized[0:FIXED_SIZE, 0:FIXED_SIZE]

dirname, fname = os.path.split(os.path.abspath(args.image))
fname, ext = os.path.splitext(fname)
if not os.path.exists(args.out_dir):
    os.mkdir(args.out_dir, 0o755)

writefname = "%s/%s%s" % (args.out_dir, fname, ext)

cv.imwrite(writefname, img_cropped)
print("Wrote image at %s" % (writefname))
