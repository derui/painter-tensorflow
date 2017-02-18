import os
import argparse
import numpy as np
import cv2 as cv

argparser = argparse.ArgumentParser(description='Extract edge layer of a color image')
argparser.add_argument('image', type=str,
                       help='the file name of image to extract edge layer')
argparser.add_argument('-d', dest='out_dir', type=str)

args = argparser.parse_args()

if args.image is None:
    print("No any files")
    exit(1)

img = cv.imread(args.image, cv.IMREAD_GRAYSCALE)
if img is None:
    print("OpenCV can not load %s" % (args.image))
    exit(2)

canny = cv.Canny(img, 50.0, 200.0, None, 3, True)

bitwise = np.copy(canny)

np.bitwise_not(canny, bitwise)

dirname, fname = os.path.split(os.path.abspath(args.image))
fname, ext = os.path.splitext(fname)
if args.out_dir is not None:
    if not os.path.exists(args.out_dir):
        os.mkdir(args.out_dir, 0o755)

    writefname = "%s/%s_converted%s" % (args.out_dir, fname, ext)
else:
    writefname = "%s/%s_converted%s" % (dirname, fname, ext)

cv.imwrite(writefname, bitwise)
print("Wrote image at %s" % (writefname))
