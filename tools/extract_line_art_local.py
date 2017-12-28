import argparse
import pathlib
from datetime import datetime
import numpy as np
import cv2 as cv
import concurrent.futures

argparser = argparse.ArgumentParser(description='Extract edge layer of a color image')
argparser.add_argument('input_dir', type=str, help='the directory contained images to process')
argparser.add_argument('-d', dest='output', type=str, required=True)
argparser.add_argument('-p', dest='parallel', type=int, default=8, help="number of task in parallel")

args = argparser.parse_args()

neiborhood8 = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]], np.uint8)


def extract_edge(img):

    img_dilate = cv.dilate(img, neiborhood8, iterations=1)
    img_diff = cv.absdiff(img, img_dilate)
    img_diff_not = cv.bitwise_not(img_diff)
    img_diff_not = cv.cvtColor(img_diff_not, cv.COLOR_RGB2GRAY)

    return img_diff_not


def read_image(filename):
    path = pathlib.Path(filename)

    img = cv.imread(str(path))
    if img is None:
        raise Exception("OpenCV can not load {}".format(path))
    return img, path


def write_image(dir_path, filename, img):

    if len(img.shape) == 2:
        img = cv.cvtColor(img, cv.COLOR_GRAY2RGB)

    out_path = pathlib.Path(dir_path, filename.name)
    out_path = out_path.with_suffix('.png')
    cv.imwrite(str(out_path), img)


def process(f, out_dir):

    filename = pathlib.Path(f)
    try:
        img, path = read_image(filename)
        img = extract_edge(img)
        write_image(out_dir, filename, img)

    except Exception as e:
        print(e)


if __name__ == "__main__":

    # Create a client
    path = pathlib.Path(args.input_dir)
    out_path = pathlib.Path(args.output)

    if not out_path.exists():
        out_path.mkdir(parents=True)

    print('{}: Start processing'.format(datetime.now()))

    num = 0
    ignored = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.parallel) as executor:
        futures = []
        for stat in path.glob("**/*"):
            if stat.is_dir():
                continue
            futures.append(executor.submit(process, stat, args.output))

        print('{}: Add all items to executor'.format(datetime.now()))
        for f in concurrent.futures.as_completed(futures):
            try:
                f.result()
            except Exception as ec:
                print("Exception raised: {}".format(ec))

            num += 1
            if num % 100 == 0:
                print("{}: Processed {} items".format(datetime.now(), num))

    print('{}: Completed {} items, {} ignored'.format(datetime.now(), num, ignored))
