import os
import argparse
import numpy as np
import cv2 as cv
import concurrent.futures
import queue
from . import util

argparser = argparse.ArgumentParser(
    description='Extract edge layer of a color image')
argparser.add_argument(
    'input_dir',
    type=str,
    help='the directory included images to extract edge layer')
argparser.add_argument('-d', dest='out_dir', type=str, required=True)
argparser.add_argument('-s', '--size', dest='size', type=int)
argparser.add_argument('-e', dest='excludes_dir', type=str, required=True)

args = argparser.parse_args()

neiborhood8 = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]], np.uint8)
FIXED_SIZE = 512 if args.size is None else args.size


class Ignore(Exception):
    pass


def extract_edge(rq, wq):

    img, path = rq.get()
    rq.task_done()

    img_dilate = cv.dilate(img, neiborhood8, iterations=1)
    img_diff = cv.absdiff(img, img_dilate)
    img_diff_not = cv.bitwise_not(img_diff)
    img_diff_not = cv.cvtColor(img_diff_not, cv.COLOR_RGB2GRAY)
    img_diff_not = cv.adaptiveThreshold(img_diff_not, 255,
                                        cv.ADAPTIVE_THRESH_GAUSSIAN_C,
                                        cv.THRESH_BINARY, 5, 8)
    img_diff_not = cv.cvtColor(img_diff_not, cv.COLOR_GRAY2RGB)

    img = util.resize_image(img_diff_not, FIXED_SIZE)

    wq.put((img, path))


def read_image(path):
    img = cv.imread(path)
    if img is None:
        raise Exception("OpenCV can not load %s" % (path))
    return img


def write_image(info, out_dir):
    img, path = info

    dirname, fname = os.path.split(os.path.abspath(path))
    fname, ext = os.path.splitext(fname)
    d = os.path.join(out_dir, fname[0:2])
    if not os.path.exists(d):
        os.makedirs(d, 0o755, exist_ok=True)

    writefname = os.path.join(d, "{}{}".format(fname, ext))

    cv.imwrite(writefname, img)


excludes = []
for r, _, files in os.walk(args.excludes_dir):
    excludes.extend(files)

image_queue = queue.LifoQueue(100)
write_queue = queue.LifoQueue(300)

queue_reading_process = util.sequential_read_dir(
    args.input_dir, image_queue, read_image, excludes=excludes)
queue_writing_process = util.queue_writer(args.out_dir, write_queue,
                                          write_image)

readerExecutor = concurrent.futures.ThreadPoolExecutor(2)
read_thread = readerExecutor.submit(queue_reading_process)

writerExecutor = concurrent.futures.ThreadPoolExecutor(4)
write_thread = [
    util.queue_writer(args.out_dir, write_queue, write_image),
    util.queue_writer(args.out_dir, write_queue, write_image),
    util.queue_writer(args.out_dir, write_queue, write_image)
]

executor = concurrent.futures.ThreadPoolExecutor(max_workers=8)


def reader_process():
    num = 0

    while True:
        futures = []
        for _ in range(6):
            futures.append(
                executor.submit(extract_edge, image_queue, write_queue))

        for future in concurrent.futures.as_completed(futures):
            try:
                future.result()
            except Ignore:
                pass
            except Exception as exc:
                print('exception: %s' % exc)
            else:
                num += 1
                if num % 100 == 0:
                    print('Completed {} items'.format(num))


executor.submit(reader_process)

read_thread.result()

executor.shutdown()

write_queue.join()
